#!/usr/bin/env python3
"""Scripted smoke call against /ws-test with human and IVR transfer scenarios.

Simulates the Twilio media-stream protocol: start event, caller audio
(mu-law 8k), then silence. Reports when bot audio arrives and how much.
"""

import asyncio
import audioop
import base64
import json
import os
import sys
import time

import httpx
import websockets
from dotenv import load_dotenv

load_dotenv()

WS_URL = "wss://phonebotpearai.online/ws-test"
SCENARIOS = {
    "human": [
        "Hello, doctor's office, who am I speaking with?",
        "Sure, can you spell the patient's last name for me, and give me the date of birth?",
    ],
    "ivr": [
        "Thank you for calling Smith Family Medicine. Please listen carefully, as our menu "
        "options have recently changed. For appointments and scheduling, press 1. For billing "
        "and insurance questions, press 2. For prescription refills, press 3. To speak with "
        "a nurse, press 4. To hear these options again, press 9.",
        "You have selected appointments and scheduling. Please hold while we connect you to "
        "the next available scheduler.",
        "Hello, scheduling department, this is Sarah. How may I help you?",
    ],
}
SCENARIO = "ivr" if "ivr" in sys.argv[1:] else "human"
CALLER_LINES = SCENARIOS[SCENARIO]


async def synth_caller_audio(text: str) -> bytes:
    r = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{os.environ['ELEVENLABS_VOICE_ID']}",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        params={"output_format": "pcm_8000"},
        json={"text": text, "model_id": "eleven_flash_v2_5"},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.content  # 16-bit PCM 8 kHz


async def main() -> None:
    lines = [await synth_caller_audio(t) for t in CALLER_LINES]
    print(f"caller audio: {[round(len(p) / 2 / 8000, 1) for p in lines]}s")

    async with websockets.connect(WS_URL) as ws:
        sid = "MZtest0000000000000000000000000000"
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({
            "event": "start",
            "streamSid": sid,
            "start": {"streamSid": sid, "callSid": "CAtest000000000000000000000000000000",
                      "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}},
        }))

        bot_audio = 0
        first_audio_at = None
        t0 = time.perf_counter()
        speech_end = None
        caller_state = "initial silence"

        async def send_pcm(pcm: bytes) -> None:
            started = time.perf_counter()
            for frame_no, i in enumerate(range(0, len(pcm), 320)):
                frame = pcm[i:i + 320]
                if len(frame) < 320:
                    frame += b"\x00" * (320 - len(frame))
                c = audioop.lin2ulaw(frame, 2)
                await ws.send(json.dumps({"event": "media", "streamSid": sid,
                                          "media": {"payload": base64.b64encode(c).decode()}}))
                deadline = started + (frame_no + 1) * 0.02
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

        async def send_silence(secs: float) -> None:
            silence = audioop.lin2ulaw(b"\x00" * 320, 2)
            started = time.perf_counter()
            for frame_no in range(int(secs * 50)):
                await ws.send(json.dumps({"event": "media", "streamSid": sid,
                                          "media": {"payload": base64.b64encode(silence).decode()}}))
                deadline = started + (frame_no + 1) * 0.02
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

        async def sender() -> None:
            nonlocal caller_state, speech_end
            await send_silence(2.0)
            caller_state = "IVR menu" if SCENARIO == "ivr" else "human receptionist"
            await send_pcm(lines[0])
            speech_end = time.perf_counter()
            print(f"caller line 1 finished at t={speech_end - t0:.1f}s")
            caller_state = "silence after line 1"
            await send_silence(9.0)
            caller_state = "hold announcement" if SCENARIO == "ivr" else "human receptionist"
            await send_pcm(lines[1])
            speech_end = time.perf_counter()
            print(f"caller line 2 finished at t={speech_end - t0:.1f}s")
            if len(lines) > 2:
                caller_state = "transfer delay"
                await send_silence(6.0)  # transfer delay before a human answers
                caller_state = "human receptionist after IVR"
                await send_pcm(lines[2])
                speech_end = time.perf_counter()
                print(f"caller line 3 (receptionist) finished at t={speech_end - t0:.1f}s")
            caller_state = "final silence"
            await send_silence(14.0)

        send_task = asyncio.create_task(sender())
        last_audio = None
        try:
            while time.perf_counter() - t0 < 80:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except (TimeoutError, asyncio.TimeoutError):
                    if send_task.done():
                        break
                    continue
                msg = json.loads(raw)
                if msg.get("event") == "media":
                    now = time.perf_counter()
                    if first_audio_at is None or (last_audio and now - last_audio > 2.0):
                        first_audio_at = now
                        print(f"BOT AUDIO BURST starts t={now - t0:.1f}s "
                              f"| caller state: {caller_state}")
                    last_audio = now
                    bot_audio += len(base64.b64decode(msg["media"]["payload"]))
        finally:
            send_task.cancel()

        print(f"total bot audio received: {bot_audio / 8000:.1f}s ({bot_audio} bytes mu-law)")


asyncio.run(main())
