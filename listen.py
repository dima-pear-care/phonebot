#!/usr/bin/env python3
"""
Live call monitor — follow an active call on the phonebot server from this PC.

Default mode shows the call as a live transcript (caller/bot lines and tool
usage such as DTMF presses) and plays the CALLER's audio through the speakers
in real time (bot lines are shown as text only — bot audio arrives in bursts
and is only faithful in the recording). Both audio channels are recorded; when
the call ends a stereo WAV (caller L / bot R) is saved to recordings/. Recorded
chunks are placed on a wall-clock timeline, so silences and response latency
appear exactly as they happened, and nothing is dropped (no realtime pacing).

--live instead plays the call audio through the local speakers in real time
(caller LEFT, bot RIGHT; --mono mixes them). Note live playback can drop bursty
bot audio to stay near-live — use the recording for faithful audio.

Hotkeys in both modes: [h] hang up the call, [q] quit (stop & save).

Usage:
    python listen.py                  # attach to the single active call, or pick one
    python listen.py CAxxxx...        # attach to a specific call SID
    python listen.py --follow         # wait for the next call and attach automatically
    python listen.py --hangup CAxxxx  # terminate a call without listening
    python listen.py --live           # realtime speaker playback instead of
                                      # transcript + recording

Config: MONITOR_TOKEN from .env (must match the server's), server URL from
--server / PHONEBOT_SERVER env (default https://phonebotpearai.online).

Requirements (local only, not in requirements.txt):
    pip install sounddevice websockets numpy httpx
"""

import argparse
import asyncio
import json
import os
import struct
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

# Console output uses box-drawing/meter glyphs; Windows redirects default to
# cp1252 which cannot encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import httpx
    import numpy as np
    import sounddevice as sd
    import websockets
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install sounddevice websockets numpy httpx")

try:
    import msvcrt  # Windows hotkeys
except ImportError:
    msvcrt = None

DEFAULT_SERVER = os.getenv("PHONEBOT_SERVER", "https://phonebotpearai.online")
DEVICE_RATE = 48000          # output stream rate; all incoming audio is resampled to this
MAX_BUFFER_SECS = 1.0        # if a channel lags more than this, drop to CATCHUP_SECS
CATCHUP_SECS = 0.2

CHANNEL_CALLER = 0
CHANNEL_BOT = 1


# ── audio plumbing ────────────────────────────────────────────────────────────

def _resample(x: "np.ndarray", src: int, dst: int) -> "np.ndarray":
    if src == dst:
        return x
    n_out = int(len(x) * dst / src)
    return np.interp(
        np.linspace(0, len(x), n_out, endpoint=False), np.arange(len(x)), x
    ).astype(np.float32)


class ChannelBuffer:
    """Thread-safe FIFO of float32 samples: WS reader writes, audio callback reads."""

    def __init__(self) -> None:
        self._chunks: deque = deque()
        self._offset = 0          # consumed samples in the head chunk
        self._total = 0
        self._lock = threading.Lock()
        self.level = 0.0          # last-chunk RMS, for the meter

    def write(self, pcm16: bytes, src_rate: int) -> None:
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.level = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
        samples = _resample(samples, src_rate, DEVICE_RATE)
        with self._lock:
            self._chunks.append(samples)
            self._total += len(samples)
            # catch-up: drop backlog so playback stays near-live
            if self._total > MAX_BUFFER_SECS * DEVICE_RATE:
                while self._total > CATCHUP_SECS * DEVICE_RATE and len(self._chunks) > 1:
                    dropped = self._chunks.popleft()
                    self._total -= len(dropped) - self._offset
                    self._offset = 0

    def read(self, n: int) -> "np.ndarray":
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                head = self._chunks[0]
                avail = len(head) - self._offset
                take = min(avail, n - filled)
                out[filled:filled + take] = head[self._offset:self._offset + take]
                filled += take
                self._offset += take
                self._total -= take
                if self._offset >= len(head):
                    self._chunks.popleft()
                    self._offset = 0
        return out


# ── server API ────────────────────────────────────────────────────────────────

def _headers(token: str) -> dict:
    return {"X-Monitor-Token": token}


async def fetch_calls(server: str, token: str) -> list:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{server}/calls", headers=_headers(token))
        if r.status_code == 403:
            sys.exit("Server rejected MONITOR_TOKEN (403) — check .env on both ends.")
        r.raise_for_status()
        return r.json()


async def hangup_call(server: str, token: str, call_sid: str) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(f"{server}/calls/{call_sid}/hangup", headers=_headers(token))
        if r.status_code == 404:
            print(f"\nNo active call {call_sid} (already ended?)")
            return
        r.raise_for_status()
        print(f"\nHangup sent ({r.json().get('method')}) — waiting for the call to end…")


# ── UI helpers ────────────────────────────────────────────────────────────────

_METER_BLOCKS = " ▁▂▃▄▅▆▇█"


def _meter(level: float) -> str:
    # speech RMS is roughly 0.02–0.3; map log-ish onto 8 blocks
    idx = min(int(level * 40), len(_METER_BLOCKS) - 1)
    return _METER_BLOCKS[idx] * 2 if idx else "  "


def pick_call(calls: list) -> str:
    print("Active calls:")
    for i, c in enumerate(calls, 1):
        print(f"  [{i}] {c['call_sid']}  ({c['duration_secs']:.0f}s, {c['listeners']} listener(s))")
    while True:
        choice = input(f"Attach to [1-{len(calls)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(calls):
            return calls[int(choice) - 1]["call_sid"]


# ── main listen loop ──────────────────────────────────────────────────────────

async def listen(server: str, token: str, call_sid: str, mono: bool) -> None:
    ws_scheme_url = server.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    ws_url = f"{ws_scheme_url}/ws-monitor/{call_sid}?token={quote(token)}"

    buffers = {CHANNEL_CALLER: ChannelBuffer(), CHANNEL_BOT: ChannelBuffer()}

    def audio_callback(outdata, frames, time_info, status) -> None:
        caller = buffers[CHANNEL_CALLER].read(frames)
        bot = buffers[CHANNEL_BOT].read(frames)
        if mono:
            mix = (caller + bot) * 0.8
            outdata[:, 0] = mix
            outdata[:, 1] = mix
        else:
            outdata[:, 0] = caller
            outdata[:, 1] = bot

    quit_requested = asyncio.Event()

    async def ui_loop() -> None:
        started = time.monotonic()
        while not quit_requested.is_set():
            if msvcrt:
                while msvcrt.kbhit():
                    ch = msvcrt.getwch().lower()
                    if ch == "h":
                        await hangup_call(server, token, call_sid)
                    elif ch == "q":
                        quit_requested.set()
                        return
            elapsed = int(time.monotonic() - started)
            line = (
                f"\r  {call_sid[:20]}…  {elapsed // 60:02d}:{elapsed % 60:02d}"
                f"  caller {_meter(buffers[CHANNEL_CALLER].level)}"
                f"  bot {_meter(buffers[CHANNEL_BOT].level)}"
                f"   [h] hang up  [q] quit  "
            )
            print(line, end="", flush=True)
            await asyncio.sleep(0.05)

    async with websockets.connect(ws_url, max_size=None) as ws:
        handshake = await ws.recv()  # JSON text: {call_sid, started_at}
        print(f"Attached to {call_sid} ({handshake})")
        if not msvcrt:
            print("(hotkeys unavailable on this platform — Ctrl+C to quit)")

        with sd.OutputStream(
            samplerate=DEVICE_RATE, channels=2, dtype="float32", callback=audio_callback
        ):
            ui_task = asyncio.create_task(ui_loop())
            try:
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)) and len(msg) > 5:
                        channel, rate = struct.unpack_from("<BI", msg)
                        if channel in buffers:
                            buffers[channel].write(bytes(msg[5:]), rate)
                    if quit_requested.is_set():
                        break
            except websockets.ConnectionClosed as exc:
                if exc.code == 4403:
                    sys.exit("\nServer rejected MONITOR_TOKEN — check .env on both ends.")
                if exc.code == 4404:
                    print("\nNo active call with that SID.")
            finally:
                ui_task.cancel()

    print("\nCall ended." if not quit_requested.is_set() else "\nDetached (call continues).")


# ── transcript + record mode (default) ────────────────────────────────────────

RECORD_RATE = 8000
RECORDINGS_DIR = Path("recordings")

_ROLE_LABELS = {"client": "Caller", "bot": "Bot", "dtmf": "DTMF"}


def _print_event(event: dict) -> None:
    role = _ROLE_LABELS.get(event.get("role", ""), event.get("role", "?"))
    try:
        ts = datetime.fromisoformat(event["ts"]).astimezone().strftime("%H:%M:%S")
    except (KeyError, ValueError):
        ts = datetime.now().strftime("%H:%M:%S")
    suffix = f"   (llm {event['llm_ms']}ms)" if event.get("llm_ms") else ""
    print(f"[{ts}]  {role:<6} │ {event.get('text', '')}{suffix}")


async def record(server: str, token: str, call_sid: str) -> None:
    """Live transcript + realtime caller audio, while recording to a stereo WAV.

    Dialogue/tool events arrive as JSON text messages and are printed as they
    happen; the caller channel is also played through the speakers (it arrives
    paced at 1x, so playback never falls behind). Bot audio is recorded only —
    it arrives in bursts faster than realtime, which live playback can't
    represent faithfully. The stream is drained as fast as it arrives, so the
    server's drop-oldest monitor queues never overflow — nothing is lost. Each
    recorded chunk is placed at its wall-clock arrival offset (never
    overlapping the previous chunk on the same channel), so gaps and response
    latency in the WAV match the real call timeline.
    """
    ws_scheme_url = server.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    ws_url = f"{ws_scheme_url}/ws-monitor/{call_sid}?token={quote(token)}"

    # per channel: list of (start_sample, int16 samples), plus current end position
    tracks: dict[int, list] = {CHANNEL_CALLER: [], CHANNEL_BOT: []}
    ends = {CHANNEL_CALLER: 0, CHANNEL_BOT: 0}
    quit_requested = asyncio.Event()

    caller_buffer = ChannelBuffer()  # realtime speaker playback of the caller side

    def audio_callback(outdata, frames, time_info, status) -> None:
        mono = caller_buffer.read(frames)
        outdata[:, 0] = mono
        outdata[:, 1] = mono

    async def key_loop() -> None:
        while not quit_requested.is_set():
            if msvcrt:
                while msvcrt.kbhit():
                    ch = msvcrt.getwch().lower()
                    if ch == "h":
                        await hangup_call(server, token, call_sid)
                    elif ch == "q":
                        quit_requested.set()
                        return
            await asyncio.sleep(0.1)

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.recv()  # handshake JSON: {call_sid, started_at}
        print(f"Attached to {call_sid} — transcript below; caller audible; recording."
              f"{'   [h] hang up  [q] stop & save' if msvcrt else ''}")
        t0 = time.monotonic()
        key_task = asyncio.create_task(key_loop())
        with sd.OutputStream(
            samplerate=DEVICE_RATE, channels=2, dtype="float32", callback=audio_callback
        ):
            try:
                async for msg in ws:
                    if quit_requested.is_set():
                        break
                    if isinstance(msg, str):
                        try:
                            event = json.loads(msg)
                        except ValueError:
                            continue
                        if event.get("type") == "dialogue":
                            _print_event(event)
                        continue
                    if len(msg) <= 5:
                        continue
                    channel, rate = struct.unpack_from("<BI", msg)
                    if channel not in tracks:
                        continue
                    if channel == CHANNEL_CALLER:
                        caller_buffer.write(bytes(msg[5:]), rate)
                    samples = np.frombuffer(bytes(msg[5:]), dtype=np.int16)
                    if rate != RECORD_RATE:
                        samples = _resample(
                            samples.astype(np.float32) / 32768.0, rate, RECORD_RATE
                        )
                        samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                    pos = max(ends[channel], int((time.monotonic() - t0) * RECORD_RATE))
                    tracks[channel].append((pos, samples))
                    ends[channel] = pos + len(samples)
            except websockets.ConnectionClosed as exc:
                if exc.code == 4403:
                    sys.exit("\nServer rejected MONITOR_TOKEN — check .env on both ends.")
                if exc.code == 4404:
                    print("\nNo active call with that SID.")
                    return
            finally:
                key_task.cancel()

    length = max(ends.values())
    if not length:
        print("\nNo audio received — nothing saved.")
        return

    stereo = np.zeros((length, 2), dtype=np.int16)
    for channel, chunks in tracks.items():
        for pos, samples in chunks:
            stereo[pos:pos + len(samples), channel] = samples

    RECORDINGS_DIR.mkdir(exist_ok=True)
    path = RECORDINGS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{call_sid}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RECORD_RATE)
        w.writeframes(stereo.tobytes())
    print(f"\nSaved {length / RECORD_RATE:.1f}s (caller L / bot R) -> {path.resolve()}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Listen to / terminate active phonebot calls.")
    parser.add_argument("call_sid", nargs="?", help="call SID to attach to")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"server base URL (default {DEFAULT_SERVER})")
    parser.add_argument("--follow", action="store_true", help="wait for the next call and attach automatically")
    parser.add_argument("--hangup", metavar="SID", help="terminate this call and exit (no listening)")
    parser.add_argument("--mono", action="store_true", help="mix caller+bot instead of stereo L/R split")
    parser.add_argument("--live", action="store_true",
                        help="realtime speaker playback instead of transcript + recording")
    args = parser.parse_args()

    server = args.server.rstrip("/")
    token = os.getenv("MONITOR_TOKEN", "")
    if not token:
        sys.exit("MONITOR_TOKEN is not set in .env — generate one and add it on both server and here.")

    if args.hangup:
        await hangup_call(server, token, args.hangup)
        return

    while True:
        call_sid = args.call_sid
        if not call_sid:
            calls = await fetch_calls(server, token)
            if not calls and args.follow:
                print("\rWaiting for a call…  (Ctrl+C to quit)", end="", flush=True)
                await asyncio.sleep(2.0)
                continue
            if not calls:
                sys.exit("No active calls. (Use --follow to wait for one.)")
            call_sid = calls[0]["call_sid"] if len(calls) == 1 else pick_call(calls)

        print()
        if args.live:
            await listen(server, token, call_sid, args.mono)
        else:
            await record(server, token, call_sid)

        if not args.follow:
            break
        args.call_sid = None  # go back to waiting for the next call


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")
