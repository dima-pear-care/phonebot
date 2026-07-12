"""Live call monitoring: active-call registry, pipeline audio taps, auth check.

A `CallHandle` is registered per call and holds the pipeline task, the call's
Twilio client (for remote hangup) and the set of subscriber queues. Two
`AudioTap` processors publish caller/bot audio chunks to every subscriber.
Publishing is non-blocking with drop-oldest, so a slow or dead monitor can
never stall the call pipeline.

Wire format per chunk (what `/ws-monitor` sends as a binary message):
    1 byte channel (0 = caller, 1 = bot)
  + 4 bytes sample rate, uint32 little-endian
  + raw PCM 16-bit mono

Dialogue events (transcripts, bot replies, tool presses) are sent on the same
socket as JSON text messages: {"type": "dialogue", "role": ..., "text": ...,
"ts": ..., ...} — published automatically because each call's dialogue list is
a `MonitoredDialogue`.
"""

import asyncio
import hmac
import json
import os
import struct
from datetime import datetime, timezone

from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

CHANNEL_CALLER = 0
CHANNEL_BOT = 1

# 20 ms chunks at 8 kHz → ~500 chunks ≈ 10 s of backlog before we drop.
_QUEUE_MAX = 500

ACTIVE_CALLS: dict[str, "CallHandle"] = {}


def monitor_token_ok(token: str | None) -> bool:
    """Constant-time check against MONITOR_TOKEN; fails closed when unset."""
    expected = os.getenv("MONITOR_TOKEN", "")
    if not expected or not token:
        return False
    return hmac.compare_digest(token, expected)


class CallHandle:
    """Per-call monitoring state shared between the pipeline and the API routes."""

    def __init__(self, call_sid: str, twilio_client) -> None:
        self.call_sid = call_sid
        self.twilio_client = twilio_client
        self.started_at = datetime.now(timezone.utc)
        self.task = None  # PipelineTask, set by run_bot() once built
        self.subscribers: set[asyncio.Queue] = set()
        self._dropped = 0

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        self.subscribers.add(q)
        logger.info(f"Monitor attached to {self.call_sid} ({len(self.subscribers)} listener(s))")
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)
        logger.info(f"Monitor detached from {self.call_sid} ({len(self.subscribers)} listener(s))")

    def publish(self, channel: int, sample_rate: int, audio: bytes) -> None:
        if not self.subscribers:
            return
        self._enqueue(struct.pack("<BI", channel, sample_rate) + audio)

    def publish_event(self, payload: dict) -> None:
        """Send a JSON text message (dialogue entry / tool usage) to all subscribers."""
        if not self.subscribers:
            return
        self._enqueue(json.dumps(payload))

    def _enqueue(self, msg) -> None:
        for q in self.subscribers:
            if q.full():
                try:
                    q.get_nowait()  # drop oldest — keep the stream near-live
                except asyncio.QueueEmpty:
                    pass
                self._dropped += 1
                if self._dropped % 250 == 1:
                    logger.warning(f"Monitor subscriber lagging on {self.call_sid}; dropping audio")
            q.put_nowait(msg)

    def close(self) -> None:
        """Signal end-of-call to all subscribers (None sentinel)."""
        for q in self.subscribers:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(None)


class MonitoredDialogue(list):
    """A dialogue list that mirrors every appended entry to monitor subscribers.

    Everything that records dialogue (ClientTracker, BotTracker, GreetingGate,
    the DTMF handler) appends dicts to the call's dialogue list; routing those
    appends through this class gives live monitors the transcript and tool
    usage without touching any of the producers.
    """

    def __init__(self, handle: "CallHandle") -> None:
        super().__init__()
        self._handle = handle

    def append(self, entry: dict) -> None:
        super().append(entry)
        self._handle.publish_event({"type": "dialogue", **entry})


def register_call(call_sid: str, twilio_client) -> CallHandle:
    handle = CallHandle(call_sid, twilio_client)
    ACTIVE_CALLS[call_sid] = handle
    return handle


def unregister_call(call_sid: str) -> None:
    handle = ACTIVE_CALLS.pop(call_sid, None)
    if handle:
        handle.close()


class AudioTap(FrameProcessor):
    """Publishes pipeline audio frames to the call's monitor subscribers.

    channel=CHANNEL_CALLER taps InputAudioRawFrame (place after transport.input());
    channel=CHANNEL_BOT taps OutputAudioRawFrame (place just before transport.output(),
    after GreetingAudioInjector so the pre-rendered greeting is heard too).
    """

    def __init__(self, handle: CallHandle, channel: int) -> None:
        super().__init__()
        self._handle = handle
        self._channel = channel
        self._frame_type = InputAudioRawFrame if channel == CHANNEL_CALLER else OutputAudioRawFrame

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, self._frame_type) and frame.audio:
            self._handle.publish(self._channel, frame.sample_rate, frame.audio)
        await self.push_frame(frame, direction)
