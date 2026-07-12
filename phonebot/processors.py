"""Generic pipeline frame processors.

Each captures or transforms frames as they flow through the Pipecat pipeline:
usage accounting, dialogue capture, per-turn latency timing, trailing-silence
padding, and optional live transcript logging.
"""

import dataclasses
import time
from datetime import datetime, timezone

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSStoppedFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from phonebot.costs import CostTracker


class UsageCollector(FrameProcessor):
    """Accumulates LLM token usage and TTS character usage from MetricsFrames."""

    def __init__(self, tracker: CostTracker) -> None:
        super().__init__()
        self._tracker = tracker

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, LLMUsageMetricsData):
                    u = d.value
                    self._tracker.llm_calls += 1
                    self._tracker.llm_prompt_tokens += u.prompt_tokens
                    self._tracker.llm_completion_tokens += u.completion_tokens
                    self._tracker.llm_cached_tokens += u.cache_read_input_tokens or 0
                elif isinstance(d, TTSUsageMetricsData):
                    self._tracker.tts_chars += d.value
        await self.push_frame(frame, direction)


class ClientTracker(FrameProcessor):
    """Captures TranscriptionFrames before the user aggregator swallows them."""

    def __init__(self, dialogue: list) -> None:
        super().__init__()
        self._dialogue = dialogue

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._dialogue.append({
                "role": "client",
                "text": frame.text.strip(),
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        await self.push_frame(frame, direction)


class BotTracker(FrameProcessor):
    """Captures assembled LLM responses after the LLM emits them."""

    def __init__(self, dialogue: list, timing: dict) -> None:
        super().__init__()
        self._dialogue = dialogue
        self._timing = timing
        self._buf: list[str] = []
        self._in_response = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._buf = []
                self._in_response = True
            elif isinstance(frame, LLMTextFrame) and self._in_response:
                self._buf.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame) and self._in_response:
                self._in_response = False
                text = "".join(self._buf).strip()
                if text:
                    self._dialogue.append({
                        "role": "bot",
                        "text": text,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "llm_ms": self._timing.pop("llm_ms", None),
                    })
        await self.push_frame(frame, direction)


class LLMInputTimer(FrameProcessor):
    def __init__(self, shared: dict) -> None:
        super().__init__()
        self._shared = shared

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            self._shared["t_in"] = time.perf_counter()
        await self.push_frame(frame, direction)


class LLMOutputTimer(FrameProcessor):
    def __init__(self, shared: dict) -> None:
        super().__init__()
        self._shared = shared
        self._logged = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._logged = False
            elif isinstance(frame, LLMTextFrame) and not self._logged:
                t_in = self._shared.get("t_in")
                if t_in is not None:
                    ms = int((time.perf_counter() - t_in) * 1000)
                    logger.info(f"LLM first-token latency: {ms} ms")
                    self._shared["llm_ms"] = ms
                self._logged = True
        await self.push_frame(frame, direction)


class AudioPadder(FrameProcessor):
    """Append a few silence frames after TTS ends to avoid abrupt cutoff."""

    _FRAME_BYTES  = 160 * 2
    _TRAIL_FRAMES = 4

    def __init__(self) -> None:
        super().__init__()
        self._last_audio_frame = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, OutputAudioRawFrame):
                self._last_audio_frame = frame
            elif isinstance(frame, TTSStoppedFrame) and self._last_audio_frame is not None:
                silence = b"\x00" * self._FRAME_BYTES
                for _ in range(self._TRAIL_FRAMES):
                    await self.push_frame(
                        dataclasses.replace(self._last_audio_frame, audio=silence),
                        direction,
                    )
                self._last_audio_frame = None
        await self.push_frame(frame, direction)


class TranscriptLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            logger.opt(colors=True).info(
                f"<cyan><bold>[TRANSCRIPT]</bold></cyan> <white>{frame.text}</white>"
            )
        await self.push_frame(frame, direction)
