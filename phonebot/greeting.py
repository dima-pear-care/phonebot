"""Pre-rendered greeting playback and IVR detection.

When a human answers, we want the opening line to play instantly without a
round-trip to the LLM, so the greeting audio is rendered ahead of time and
injected directly. When an IVR menu answers instead, the first transcription is
forwarded to the LLM so it can navigate. `GreetingGate` makes that human-vs-IVR
decision on the first utterance; `GreetingAudioInjector` turns the carried PCM
into output audio frames at the tail of the pipeline.
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from loguru import logger
from pipecat.frames.frames import Frame, OutputAudioRawFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from phonebot.costs import CostTracker


@dataclass
class PlayGreetingFrame(Frame):
    """Carries pre-rendered greeting PCM through the pipeline to GreetingAudioInjector."""
    audio: bytes = field(default_factory=bytes)


async def prerender_greeting(
    api_key: str, voice_id: str, model: str, text: str, tracker: "CostTracker | None" = None
) -> bytes:
    """Calls ElevenLabs REST API and returns greeting audio as raw PCM 8 kHz 16-bit mono."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            url,
            headers={"xi-api-key": api_key},
            params={"output_format": "pcm_8000"},
            json={"text": text, "model_id": model},
        )
        r.raise_for_status()
        if tracker is not None:
            tracker.greeting_chars += len(text)  # billed even if never played (IVR path)
        logger.info(f"Greeting pre-rendered: {len(r.content)} bytes")
        return r.content


class GreetingGate(FrameProcessor):
    """Intercepts the first transcription after call connect.

    Human answer  → plays pre-rendered greeting, injects it into context,
                    does NOT forward the TranscriptionFrame to the LLM.
    IVR detected  → forwards normally so the LLM handles navigation.
    Ambiguous     → forwards to the LLM, which says the opening line itself
                    (the greeting text is part of the system prompt).
    Fallback      → if pre-render isn't ready or fails, falls back to LLM.
    """

    _IVR_WORDS = {"press", "dial", "option", "options", "menu", "directory",
                  "extension", "department", "touchtone", "keypad", "recording"}
    # Strong phrases are recording-only language a live human never uses.
    _IVR_STRONG_PHRASES = ("please listen", "options have changed", "menu options",
                           "main menu", "after the tone", "after the beep",
                           "your call may be", "you've reached", "you have reached",
                           "para español")
    _IVR_WEAK_PHRASES = ("thank you for calling", "please hold", "please press")
    _HUMAN_PHRASES = ("this is", "speaking", "how can i help", "how may i help",
                      "what can i do", "who am i")
    _IVR_PRESS_RE = re.compile(
        r"\bpress\s+(?:a\s+)?(?:\d|one|two|three|four|five|six|seven|eight|nine|zero|star|pound)\b",
        re.IGNORECASE,
    )
    _IVR_LENGTH = 25       # words; IVR preambles are long
    _IVR_THRESHOLD = 2     # score needed to classify as IVR

    def __init__(
        self,
        context: LLMContext,
        greeting_text: str,
        render_task: asyncio.Task,
        dialogue: list,
    ) -> None:
        super().__init__()
        self._context      = context
        self._greeting     = greeting_text
        self._render_task  = render_task
        self._dialogue     = dialogue
        self._done         = False

    def _classify_answer(self, text: str) -> str:
        """Return ``ivr``, ``human``, or ``ambiguous`` for the first transcript.

        A misclassified human costs ~400ms (greeting falls through to the LLM);
        a missed IVR means speaking the greeting into a menu — so phrase
        signals lean toward IVR. Only explicit human language takes the direct,
        zero-LLM-latency greeting path; everything else is left to the LLM.
        """
        t = " ".join(text.lower().split())
        words = t.split()
        human_markers = sum(1 for p in self._HUMAN_PHRASES if p in t)
        score = 0
        if self._IVR_PRESS_RE.search(t):
            score += 2
        score += 2 * sum(1 for p in self._IVR_STRONG_PHRASES if p in t)
        score += sum(1 for p in self._IVR_WEAK_PHRASES if p in t)
        score += sum(1 for w in words if w.strip(",.?!") in self._IVR_WORDS)
        if len(words) >= self._IVR_LENGTH:
            score += 1
        score -= 2 * human_markers
        if score >= self._IVR_THRESHOLD:
            return "ivr"
        if human_markers:
            return "human"
        return "ambiguous"

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if (not self._done
                and direction == FrameDirection.DOWNSTREAM
                and isinstance(frame, TranscriptionFrame)
                and frame.text.strip()):
            self._done = True

            classification = self._classify_answer(frame.text)
            if classification == "ivr":
                logger.info(f"IVR detected — routing to LLM: '{frame.text[:80]}'")
                await self.push_frame(frame, direction)
                return
            if classification == "ambiguous":
                logger.info(f"Ambiguous answer — routing to LLM: '{frame.text[:80]}'")
                await self.push_frame(frame, direction)
                return

            # Human answered — try pre-rendered audio
            try:
                pcm = await asyncio.wait_for(asyncio.shield(self._render_task), timeout=3.0)
            except Exception as exc:
                logger.warning(f"Pre-render unavailable ({exc}), falling back to LLM")
                await self.push_frame(frame, direction)
                return

            logger.info("Playing pre-rendered greeting")

            # Inject greeting into context so LLM knows what the bot said
            self._context.messages.append({"role": "assistant", "content": self._greeting})

            # Record greeting in transcript dialogue
            self._dialogue.append({
                "role": "bot",
                "text": self._greeting,
                "ts": datetime.now(timezone.utc).isoformat(),
                "llm_ms": None,
            })

            # Send custom frame — GreetingAudioInjector converts it to audio at the end
            await self.push_frame(PlayGreetingFrame(audio=pcm), direction)
            return

        await self.push_frame(frame, direction)


class GreetingAudioInjector(FrameProcessor):
    """Placed just before transport.output(). Converts PlayGreetingFrame to audio chunks."""

    _CHUNK = 320  # 160 samples × 2 bytes = 20 ms at 8 kHz

    async def play(self, audio: bytes) -> None:
        """Push cached greeting audio as output frames."""
        for i in range(0, len(audio), self._CHUNK):
            chunk = audio[i:i + self._CHUNK]
            if len(chunk) < self._CHUNK:
                chunk += b"\x00" * (self._CHUNK - len(chunk))
            await self.push_frame(
                OutputAudioRawFrame(audio=chunk, sample_rate=8000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, PlayGreetingFrame):
            await self.play(frame.audio)
            return  # don't forward the PlayGreetingFrame itself
        await self.push_frame(frame, direction)
