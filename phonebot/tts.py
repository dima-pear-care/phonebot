"""TTS text sanitizing.

The " -- " / " — " hesitation markup in our prompts is tuned for
eleven_multilingual_v2. On flash/turbo models the dashes stochastically cause
stutter-repeats, hallucinated words and multi-second silences (verified with
tts_compare.py; see guides/tts_artifact_debugging.md). Commas are the safe
pause marker for those models.
"""

import re

from loguru import logger
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService


def needs_dash_sanitizing(model: str) -> bool:
    return "flash" in model or "turbo" in model


def sanitize_dashes(text: str) -> str:
    text = re.sub(r"\s*(?:--+|—)\s*", ", ", text)
    text = re.sub(r",\s*,", ",", text)
    return text.lstrip(", ")


class SanitizingElevenLabsHttpTTSService(ElevenLabsHttpTTSService):
    """Drops [SILENT] utterances and strips dash markup on models that glitch on it."""

    async def run_tts(self, text: str, context_id: str):
        # The prompt tells the LLM to answer "[SILENT]" when it must not speak
        # (IVR menu playing, on hold, just pressed a digit) — an LLM cannot
        # output nothing, so silence is a token we filter out here.
        if "[silent" in text.lower():
            logger.debug(f"TTS suppressed silent utterance: [{text}]")
            return
        if needs_dash_sanitizing(self._settings.model or ""):
            clean = sanitize_dashes(text)
            if clean != text:
                logger.debug(f"TTS dash-sanitized: [{text}] → [{clean}]")
            text = clean
        async for frame in super().run_tts(text, context_id):
            yield frame
