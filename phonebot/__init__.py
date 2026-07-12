"""AI Phone Bot — Twilio + Deepgram STT + ElevenLabs TTS + Groq/Fireworks LLM.

Powered by Pipecat. `load_dotenv()` runs here, before any submodule is imported,
so modules that read API keys / settings from the environment at import time (e.g.
`phonebot.config`) see a populated `.env` no matter which entry point — the server
shim or the local `test.py` harness — pulls the package in first.

Kept deliberately light: importing a single submodule (e.g. `phonebot.prompts`)
should not drag in FastAPI or the whole pipeline. Import the app via
`from phonebot.app import app` and the runner via `from phonebot.pipeline import run_bot`.
"""

from dotenv import load_dotenv

load_dotenv()
