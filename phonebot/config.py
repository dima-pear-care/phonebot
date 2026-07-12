"""Runtime configuration, paths, CLI flags, and logging setup.

`CONFIG` is the live settings dict (mutated in place by the dashboard via
`update_config`); `CONFIG_KEYS` is the allow-list of persisted keys. Module-level
code here runs at import: it parses `--transcript`, configures loguru, ensures the
calls directory exists, and overlays `config.json` onto the env-seeded defaults.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

from phonebot.prompts import DEFAULT_GREETING, DEFAULT_PROMPT

# ── CLI args ──────────────────────────────────────────────────────────────────
# parse_known_args so this is harmless under uvicorn (which has its own argv).

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--transcript", action="store_true")
_args, _ = _parser.parse_known_args()
TRANSCRIPT_LOG: bool = _args.transcript

# ── Logging ───────────────────────────────────────────────────────────────────

log_level = os.getenv("LOG_LEVEL", "INFO")
logger.remove()
logger.add(sys.stderr, level=log_level, colorize=True)

# ── Paths & storage ───────────────────────────────────────────────────────────

CALLS_DIR = Path("database/calls")
CALLS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = Path("config.json")

CONFIG_KEYS = (
    "system_prompt", "greeting", "llm_provider", "llm_model",
    "tts_model", "tts_voice_id", "stt_model", "stt_language", "recording",
)

CONFIG: dict = {
    "system_prompt": os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT),
    "greeting":      os.getenv("GREETING",       DEFAULT_GREETING),
    "llm_provider":  os.getenv("LLM_PROVIDER",   "groq"),
    "llm_model":     os.getenv("LLM_MODEL",      "llama-3.3-70b-versatile"),
    "tts_model":     os.getenv("TTS_MODEL",       "eleven_flash_v2_5"),
    "tts_voice_id":  os.getenv("ELEVENLABS_VOICE_ID", ""),
    "stt_model":     os.getenv("STT_MODEL",       "nova-2"),
    "stt_language":  os.getenv("STT_LANGUAGE",    "en"),
    "recording":     False,
}


def load_config() -> None:
    if not CONFIG_FILE.exists():
        return
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        for k in CONFIG_KEYS:
            if k in data and data[k]:
                CONFIG[k] = data[k]
    except Exception as exc:
        logger.warning(f"Failed to load {CONFIG_FILE}: {exc}")


def save_config() -> None:
    try:
        existing: dict = {}
        if CONFIG_FILE.exists():
            existing = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        for k in CONFIG_KEYS:
            existing[k] = CONFIG[k]
        CONFIG_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Failed to save {CONFIG_FILE}: {exc}")


load_config()
