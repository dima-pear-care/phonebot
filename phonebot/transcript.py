"""Transcript rendering, on-disk persistence, and Telegram notification."""

import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from loguru import logger

from phonebot.config import CALLS_DIR
from phonebot.costs import CostTracker, render_cost_block, save_cost_sidecar

_KYIV = ZoneInfo("Europe/Kiev")
_SEP  = "─" * 64


def to_kyiv(iso: str) -> datetime:
    return datetime.fromisoformat(iso).astimezone(_KYIV)


def save_transcript(
    started_at: str, ended_at: str, dialogue: list[dict], tracker: CostTracker | None = None
) -> tuple[str, Path]:
    try:
        t0 = to_kyiv(started_at)
        t1 = to_kyiv(ended_at)
        secs = int((t1 - t0).total_seconds())
        duration = f"{secs // 60}m {secs % 60}s"
        fname = t0.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    except Exception:
        fname = f"call_{int(time.time())}.txt"
        t0 = t1 = None
        duration = "unknown"

    if t0 and t1:
        header_time = f"{t0.strftime('%d %b %Y')}   {t0.strftime('%H:%M:%S')} – {t1.strftime('%H:%M:%S')} Kyiv   ({duration})"
    else:
        header_time = "unknown time"

    lines = [
        _SEP,
        f"  {header_time}",
        _SEP,
        "",
    ]

    for entry in dialogue:
        try:
            ts = to_kyiv(entry["ts"]).strftime("%H:%M:%S")
        except Exception:
            ts = "--:--:--"
        role = {"bot": "Bot  ", "client": "Human", "dtmf": "DTMF "}.get(entry["role"], "?    ")
        lines.append(f"[{ts}]  {role}  │  {entry['text']}")
        if entry["role"] == "bot" and entry.get("llm_ms") is not None:
            lines.append(f"            ↳ llm={entry['llm_ms']}ms")

    text = "\n".join(lines) + "\n"
    txt_path = CALLS_DIR / fname
    if tracker is not None:
        text += render_cost_block(tracker)
    txt_path.write_text(text, encoding="utf-8")
    if tracker is not None:
        save_cost_sidecar(txt_path.with_suffix(".json"), tracker)
    logger.info(f"Transcript saved: {fname} ({len(dialogue)} lines)")
    return text, txt_path


async def send_telegram(text: str) -> None:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram hard limit is 4096 chars; split if needed
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
    async with httpx.AsyncClient(timeout=15.0) as client:
        for chunk in chunks:
            try:
                await client.post(url, json={"chat_id": chat_id, "text": chunk})
            except Exception as exc:
                logger.warning(f"Telegram send failed: {exc}")
