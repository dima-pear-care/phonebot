"""Per-call cost tracking.

Accuracy tiers, highest available wins, provenance shown in the report:
    actual          — billed amount fetched from the provider (Twilio price)
    usage × rate    — provider-reported units (tokens, chars) × pricing.json
    measured × rate — locally measured units (stream seconds) × pricing.json
Models missing from pricing.json show UNKNOWN RATE instead of a guess.
"""

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger

_PRICING_FILE = Path("pricing.json")
try:
    PRICING: dict = json.loads(_PRICING_FILE.read_text(encoding="utf-8"))
except Exception as exc:
    logger.warning(f"Failed to load {_PRICING_FILE}: {exc} — costs will be UNKNOWN")
    PRICING = {}


@dataclass
class CostTracker:
    llm_provider: str = ""
    llm_model: str = ""
    stt_model: str = ""
    tts_model: str = ""
    duration_secs: float = 0.0
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cached_tokens: int = 0
    tts_chars: int = 0
    greeting_chars: int = 0
    el_subscription_tier: str | None = None     # auto-detected when key has user_read
    el_credits_start: float | None = None       # subscription character_count snapshots
    el_credits_end: float | None = None


def compute_costs(t: CostTracker) -> dict:
    """AI service costs only (STT + LLM + TTS); telephony intentionally excluded.

    Returns {'lines': [(label, detail, usd|None, provenance)], 'total': float, 'unknown': bool}.
    """
    lines: list[tuple[str, str, float | None, str]] = []
    unknown = False
    mins = t.duration_secs / 60 if t.duration_secs else 0.0

    # Deepgram STT — billed on streamed audio duration == connection duration
    stt_rate = PRICING.get("stt_usd_per_min", {}).get(t.stt_model)
    if stt_rate is not None:
        lines.append(("Deepgram STT", f"{int(t.duration_secs)}s · {t.stt_model}", round(mins * stt_rate, 6), "measured × rate"))
    else:
        lines.append(("Deepgram STT", f"{int(t.duration_secs)}s · {t.stt_model}", None, "UNKNOWN RATE"))
        unknown = True

    # LLM — provider-reported token usage (includes warmup call)
    llm_detail = (f"{t.llm_calls} calls · {t.llm_prompt_tokens:,} in / {t.llm_completion_tokens:,} out"
                  + (f" ({t.llm_cached_tokens:,} cached)" if t.llm_cached_tokens else ""))
    rates = PRICING.get("llm_per_mtok", {}).get(t.llm_provider, {}).get(t.llm_model)
    if rates:
        cached = min(t.llm_cached_tokens, t.llm_prompt_tokens)
        cached_rate = rates.get("cached_in", rates["in"])
        usd = ((t.llm_prompt_tokens - cached) * rates["in"] + cached * cached_rate
               + t.llm_completion_tokens * rates["out"]) / 1_000_000
        lines.append((f"LLM ({t.llm_provider})", llm_detail, round(usd, 6), "usage × rate"))
    else:
        lines.append((f"LLM ({t.llm_provider})", f"{llm_detail} · {t.llm_model}", None, "UNKNOWN RATE"))
        unknown = True

    # ElevenLabs TTS — chars submitted (billed even if playback interrupted) + greeting
    cpc = PRICING.get("tts_credits_per_char", {}).get(t.tts_model)
    chars = t.tts_chars + t.greeting_chars
    char_detail = f"{t.tts_chars} chars + {t.greeting_chars} greeting"
    if cpc is not None:
        credits = chars * cpc
        plan = t.el_subscription_tier or PRICING.get("elevenlabs_plan", "")
        per_credit = PRICING.get("elevenlabs_usd_per_credit", {}).get(plan)
        # Reconciliation against subscription counter (needs user_read API scope)
        recon = ""
        if t.el_credits_start is not None and t.el_credits_end is not None:
            delta = t.el_credits_end - t.el_credits_start
            recon = " ✓" if abs(delta - credits) <= max(2.0, credits * 0.02) else f" ⚠ provider says {delta:.0f}cr"
        if per_credit is not None:
            lines.append(("ElevenLabs TTS", f"{char_detail} · {credits:.0f}cr ({plan}){recon}",
                          round(credits * per_credit, 6), "usage × rate"))
        else:
            lines.append(("ElevenLabs TTS", f"{char_detail} · {credits:.0f}cr · plan '{plan}'{recon}", None, "UNKNOWN RATE"))
            unknown = True
    else:
        lines.append(("ElevenLabs TTS", f"{char_detail} · {t.tts_model}", None, "UNKNOWN RATE"))
        unknown = True

    total = round(sum(usd for _, _, usd, _ in lines if usd is not None), 6)
    return {"lines": lines, "total": total, "unknown": unknown}


def render_cost_block(t: CostTracker) -> str:
    c = compute_costs(t)
    out = ["", "──────── Cost ────────"]
    for label, detail, usd, prov in c["lines"]:
        usd_s = f"${usd:.4f}" if usd is not None else "?"
        out.append(f"{label:<16} {detail:<44} {usd_s:>9}  ({prov})")
    out.append(f"{'Total spent on this call':<61} ${c['total']:.4f}")
    if c["unknown"]:
        out.append("⚠ some rates missing from pricing.json — total is incomplete")
    return "\n".join(out) + "\n"


def save_cost_sidecar(json_path: Path, t: CostTracker) -> None:
    try:
        c = compute_costs(t)
        json_path.write_text(json.dumps({
            "usage": dataclasses.asdict(t),
            "costs": {
                "lines": [{"label": l, "detail": d, "usd": u, "provenance": p} for l, d, u, p in c["lines"]],
                "total_usd": c["total"],
                "unknown_rates": c["unknown"],
            },
            "pricing_as_of": PRICING.get("as_of"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Failed to save cost sidecar: {exc}")


_EL_SCOPE_WARNED = False


async def el_subscription_snapshot(tracker: CostTracker, which: str) -> None:
    """Reads ElevenLabs subscription counter for reconciliation. Needs user_read scope."""
    global _EL_SCOPE_WARNED
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")},
            )
            if r.status_code != 200:
                if not _EL_SCOPE_WARNED:
                    logger.info("ElevenLabs reconciliation disabled (API key lacks user_read scope)")
                    _EL_SCOPE_WARNED = True
                return
            data = r.json()
            count = float(data.get("character_count", 0))
            tracker.el_subscription_tier = data.get("tier")
            if which == "start":
                tracker.el_credits_start = count
            else:
                tracker.el_credits_end = count
    except Exception as exc:
        logger.warning(f"ElevenLabs subscription snapshot failed: {exc}")
