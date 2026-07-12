"""Server-initiated outbound calls: Twilio REST dial-out + per-call prompt overrides.

`POST /outbound-call` places the call; the contact's doctor name and call context
are parked here keyed by call SID until the Twilio media stream connects and
`run_bot()` picks them up. Overrides customize that call's cfg snapshot only —
the global CONFIG / config.json are never touched, and the base system prompt
(with the AI disclosure rules) is never replaced, only extended.
"""

import asyncio
import time

from loguru import logger
from twilio.rest import Client as TwilioClient

# call_sid -> (registered_at, overrides). Entries for calls that never connect
# (busy / no answer / rejected) are pruned lazily after _TTL_SECS.
_OVERRIDES: dict[str, tuple[float, dict]] = {}
_TTL_SECS = 3600.0


def register_call_overrides(call_sid: str, overrides: dict) -> None:
    now = time.monotonic()
    for sid, (ts, _) in list(_OVERRIDES.items()):
        if now - ts > _TTL_SECS:
            del _OVERRIDES[sid]
    if overrides:
        _OVERRIDES[call_sid] = (now, overrides)


def pop_call_overrides(call_sid: str | None) -> dict | None:
    entry = _OVERRIDES.pop(call_sid or "", None)
    return entry[1] if entry else None


def apply_call_overrides(cfg: dict, overrides: dict) -> None:
    """Customize this call's cfg snapshot: doctor-name substitution + context block."""
    doctor = (overrides.get("doctor_name") or "").strip()
    if doctor:
        title_name = doctor if doctor.lower().startswith(("dr.", "dr ")) else f"Dr. {doctor}"
        cfg["system_prompt"] = cfg["system_prompt"].replace("Dr. Smith", title_name)
        cfg["greeting"] = cfg["greeting"].replace("Dr. Smith", title_name)
        cfg["system_prompt"] += f"\n\nFor this call, the doctor to ask about is {title_name}."
    patient = (overrides.get("patient_name") or "").strip()
    if patient:
        # "John Doe" is the placeholder patient in the default prompt/greeting.
        cfg["system_prompt"] = cfg["system_prompt"].replace("John Doe", patient)
        cfg["greeting"] = cfg["greeting"].replace("John Doe", patient)
    context = (overrides.get("call_context") or "").strip()
    if context:
        cfg["system_prompt"] += (
            "\n\nCall context — the actual patient for THIS call (overrides any default "
            "patient info above; do not volunteer these details, share one only if the "
            "office explicitly asks for it):\n" + context
        )
    # Per-call end-of-turn / interruption strategy (see build_turn_detection).
    turn_mode = (overrides.get("turn_mode") or "").strip().lower()
    if turn_mode in ("smart", "vad"):
        cfg["turn_mode"] = turn_mode


async def place_outbound_call(
    *,
    to: str,
    from_number: str,
    domain: str,
    account_sid: str,
    auth_token: str,
    overrides: dict | None = None,
) -> str:
    """Dial `to` from the Twilio number and connect the answered call to /ws."""
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="wss://{domain}/ws" /></Connect></Response>'
    )
    client = TwilioClient(account_sid, auth_token)
    call = await asyncio.to_thread(
        lambda: client.calls.create(to=to, from_=from_number, twiml=twiml)
    )
    register_call_overrides(call.sid, overrides or {})
    logger.info(f"Outbound call placed: {call.sid} -> {to}")
    return call.sid
