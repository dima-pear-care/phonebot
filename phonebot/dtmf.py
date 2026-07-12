"""DTMF tool definition and the press handler the LLM calls.

`DTMF_MODE=mock` (default) only logs the press; `DTMF_MODE=live` sends real
digits via the Twilio REST API. Both modes return the identical result payload
so the LLM's behavior doesn't change when switching.
"""

import time
from datetime import datetime, timezone

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema

_DUP_PRESS_WINDOW = 15.0  # seconds; same digits within this window = duplicate


def build_dtmf_tool() -> FunctionSchema:
    return FunctionSchema(
        name="press_dtmf",
        description=(
            "Press phone keypad buttons (DTMF tones) to navigate an automated "
            "phone menu (IVR). Use ONLY when a recorded/automated system asks "
            "to press digits — never while talking to a human."
        ),
        properties={
            "digits": {
                "type": "string",
                "description": "Digit(s) to press, e.g. '1'. Allowed: 0-9 * # w (w = 0.5s pause).",
            },
            "reason": {
                "type": "string",
                "description": "The menu option being selected, quoting the IVR wording.",
            },
        },
        required=["digits", "reason"],
    )


def make_press_handler(twilio_client, call_sid, dialogue: list, dtmf_mode: str):
    """Build the `press_dtmf` function handler bound to this call's state."""
    last_press = {"digits": None, "t": 0.0}

    async def press_dtmf_handler(params) -> None:
        digits = str(params.arguments.get("digits", "")).strip()
        reason = str(params.arguments.get("reason", "")).strip()
        if not digits or any(c not in "0123456789*#w" for c in digits):
            logger.warning(f"DTMF rejected invalid digits: {digits!r}")
            await params.result_callback(
                {"success": False, "error": "invalid digits — allowed characters: 0-9 * # w"}
            )
            return
        now = time.monotonic()
        if digits == last_press["digits"] and now - last_press["t"] < _DUP_PRESS_WINDOW:
            # Smart-turn can split one long menu into several turns, making the
            # LLM re-decide; don't send the same digits twice on a real call.
            logger.info(f"DTMF duplicate press '{digits}' suppressed")
            await params.result_callback({"success": True, "pressed": digits})
            return
        last_press["digits"], last_press["t"] = digits, now
        dialogue.append({
            "role": "dtmf",
            "text": f"pressed '{digits}' — {reason}" + (" [mock]" if dtmf_mode != "live" else ""),
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        if dtmf_mode == "live":
            logger.info(f"DTMF → pressing '{digits}' | reason: {reason}")
            try:
                twilio_client.calls(call_sid).update(send_digits=digits)
            except Exception as exc:
                logger.error(f"DTMF error: {exc}")
                await params.result_callback({"success": False, "error": str(exc)})
                return
        else:
            logger.info(f"[DTMF MOCK] would press '{digits}' | reason: {reason}")
        await params.result_callback({"success": True, "pressed": digits})

    return press_dtmf_handler
