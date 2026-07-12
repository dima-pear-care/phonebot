"""FastAPI app: HTTP routes (health, dashboard, config) and the Twilio WebSocket."""

import asyncio
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from twilio.base.exceptions import TwilioRestException
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VoiceGrant
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse

from phonebot.config import CONFIG, CONFIG_KEYS, save_config
from phonebot.monitor import ACTIVE_CALLS, monitor_token_ok
from phonebot.outbound import place_outbound_call
from phonebot.pipeline import run_bot

app = FastAPI(title="AI Phone Bot")
app.mount("/static", StaticFiles(directory="static"), name="static")


def _require_monitor_token(request: Request) -> None:
    token = request.headers.get("x-monitor-token") or request.query_params.get("token")
    if not monitor_token_ok(token):
        raise HTTPException(status_code=403, detail="invalid or missing monitor token")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def dashboard() -> HTMLResponse:
    html = Path("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/config")
async def get_config() -> dict:
    return {k: CONFIG[k] for k in CONFIG_KEYS}


@app.post("/config")
async def update_config(request: Request) -> dict:
    body = await request.json()
    for k, v in body.items():
        if k not in CONFIG_KEYS:
            continue
        if k == "recording":
            CONFIG[k] = bool(v) if isinstance(v, bool) else str(v).lower() == "true"
        else:
            CONFIG[k] = v
    save_config()
    return {k: CONFIG[k] for k in CONFIG_KEYS}


def _inbound_enabled() -> bool:
    """Whether real inbound phone calls to the Twilio number reach the bot.

    Off by default: this deployment only places outbound calls (dial.py /
    /outbound-call) plus the /ws-test testing stream. Flip INBOUND_ENABLED=true
    to answer inbound calls again.
    """
    return os.getenv("INBOUND_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


@app.post("/incoming-call")
async def incoming_call(request: Request) -> HTMLResponse:
    if not _inbound_enabled():
        logger.info("Inbound call rejected — inbound is disabled (set INBOUND_ENABLED=true to allow)")
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Response><Reject reason="rejected"/></Response>'
        )
        return HTMLResponse(content=twiml, media_type="application/xml")
    host = request.headers.get("host", os.getenv("DOMAIN", "localhost"))
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/ws" />
    </Connect>
</Response>"""
    return HTMLResponse(content=twiml, media_type="application/xml")


def _required_env(*names: str, detail: str = "Browser dialer is not configured") -> dict[str, str]:
    values = {name: os.getenv(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        raise HTTPException(status_code=503, detail=detail)
    return values


def _masked_number(number: str) -> str:
    return f"{'*' * max(0, len(number) - 4)}{number[-4:]}"


def _public_request_url(request: Request) -> str:
    """Rebuild the URL Twilio signed before Caddy terminated TLS."""
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip()
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    path = request.url.path
    if request.url.query:
        path += f"?{request.url.query}"
    return f"{scheme}://{host}{path}"


@app.get("/api/twilio-token")
async def twilio_token(request: Request) -> dict:
    """Issue a short-lived token that can only use the configured TwiML App."""
    dialer_password = os.getenv("DIALER_PASSWORD", "")
    supplied_password = request.headers.get("x-dialer-password", "")
    if not dialer_password:
        raise HTTPException(status_code=503, detail="DIALER_PASSWORD is not configured")
    if not secrets.compare_digest(supplied_password, dialer_password):
        raise HTTPException(status_code=401, detail="Invalid dialer password")

    env = _required_env(
        "TWILIO_ACCOUNT_SID",
        "TWILIO_API_KEY_SID",
        "TWILIO_API_KEY_SECRET",
        "TWILIO_TWIML_APP_SID",
        "BOT_PHONE_NUMBER",
        "TWILIO_CALLER_ID",
    )
    identity = f"browser-{secrets.token_hex(8)}"
    token = AccessToken(
        env["TWILIO_ACCOUNT_SID"],
        env["TWILIO_API_KEY_SID"],
        env["TWILIO_API_KEY_SECRET"],
        identity=identity,
        ttl=600,
    )
    token.add_grant(VoiceGrant(outgoing_application_sid=env["TWILIO_TWIML_APP_SID"]))
    return {
        "token": token.to_jwt(),
        "callerId": _masked_number(env["TWILIO_CALLER_ID"]),
        "destination": _masked_number(env["BOT_PHONE_NUMBER"]),
    }


@app.post("/api/twilio-outbound-call")
async def twilio_outbound_call(request: Request) -> Response:
    """TwiML App webhook: bridge the browser client to the fixed bot number."""
    env = _required_env("TWILIO_AUTH_TOKEN", "BOT_PHONE_NUMBER", "TWILIO_CALLER_ID")
    body = await request.body()
    params = dict(parse_qsl(body.decode("utf-8"), keep_blank_values=True))
    signature = request.headers.get("x-twilio-signature", "")
    if not RequestValidator(env["TWILIO_AUTH_TOKEN"]).validate(
        _public_request_url(request), params, signature
    ):
        logger.warning("Rejected browser-call webhook with an invalid Twilio signature")
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    response = VoiceResponse()
    dial = response.dial(caller_id=env["TWILIO_CALLER_ID"], answer_on_bridge=True)
    dial.number(env["BOT_PHONE_NUMBER"])
    return Response(content=str(response), media_type="application/xml")


@app.post("/outbound-call")
async def outbound_call(request: Request) -> dict:
    """Place a bot call to an arbitrary number (requires MONITOR_TOKEN).

    Body: {"to": "+15551234567", "doctor_name": "...", "call_context": "...",
    "turn_mode": "smart"|"vad"} — doctor_name / patient_name / call_context are
    optional per-call prompt overrides; turn_mode selects end-of-turn detection.
    """
    _require_monitor_token(request)
    body = await request.json()
    to = str(body.get("to", "")).strip()
    if not re.fullmatch(r"\+\d{8,15}", to):
        raise HTTPException(status_code=400, detail="'to' must be E.164, e.g. +15551234567")
    env = _required_env(
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_CALLER_ID", "DOMAIN",
        detail="Outbound calling is not configured (need TWILIO_* + DOMAIN)",
    )
    overrides = {
        "doctor_name": str(body.get("doctor_name") or ""),
        "patient_name": str(body.get("patient_name") or ""),
        "call_context": str(body.get("call_context") or ""),
        "turn_mode": str(body.get("turn_mode") or ""),
    }
    try:
        call_sid = await place_outbound_call(
            to=to,
            from_number=env["TWILIO_CALLER_ID"],
            domain=env["DOMAIN"],
            account_sid=env["TWILIO_ACCOUNT_SID"],
            auth_token=env["TWILIO_AUTH_TOKEN"],
            overrides=overrides,
        )
    except TwilioRestException as exc:
        raise HTTPException(status_code=502, detail=f"Twilio: {exc.msg}")
    return {"call_sid": call_sid, "to": to, "from": _masked_number(env["TWILIO_CALLER_ID"])}


@app.get("/calls")
async def list_calls(request: Request) -> list:
    _require_monitor_token(request)
    now = datetime.now(timezone.utc)
    return [
        {
            "call_sid": h.call_sid,
            "started_at": h.started_at.isoformat(),
            "duration_secs": round((now - h.started_at).total_seconds(), 1),
            "listeners": len(h.subscribers),
        }
        for h in ACTIVE_CALLS.values()
    ]


@app.get("/calls/{call_sid}/status")
async def call_status(call_sid: str, request: Request) -> dict:
    """Twilio-side call status (queued/ringing/in-progress/...); requires MONITOR_TOKEN."""
    _require_monitor_token(request)
    if call_sid.startswith("CAtest"):
        active = call_sid in ACTIVE_CALLS
        return {"call_sid": call_sid, "status": "in-progress" if active else "completed",
                "active_pipeline": active}
    client = TwilioClient(os.getenv("TWILIO_ACCOUNT_SID", ""), os.getenv("TWILIO_AUTH_TOKEN", ""))
    try:
        call = await asyncio.to_thread(lambda: client.calls(call_sid).fetch())
    except TwilioRestException as exc:
        raise HTTPException(status_code=404, detail=f"Twilio: {exc.msg}")
    return {"call_sid": call_sid, "status": call.status, "active_pipeline": call_sid in ACTIVE_CALLS}


@app.post("/calls/{call_sid}/hangup")
async def hangup_call(call_sid: str, request: Request) -> dict:
    _require_monitor_token(request)
    handle = ACTIVE_CALLS.get(call_sid)
    if call_sid.startswith("CAtest"):
        # Test calls have no real Twilio leg — just stop the pipeline.
        if handle is None:
            raise HTTPException(status_code=404, detail="no active call with that SID")
        if handle.task is not None:
            await handle.task.cancel()
        method = "task-cancel"
    else:
        # Ends the actual phone call; Twilio then closes the media stream and
        # the on_client_disconnected handler stops the pipeline normally, so
        # transcript/cost/Telegram flow is untouched. Works without a handle
        # too, so an outbound call can be canceled while it is still ringing.
        client = handle.twilio_client if handle else TwilioClient(
            os.getenv("TWILIO_ACCOUNT_SID", ""), os.getenv("TWILIO_AUTH_TOKEN", "")
        )
        try:
            await asyncio.to_thread(
                lambda: client.calls(call_sid).update(status="completed")
            )
        except TwilioRestException as exc:
            raise HTTPException(
                status_code=404 if handle is None else 502, detail=f"Twilio: {exc.msg}"
            )
        method = "twilio"
    logger.info(f"Remote hangup requested for {call_sid} (method={method})")
    return {"ok": True, "call_sid": call_sid, "method": method}


@app.websocket("/ws-monitor/{call_sid}")
async def websocket_monitor(websocket: WebSocket, call_sid: str) -> None:
    await websocket.accept()
    if not monitor_token_ok(websocket.query_params.get("token")):
        await websocket.close(code=4403, reason="invalid or missing monitor token")
        return
    handle = ACTIVE_CALLS.get(call_sid)
    if handle is None:
        await websocket.close(code=4404, reason="no active call with that SID")
        return
    queue = handle.subscribe()
    try:
        await websocket.send_json({
            "call_sid": handle.call_sid,
            "started_at": handle.started_at.isoformat(),
        })
        while True:
            msg = await queue.get()
            if msg is None:  # call ended
                await websocket.close(code=1000, reason="call ended")
                return
            if isinstance(msg, str):  # dialogue/tool event
                await websocket.send_text(msg)
            else:
                await websocket.send_bytes(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass  # listener went away — the call itself is unaffected
    finally:
        handle.unsubscribe(queue)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await run_bot(websocket)


@app.websocket("/ws-test")
async def websocket_test_endpoint(websocket: WebSocket) -> None:
    await run_bot(websocket, call_sid_override="CAtest000000000000000000000000000000")
