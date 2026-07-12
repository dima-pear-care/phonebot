"""Bot pipeline assembly and the per-call runner.

`run_bot()` is the orchestrator: it handshakes the Twilio stream, builds the
services / context / pipeline stages via the `build_*` helpers, runs the pipeline,
then persists the transcript + cost sidecar and fires a Telegram notification.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

import aiohttp
from fastapi import WebSocket
from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from twilio.rest import Client as TwilioClient

from phonebot.config import CONFIG, TRANSCRIPT_LOG
from phonebot.costs import CostTracker, el_subscription_snapshot
from phonebot.dtmf import build_dtmf_tool, make_press_handler
from phonebot.greeting import (
    GreetingAudioInjector,
    GreetingGate,
    prerender_greeting,
)
from phonebot.monitor import (
    CHANNEL_BOT,
    CHANNEL_CALLER,
    AudioTap,
    MonitoredDialogue,
    register_call,
    unregister_call,
)
from phonebot.outbound import apply_call_overrides, pop_call_overrides
from phonebot.processors import (
    AudioPadder,
    BotTracker,
    ClientTracker,
    LLMInputTimer,
    LLMOutputTimer,
    TranscriptLogger,
    UsageCollector,
)
from phonebot.prompts import build_system_prompt
from phonebot.transcript import save_transcript, send_telegram
from phonebot.tts import SanitizingElevenLabsHttpTTSService, needs_dash_sanitizing, sanitize_dashes


async def _await_twilio_start(websocket: WebSocket, call_sid_override: str | None):
    """Read Twilio WebSocket events until 'start'; returns (stream_sid, call_sid)."""
    while True:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        event = data.get("event", "")
        if event == "connected":
            logger.info("Twilio WebSocket protocol: connected")
            continue
        if event == "start":
            stream_sid = data.get("streamSid")
            call_sid = call_sid_override or data["start"].get("callSid")
            logger.info(f"Stream started | stream_sid={stream_sid} call_sid={call_sid}")
            return stream_sid, call_sid


def build_turn_detection(cfg: dict):
    """Pick the end-of-turn / interruption strategy for this call's cfg snapshot.

    Returns (vad_analyzer, turn_analyzer) for FastAPIWebsocketParams. Selected per
    call via cfg["turn_mode"] (default "smart"), which /outbound-call can override:

    - "smart" — Silero VAD as a fast pre-filter (short 0.2s silence window) plus
      smart-turn-v3 for the actual semantic end-of-turn decision, with a 2.0s
      ceiling when the model keeps saying "incomplete". Tolerates natural pauses.
    - "vad" — no semantic model; end-of-turn is a fixed silence gap (0.8s). Snappier
      and cheaper, but cuts the caller off during mid-sentence pauses.

    Barge-in (allow_interruptions) is on in both modes; this only changes how the
    end of the caller's turn is detected.
    """
    mode = (cfg.get("turn_mode") or "smart").strip().lower()
    if mode == "vad":
        return SileroVADAnalyzer(params=VADParams(stop_secs=0.8)), None
    return (
        SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=2.0)),
    )


def build_services(cfg: dict, websocket: WebSocket, stream_sid: str, call_sid: str | None):
    """Construct transport, STT, TTS, LLM and the TTS HTTP session for this call."""
    vad_analyzer, turn_analyzer = build_turn_detection(cfg)
    logger.info(f"Turn detection mode: {(cfg.get('turn_mode') or 'smart').lower()}")
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=vad_analyzer,
            turn_analyzer=turn_analyzer,
            serializer=TwilioFrameSerializer(
                stream_sid,
                call_sid=call_sid,
                account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            ),
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        settings=DeepgramSTTService.Settings(
            model=cfg["stt_model"],
            language=cfg["stt_language"],
            endpointing=200,
        ),
    )

    # HTTP streaming TTS (one request per utterance) instead of the
    # multi-stream WebSocket: flash/turbo stochastically produce repeats,
    # junk words and long silences on the WebSocket path, but are clean
    # over HTTP (see guides/tts_artifact_debugging.md).
    tts_session = aiohttp.ClientSession()
    tts = SanitizingElevenLabsHttpTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        aiohttp_session=tts_session,
        sample_rate=8000,
        settings=ElevenLabsHttpTTSService.Settings(
            voice=cfg["tts_voice_id"],
            model=cfg["tts_model"],
        ),
    )

    if cfg["llm_provider"] == "fireworks":
        llm = OpenAILLMService(
            api_key=os.getenv("FIREWORKS_API_KEY", ""),
            base_url="https://api.fireworks.ai/inference/v1",
            model=cfg["llm_model"],
        )
    else:
        llm = GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY", ""),
            settings=GroqLLMService.Settings(model=cfg["llm_model"]),
        )

    return transport, stt, tts, tts_session, llm


def build_context(cfg: dict):
    """Build the LLM context (system prompt + tools) and its aggregator pair."""
    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(cfg)}],
        tools=ToolsSchema(standard_tools=[build_dtmf_tool()]),
    )
    return context, LLMContextAggregatorPair(context)


async def _warmup_llm(llm, cfg: dict, tracker: CostTracker) -> None:
    """Open the LLM connection early so the first real turn isn't cold."""
    try:
        resp = await llm._client.chat.completions.create(
            model=cfg["llm_model"],
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            stream=False,
        )
        if resp.usage:
            tracker.llm_calls += 1
            tracker.llm_prompt_tokens += resp.usage.prompt_tokens or 0
            tracker.llm_completion_tokens += resp.usage.completion_tokens or 0
        logger.info(f"LLM connection warmed up ({cfg['llm_provider']})")
    except Exception as exc:
        logger.warning(f"LLM warmup failed: {exc}")


def _start_greeting_prerender(cfg: dict, tracker: CostTracker) -> asyncio.Task | None:
    """Kick off greeting TTS rendering in the background; None if no greeting set."""
    if not cfg["greeting"]:
        return None
    greeting_tts_text = (
        sanitize_dashes(cfg["greeting"])
        if needs_dash_sanitizing(cfg["tts_model"])
        else cfg["greeting"]
    )
    return asyncio.create_task(
        prerender_greeting(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id=cfg["tts_voice_id"],
            model=cfg["tts_model"],
            text=greeting_tts_text,
            tracker=tracker,
        )
    )


def build_stages(cfg, transport, stt, llm, tts, context, context_aggregator,
                 tracker, dialogue, turn_timing, render_task, monitor_handle,
                 greeting_injector) -> list:
    """Assemble the ordered list of pipeline processors."""
    transcript_logger = TranscriptLogger() if TRANSCRIPT_LOG else None

    stages: list = [transport.input(), AudioTap(monitor_handle, CHANNEL_CALLER), stt]
    if transcript_logger:
        stages.append(transcript_logger)
    stages.append(ClientTracker(dialogue))
    if render_task:
        stages.append(GreetingGate(context, cfg["greeting"], render_task, dialogue))
    stages += [
        context_aggregator.user(),
        LLMInputTimer(turn_timing),
        llm,
        LLMOutputTimer(turn_timing),
        BotTracker(dialogue, turn_timing),
        tts,
        UsageCollector(tracker),
        context_aggregator.assistant(),
        AudioPadder(),
        greeting_injector,
        AudioTap(monitor_handle, CHANNEL_BOT),
        transport.output(),
    ]
    return stages


async def run_bot(websocket: WebSocket, call_sid_override: str | None = None) -> None:
    await websocket.accept()

    stream_sid, call_sid = await _await_twilio_start(websocket, call_sid_override)
    if not stream_sid:
        logger.error("Never received Twilio 'start' event — closing WebSocket")
        await websocket.close()
        return

    started_at = datetime.now(timezone.utc).isoformat()

    twilio_client = TwilioClient(
        os.getenv("TWILIO_ACCOUNT_SID", ""),
        os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    monitor_handle = register_call(call_sid or stream_sid, twilio_client)

    cfg = dict(CONFIG)  # snapshot here so recording flag is consistent with other settings

    # Outbound calls placed via /outbound-call may carry per-contact overrides
    # (doctor name, patient context); they customize this snapshot only.
    overrides = pop_call_overrides(call_sid)
    if overrides:
        apply_call_overrides(cfg, overrides)

    tracker = CostTracker(
        llm_provider=cfg["llm_provider"],
        llm_model=cfg["llm_model"],
        stt_model=cfg["stt_model"],
        tts_model=cfg["tts_model"],
    )
    asyncio.create_task(el_subscription_snapshot(tracker, "start"))

    if cfg["recording"] and call_sid and not call_sid.startswith("CAtest"):
        try:
            twilio_client.calls(call_sid).recordings.create(recording_channels="dual")
            logger.info(f"Dual-channel recording started for {call_sid}")
        except Exception as exc:
            logger.warning(f"Failed to start recording: {exc}")

    transport, stt, tts, tts_session, llm = build_services(cfg, websocket, stream_sid, call_sid)

    asyncio.create_task(_warmup_llm(llm, cfg, tracker))
    render_task = _start_greeting_prerender(cfg, tracker)
    greeting_injector = GreetingAudioInjector()

    # Per-call tools — handlers are bound to this call's twilio client + dialogue.
    dtmf_mode = os.getenv("DTMF_MODE", "mock").lower()
    dialogue: list[dict] = MonitoredDialogue(monitor_handle)
    llm.register_function("press_dtmf", make_press_handler(twilio_client, call_sid, dialogue, dtmf_mode))

    context, context_aggregator = build_context(cfg)

    turn_timing: dict = {}
    stages = build_stages(cfg, transport, stt, llm, tts, context, context_aggregator,
                          tracker, dialogue, turn_timing, render_task, monitor_handle,
                          greeting_injector)

    task = PipelineTask(
        Pipeline(stages),
        params=PipelineParams(allow_interruptions=True, enable_metrics=True, enable_usage_metrics=True),
    )
    monitor_handle.task = task

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket) -> None:
        logger.info("Client disconnected — stopping pipeline")
        await task.cancel()

    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        unregister_call(call_sid or stream_sid)
        await tts_session.close()

    try:
        ended_at = datetime.now(timezone.utc).isoformat()
        tracker.duration_secs = (
            datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)
        ).total_seconds()
        await el_subscription_snapshot(tracker, "end")
        text, txt_path = save_transcript(started_at, ended_at, dialogue, tracker)
        asyncio.create_task(send_telegram(text))
    except Exception as exc:
        logger.warning(f"Failed to save/send transcript: {exc}")
