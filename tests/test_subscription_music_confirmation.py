"""Offline wire-to-handler music consent; metadata fixtures are not model proof."""

from __future__ import annotations
import json
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

import dj_reachy_remix.config as app_config
import dj_reachy_remix.openai_realtime as realtime
import dj_reachy_remix.subscription_voice as subscription
from dj_reachy_remix.runtime_settings import DEFAULT_SETTINGS
from dj_reachy_remix.tools.core_tools import ToolDependencies
from dj_reachy_remix.subscription_broker import BrokerService, CodexCredential


ORIGINAL = "Make a ten-second song about our coastal road trip, with a gentle jazz mood."
CANDIDATE = {"prompt": "Gentle jazz about our coastal road trip", "music_length_ms": 10000}


class WireHarness:
    """Keep the actual adapter, planner, domain and handler; fake terminal IO only."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Own settings and every provider/media boundary without opening a session."""
        self.app = realtime.OpenaiRealtimeHandler(
            ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        )
        self.connection = subscription.SubscriptionRealtimeAdapter().connect()
        self.app.connection = self.connection
        self.messages: list[dict[str, Any]] = []
        self.connection._channel = SimpleNamespace(
            readyState="open", send=lambda text: self.messages.append(json.loads(text))
        )
        self.connection.ready.set()
        self.settings = deepcopy(DEFAULT_SETTINGS)
        monkeypatch.setattr(app_config.runtime_settings_store, "read", lambda: deepcopy(self.settings))
        self.requests: list[dict[str, Any]] = []
        self.upstream: list[dict[str, Any]] = []
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.decision = "approve"
        self.revision: dict[str, Any] | None = None
        self.normal_name = "make_song_and_dance"
        self.normal_args = deepcopy(CANDIDATE)
        self.classifying = asyncio.Event()
        self.classifier_release: asyncio.Event | None = None
        self.classifier_releases: dict[str, asyncio.Event] = {}
        self.classifier_errors: set[str] = set()
        self.decision_text = "Fixture decision text."
        self.planning = asyncio.Event()
        self.planner_release: asyncio.Event | None = None
        self.generating = asyncio.Event()
        self.generation_release: asyncio.Event | None = None
        self.ignore_classifier_cancel = False
        self.fail_preview = False
        real_client, real_async = httpx.Client, httpx.AsyncClient
        service = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account"))

        def upstream(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.upstream.append(payload)
            confirmation = payload["tools"][0].get("tools", [{}])[0].get("name") == "music_confirmation_decision"
            output: list[dict[str, Any]]
            if confirmation:
                output = [
                    {
                        "type": "function_call",
                        "call_id": "fixture-metadata",
                        "namespace": "dj_reachy_remix",
                        "name": "music_confirmation_decision",
                        "arguments": json.dumps(
                            {"decision": self.decision, "arguments": self.revision, "text": self.decision_text}
                        ),
                    }
                ]
            elif any(item.get("type") == "function_call_output" for item in payload["input"]):
                output = [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Fixture app result."}],
                    }
                ]
            else:
                output = [
                    {
                        "type": "function_call",
                        "call_id": "fixture-candidate",
                        "namespace": "dj_reachy_remix",
                        "name": self.normal_name,
                        "arguments": json.dumps(self.normal_args),
                    }
                ]
            return httpx.Response(
                200,
                text="data: "
                + json.dumps({"type": "response.completed", "response": {"status": "completed", "output": output}})
                + "\n\n",
            )

        async def broker(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "https://broker.invalid/plan"
            body = json.loads(request.content)
            self.requests.append(body)
            if "pending_confirmation" in body:
                self.classifying.set()
                release = self.classifier_releases.get(body["transcript"], self.classifier_release)
                if release is not None:
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        if not self.ignore_classifier_cancel:
                            raise
                        await release.wait()
            else:
                self.planning.set()
                if self.planner_release is not None:
                    try:
                        await self.planner_release.wait()
                    except asyncio.CancelledError:
                        await self.planner_release.wait()
            if body.get("transcript") in self.classifier_errors:
                raise ConnectionError("Fixture classifier failure")
            return httpx.Response(200, json=service.plan(body))

        async def dispatch(name: str, arguments: str, deps: ToolDependencies) -> dict[str, Any]:
            self.executed.append((name, json.loads(arguments)))
            if name == "make_song_and_dance":
                self.generating.set()
                if self.generation_release is not None:
                    await self.generation_release.wait()
                # Never fabricate a successful paid generation in an offline test.
                return {"error": "FixtureTerminal", "message": "No provider job was submitted."}
            if name == "camera":
                return {"b64_im": "ZmFrZS1pbWFnZQ==", "mime_type": "image/jpeg"}
            return {"status": "fixture-control"}

        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(upstream), **kwargs)
        )
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: real_async(transport=httpx.MockTransport(broker), **kwargs)
        )
        monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
        monkeypatch.setenv(subscription.BROKER_URL_ENV, "https://broker.invalid/session")
        monkeypatch.setenv(subscription.BROKER_TOKEN_ENV, "fixture-scoped-token")

    async def event(self, value: dict[str, Any]) -> Any:
        """Consume actual queued wire data with adapter-assigned turn identity."""
        self.connection._enqueue(SimpleNamespace(**value))
        event = await asyncio.wait_for(self.connection.__anext__(), 1)
        await self.app._handle_event(event)
        return event

    async def fragment(self, text: str) -> Any:
        """Supply the first or a continuing actual user fragment."""
        return await self.event({"type": "input_transcript.added", "item": {"text": text}})

    async def final(self, text: str) -> Any:
        """Complete an actual user ASR turn."""
        return await self.event({"type": "turn.done", "turn": {"role": "user", "transcript": text}})

    async def user(self, text: str) -> Any:
        """Supply both wire forms of a complete user turn."""
        await self.fragment(text)
        return await self.final(text)

    async def delegate(self, identity: str = "real-live-id", text: str = ORIGINAL) -> Any:
        """Use genuine delegation shape and preserve provider identity."""
        return await self.event(
            {
                "type": "delegation.created",
                "item": {
                    "id": identity,
                    "type": "delegation",
                    "target": "client",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def drain(self) -> None:
        """Drain bounded owned work, including newly spawned reply tasks."""
        for _ in range(4):
            tasks = list(self.app._work_tasks | set(self.app._subscription_tasks.values()))
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks), 2)
            await asyncio.sleep(0)

    async def prepare(self) -> Any:
        """Make a candidate through actual initial delegation and normal SSE planning."""
        await self.user(ORIGINAL)
        await self.delegate()
        await self.drain()
        draft = self.app._pending_music.pending
        assert draft is not None
        return draft

    @property
    def songs(self) -> list[dict[str, Any]]:
        """Return only requests reaching the paid-tool terminal boundary."""
        return [args for name, args in self.executed if name == "make_song_and_dance"]


@pytest_asyncio.fixture
async def wire(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[WireHarness]:
    """Keep every test isolated and close all handler work on failure too."""
    harness = WireHarness(monkeypatch)
    try:
        yield harness
    finally:
        if harness.classifier_release is not None:
            harness.classifier_release.set()
        if harness.generation_release is not None:
            harness.generation_release.set()
        if harness.planner_release is not None:
            harness.planner_release.set()
        for release in harness.classifier_releases.values():
            release.set()
        await harness.app.shutdown()


@pytest.mark.asyncio
async def test_initial_delegation_offers_complete_frozen_preview_without_execution(wire: WireHarness) -> None:
    """A normal validated music call becomes an offered draft, never paid work."""
    await wire.user(ORIGINAL)
    await wire.delegate()
    await wire.drain()
    assert wire.songs == []
    draft = wire.app._pending_music.pending
    assert draft is not None
    assert draft.original_user_text == ORIGINAL
    assert draft.original_turn == 1
    assert draft.arguments["music_length_ms"] == 10000
    assert draft.arguments["model_id"] == "music_v2_5"
    assert len(wire.requests) == 1
    sent = json.dumps(wire.messages)
    assert draft.preview in sent.replace('\\"', '"')
    assert "real-live-id" in sent
    assert draft.id != "real-live-id"


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["Go ahead", "Sí, adelante con esa canción", "はい、その曲を作ってください"])
async def test_actual_fresh_reply_executes_frozen_args_once_without_another_live_event(
    wire: WireHarness, reply: str
) -> None:
    """Synthetic multilingual metadata tests wire causality, not model semantics."""
    draft = await wire.prepare()
    await wire.event(
        {"type": "turn.done", "turn": {"role": "assistant", "transcript": "Shall I generate this exact request?"}}
    )
    wire.messages.clear()
    await wire.user(reply)
    await wire.drain()
    assert wire.songs == [draft.arguments]
    assert wire.app._pending_music.pending is None
    assert len(wire.requests) == 2
    confirmation = wire.requests[-1]
    assert confirmation["transcript"] == reply
    assert confirmation["history"] == []
    assert confirmation["pending_confirmation"] == draft.pending_payload(["Shall I generate this exact request?"])
    assert wire.upstream[-1]["tools"][0]["tools"][0]["name"] == "music_confirmation_decision"
    assert "real-live-id" not in json.dumps(wire.messages)
    assert all("delegation" not in message.get("type", "") for message in wire.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "decision"), [("No, do not generate it", "decline"), ("Maybe later, how are you?", "none")]
)
async def test_refusal_and_ambiguity_never_execute_and_deliver_outer_text(
    wire: WireHarness, reply: str, decision: str
) -> None:
    """Use real domain outcomes, with RPC outer text explicitly lifted into apply."""
    draft = await wire.prepare()
    wire.decision = decision
    wire.messages.clear()
    await wire.user(reply)
    await wire.drain()
    assert wire.songs == []
    assert (wire.app._pending_music.pending is draft) == (decision == "none")
    assert "Fixture decision text." in json.dumps(wire.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["yes-but", "settings-drift"])
async def test_changes_repreview_then_require_another_fresh_actual_reply(wire: WireHarness, change: str) -> None:
    """Neither apparent consent with changes nor changed defaults authorize billing."""
    first = await wire.prepare()
    if change == "yes-but":
        wire.decision = "revise"
        wire.revision = {"prompt": "Faster jazz about our coastal road trip", "music_length_ms": 15000}
        reply = "Yes, but make it faster and fifteen seconds"
    else:
        wire.settings["music"]["positive_styles"] = ["swing"]
        reply = "Go ahead"
    await wire.user(reply)
    await wire.drain()
    second = wire.app._pending_music.pending
    assert second is not None and second.revision == first.revision + 1
    assert second.id == first.id
    assert wire.songs == []
    assert second.preview in json.dumps(wire.messages).replace('\\"', '"')
    wire.decision, wire.revision = "approve", None
    await wire.user("Proceed with that updated request")
    await wire.drain()
    assert wire.songs == [second.arguments]


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", ["early", "during-classifier", "after-result"])
async def test_live_confirmation_delegations_dedupe_against_actual_final(wire: WireHarness, ordering: str) -> None:
    """Model narration and duplicate IDs do not produce another assessment or song."""
    draft = await wire.prepare()
    wire.classifier_release = asyncio.Event()
    await wire.fragment("Go ahead")
    if ordering == "early":
        await wire.delegate("real-confirm-id", "The user confirmed the exact music request")
        await asyncio.sleep(0)
        assert len(wire.requests) == 1
    await wire.final("Go ahead")
    await asyncio.wait_for(wire.classifying.wait(), 1)
    if ordering == "during-classifier":
        await wire.delegate("real-confirm-id", "The user confirmed the exact music request")
    wire.classifier_release.set()
    await wire.drain()
    if ordering == "after-result":
        await wire.delegate("real-confirm-id", "The user confirmed the exact music request")
    await wire.delegate("real-confirm-id", "Duplicate same provider ID")
    await wire.delegate("real-second-confirm-id", "Reordered new provider ID for the same turn")
    await wire.drain()
    assert len(wire.requests) == 2
    assert wire.songs == [draft.arguments]
    assert wire.app._pending_music.pending is None
    assert "real-confirm-id" in json.dumps(wire.messages)


@pytest.mark.asyncio
async def test_delegation_before_original_final_waits_for_whole_actual_intent(wire: WireHarness) -> None:
    """A still-open original ASR turn must not lose trailing constraints or self-approve."""
    partial = "Make a song about our coastal road trip"
    event = await wire.fragment(partial)
    delegated = await wire.delegate(text="Make the requested song")
    assert event.turn_id == delegated.turn_id == 1
    await asyncio.sleep(0)
    assert wire.requests == []
    final = await wire.final(ORIGINAL)
    assert final.turn_id == delegated.turn_id
    await wire.drain()
    draft = wire.app._pending_music.pending
    assert draft is not None
    assert draft.original_user_text == ORIGINAL
    assert wire.requests[0]["transcript"] == ORIGINAL
    assert len(wire.requests) == 1
    assert wire.songs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "cancel", "close", "reconnect", "cancel-task"])
async def test_lifecycle_fences_late_classifier_results(wire: WireHarness, action: str) -> None:
    """A slow or cancellation-resistant classifier cannot revive cleared authority."""
    await wire.prepare()
    wire.classifier_release = asyncio.Event()
    wire.ignore_classifier_cancel = action in ("stop", "cancel", "cancel-task")
    await wire.user("Go ahead")
    await asyncio.wait_for(wire.classifying.wait(), 1)
    if action in ("stop", "cancel"):
        await wire.user(action)
        assert wire.app._pending_music.pending is None
    elif action == "close":
        await wire.connection.close()
    elif action == "reconnect":
        # Same handler receives a replacement connection; old tasks retain old identity.
        wire.app.connection = subscription.SubscriptionRealtimeAdapter().connect()
        wire.classifier_release.set()
        await wire.app._end_session_work()
    else:
        task = wire.app._reply_tasks[2]
        task.cancel()
    wire.classifier_release.set()
    await wire.drain()
    assert wire.songs == []
    assert wire.app._pending_music.pending is None


@pytest.mark.asyncio
async def test_pending_none_preserves_genuine_camera_but_blocks_duplicate_music(wire: WireHarness) -> None:
    """Only a genuine delegation may plan another action while a draft is pending."""
    draft = await wire.prepare()
    wire.decision = "none"
    wire.normal_name, wire.normal_args = "camera", {"question": "What can you see?"}
    await wire.user("What can you see?")
    await wire.delegate("camera-id", "Look at the camera")
    await wire.drain()
    assert [name for name, _ in wire.executed] == ["camera"]
    assert wire.app._pending_music.pending is draft
    assert "ZmFrZS1pbWFnZQ==" not in json.dumps(wire.messages)
    wire.normal_name, wire.normal_args = "make_song_and_dance", deepcopy(CANDIDATE)
    await wire.user("How is the weather?")
    await wire.delegate("duplicate-music-id", "Make the previously described song")
    await wire.drain()
    assert wire.songs == []
    assert wire.app._pending_music.pending is draft


@pytest.mark.asyncio
async def test_partial_new_speech_fences_older_approval_before_final_asr(wire: WireHarness) -> None:
    """The receiver advances the causal turn on the first new input fragment."""
    await wire.prepare()
    wire.classifier_release = asyncio.Event()
    await wire.user("Go ahead")
    await asyncio.wait_for(wire.classifying.wait(), 1)
    fragment = await wire.fragment("Actually, wait")
    assert fragment.turn_id == 3
    wire.classifier_release.set()
    await wire.drain()
    assert wire.songs == []


@pytest.mark.asyncio
async def test_generation_is_owned_without_blocking_receiver_or_busy_playback(wire: WireHarness) -> None:
    """Pending approval consumes before generation, with the existing song owner intact."""
    await wire.prepare()
    wire.generation_release = asyncio.Event()
    await wire.user("Go ahead")
    await asyncio.wait_for(wire.generating.wait(), 1)
    owner = wire.app._song_task
    assert owner is not None and wire.app._music_call_active
    await asyncio.wait_for(wire.user("Hello, are you still there?"), 0.2)
    assert len(wire.requests) == 2
    busy = await wire.app._execute("play_saved_song", {"id": "fixture-song"}, wire.connection, wire.app._work_epoch)
    assert busy["status"] == "busy"
    assert wire.app._song_task is owner
    wire.app.mute_output_audio("new-playback-owner")
    wire.generation_release.set()
    await wire.drain()
    assert wire.app._mute_output_audio
    assert wire.app._audio_mute_owner == "new-playback-owner"


@pytest.mark.asyncio
async def test_ordinary_chat_without_draft_does_not_trigger_planning(wire: WireHarness) -> None:
    """No transcript-per-turn normal planning is introduced by this repair."""
    await wire.user("Hello")
    await wire.user("Go ahead")
    await wire.drain()
    assert wire.requests == [] and wire.songs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("preview_state", ["failed", "not-ready", "speech-during-delivery"])
async def test_unoffered_preview_cannot_accept_early_or_later_approval(
    wire: WireHarness, monkeypatch: pytest.MonkeyPatch, preview_state: str
) -> None:
    """Only a successful ready-transport preview activates confirmation authority."""
    entered, release = asyncio.Event(), asyncio.Event()
    original_send = wire.connection.send_event

    async def terminal_send(message: dict[str, Any]) -> None:
        if "Music request (requires a fresh user reply)" in json.dumps(message):
            if preview_state == "failed":
                raise ConnectionError("Fixture delivery failure")
            if preview_state == "speech-during-delivery":
                entered.set()
                await release.wait()
        await original_send(message)

    monkeypatch.setattr(wire.connection, "send_event", terminal_send)
    if preview_state == "not-ready":
        wire.connection.ready.clear()
    await wire.user(ORIGINAL)
    await wire.delegate()
    if preview_state == "speech-during-delivery":
        await asyncio.wait_for(entered.wait(), 1)
        await wire.user("Go ahead")
        release.set()
    await wire.drain()
    assert wire.app._pending_music.pending is None
    await wire.user("Go ahead")
    await wire.drain()
    assert len(wire.requests) == 1
    assert wire.songs == []


@pytest.mark.asyncio
async def test_stale_approval_truthfully_reports_no_submission_without_newer_owner(wire: WireHarness) -> None:
    """Do not leave an earlier voice promise looking like submitted generation."""
    await wire.prepare()
    wire.classifier_release = asyncio.Event()
    await wire.user("Go ahead")
    await asyncio.wait_for(wire.classifying.wait(), 1)
    await wire.fragment("Actually, wait")
    wire.messages.clear()
    wire.classifier_release.set()
    await wire.drain()
    assert wire.songs == []
    assert "No music was submitted" in json.dumps(wire.messages)


@pytest.mark.asyncio
async def test_current_controls_keep_real_correlation_while_pending(wire: WireHarness) -> None:
    """Local controls keep their path and never become confirmation requests."""
    await wire.prepare()
    for text, name in [("pause music", "pause_music"), ("resume music", "resume_music"), ("stop music", "stop_music")]:
        await wire.user(text)
        await wire.delegate("real-" + name, text)
        await wire.drain()
    assert [name for name, _ in wire.executed] == ["pause_music", "resume_music", "stop_music"]
    assert len(wire.requests) == 1
    sent = [message for message in wire.messages if message.get("channel") == "speakable"]
    assert sent[-1]["delegation_item_id"] == "real-stop_music"


@pytest.mark.asyncio
async def test_api_key_function_call_path_does_not_use_subscription_drafts(
    wire: WireHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit API-key Realtime still uses its correlated tool contract."""
    monkeypatch.setattr(realtime.voice_config, "VOICE_BACKEND", "api_key")
    outputs: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> None:
        outputs.append(kwargs)

    wire.app.connection = SimpleNamespace(
        conversation=SimpleNamespace(item=SimpleNamespace(create=create)),
        response=SimpleNamespace(create=create),
        close=wire.connection.close,
    )
    await wire.app._handle_event(
        SimpleNamespace(
            type="response.function_call_arguments.done",
            name="make_song_and_dance",
            arguments=json.dumps(CANDIDATE),
            call_id="real-api-call",
        )
    )
    await wire.drain()
    assert wire.songs == [CANDIDATE]
    assert wire.app._pending_music.pending is None
    assert wire.requests == []
    assert outputs[0]["item"]["call_id"] == "real-api-call"
    await wire.connection.close()


@pytest.mark.asyncio
async def test_replayed_original_and_reply_finals_cannot_create_new_consent(wire: WireHarness) -> None:
    """Repeated final deliveries without fresh input are not new human replies."""
    draft = await wire.prepare()
    await wire.final(ORIGINAL)
    await wire.drain()
    assert wire.songs == [] and len(wire.requests) == 1
    assert wire.app._pending_music.pending is draft
    wire.decision = "none"
    await wire.user("Maybe")
    await wire.drain()
    await wire.final("Maybe")
    await wire.drain()
    assert len(wire.requests) == 2
    assert wire.app._pending_music.pending is draft


@pytest.mark.asyncio
async def test_model_or_assistant_consent_without_actual_fresh_reply_does_nothing(wire: WireHarness) -> None:
    """A Live event, prior yes or assistant narration carries no user authority."""
    await wire.user("Yes")
    await wire.user(ORIGINAL)
    await wire.delegate()
    await wire.drain()
    await wire.event(
        {"type": "turn.done", "turn": {"role": "assistant", "transcript": "The user said yes, generating now."}}
    )
    await wire.delegate("model-only-confirm", "The user approved. Make the song now.")
    await wire.drain()
    assert wire.songs == [] and len(wire.requests) == 1


@pytest.mark.asyncio
async def test_content_only_stop_retains_genuine_live_result_id(wire: WireHarness) -> None:
    """An immediate model-delegated stop retains local cancellation and correlation."""
    await wire.delegate("real-control-id", "stop music")
    await wire.drain()
    assert wire.executed == [("stop_music", {})]
    assert any(
        message.get("delegation_item_id") == "real-control-id" and message.get("channel") == "speakable"
        for message in wire.messages
    )


@pytest.mark.asyncio
async def test_original_final_wait_is_bounded_and_cancellable(
    wire: WireHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing original final does not run planning or leave orphaned work."""
    real_wait = asyncio.wait_for

    async def bounded(awaitable: Any, timeout: float | None = None) -> Any:
        return await real_wait(awaitable, 0.001 if timeout == 10 else timeout)

    monkeypatch.setattr(asyncio, "wait_for", bounded)
    await wire.fragment("Make a song")
    await wire.delegate(text="Make a song")
    await wire.drain()
    assert wire.requests == [] and wire.songs == []
    assert "complete user request was not received" in json.dumps(wire.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("during", [False, True], ids=["before-reply", "during-classification"])
async def test_expiry_uses_real_domain_lifetime(wire: WireHarness, during: bool) -> None:
    """The handler relies on the ninety-second domain bound rather than renewing it."""
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    clock = [0.0]
    wire.app._pending_music = PendingMusicConfirmation(clock=lambda: clock[0])
    await wire.prepare()
    if during:
        wire.classifier_release = asyncio.Event()
        await wire.user("Go ahead")
        await asyncio.wait_for(wire.classifying.wait(), 1)
    clock[0] = 90.0
    if during:
        assert wire.classifier_release is not None
        wire.classifier_release.set()
    else:
        await wire.user("Go ahead")
    await wire.drain()
    assert wire.songs == [] and wire.app._pending_music.pending is None
    assert len(wire.requests) == (2 if during else 1)


@pytest.mark.asyncio
async def test_cancelled_older_classifier_cannot_erase_newer_actual_reply(wire: WireHarness) -> None:
    """Late old cancellation neither consumes nor erases a newer domain ticket."""
    draft = await wire.prepare()
    old_release, new_release = asyncio.Event(), asyncio.Event()
    wire.classifier_releases = {"Proceed": old_release, "Please generate that exact draft": new_release}
    wire.ignore_classifier_cancel = True
    await wire.user("Proceed")
    await asyncio.wait_for(wire.classifying.wait(), 1)
    old = wire.app._reply_tasks[2]
    old.cancel()
    await asyncio.sleep(0)
    wire.classifying.clear()
    await wire.user("Please generate that exact draft")
    await asyncio.wait_for(wire.classifying.wait(), 1)
    old_release.set()
    await old
    assert wire.app._pending_music.pending is draft
    assert "No music was submitted" not in json.dumps(wire.messages)
    new_release.set()
    await wire.drain()
    assert wire.songs == [draft.arguments]


@pytest.mark.asyncio
async def test_old_preview_delivery_cannot_erase_newer_offered_request(
    wire: WireHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presentation ownership is separate from domain state and provider IDs."""
    entered, release = asyncio.Event(), asyncio.Event()
    original_send = wire.connection.send_event
    delayed = False

    async def terminal_send(message: dict[str, Any]) -> None:
        nonlocal delayed
        if not delayed and "Music request (requires a fresh user reply)" in json.dumps(message):
            delayed = True
            entered.set()
            await release.wait()
        await original_send(message)

    monkeypatch.setattr(wire.connection, "send_event", terminal_send)
    await wire.user(ORIGINAL)
    await wire.delegate()
    await asyncio.wait_for(entered.wait(), 1)
    await wire.user("Instead, make a fifteen-second lullaby about the moon")
    wire.normal_args = {"prompt": "Moon lullaby", "music_length_ms": 15000}
    await wire.delegate("real-new-request", "Prepare the new lullaby")
    await asyncio.wait_for(wire.app._subscription_tasks["real-new-request"], 1)
    new_draft = wire.app._pending_music.pending
    assert new_draft is not None and new_draft.arguments["prompt"] == "Moon lullaby"
    release.set()
    await wire.drain()
    assert wire.app._pending_music.pending is new_draft
    await wire.user("Proceed with the lullaby")
    await wire.drain()
    assert wire.songs == [new_draft.arguments]


@pytest.mark.asyncio
async def test_cancelled_initial_planner_cannot_offer_late_candidate(wire: WireHarness) -> None:
    """Cancellation-resistant terminal IO does not authorize a delayed preview."""
    wire.planner_release = asyncio.Event()
    await wire.user(ORIGINAL)
    await wire.delegate()
    await asyncio.wait_for(wire.planning.wait(), 1)
    task = wire.app._subscription_tasks["real-live-id"]
    task.cancel()
    await asyncio.sleep(0)
    wire.planner_release.set()
    await wire.drain()
    assert wire.app._pending_music.pending is None and wire.songs == []


@pytest.mark.asyncio
async def test_subscription_api_shaped_music_event_cannot_bypass_draft_gate(wire: WireHarness) -> None:
    """A consumer event in the wrong dialect is never a paid-tool authority."""
    await wire.prepare()
    await wire.event(
        {
            "type": "response.function_call_arguments.done",
            "name": "make_song_and_dance",
            "arguments": json.dumps(CANDIDATE),
            "call_id": "wrong-dialect-call",
        }
    )
    await asyncio.gather(*list(wire.app._tool_tasks.values()), return_exceptions=True)
    assert wire.songs == []


@pytest.mark.asyncio
async def test_end_session_clears_private_turn_and_preview_ownership(wire: WireHarness) -> None:
    """A replacement session cannot inherit old ASR, assistant or consent state."""
    await wire.prepare()
    await wire.event({"type": "turn.done", "turn": {"role": "assistant", "transcript": "Private old conversation"}})
    await wire.app._end_session_work()
    assert wire.app._latest_user_turn == 0
    assert wire.app._final_user_turns == {}
    assert wire.app._assistant_context == []
    assert wire.app._final_events == {}
    assert wire.app._preview_draft is None
    assert wire.app._pending_music.pending is None


@pytest.mark.asyncio
async def test_reordered_final_with_real_turn_id_is_not_a_new_reply(wire: WireHarness) -> None:
    """Provider turn identities survive reordered delivery without consent reuse."""
    original = {"type": "turn.done", "turn": {"id": "real-user-turn-one", "role": "user", "transcript": ORIGINAL}}
    await wire.fragment(ORIGINAL)
    await wire.event(original)
    await wire.delegate()
    await wire.drain()
    wire.decision = "none"
    await wire.fragment("Maybe")
    await wire.event(
        {"type": "turn.done", "turn": {"id": "real-user-turn-two", "role": "user", "transcript": "Maybe"}}
    )
    await wire.drain()
    wire.decision = "approve"
    await wire.event(original)
    await wire.drain()
    assert wire.songs == [] and len(wire.requests) == 2


@pytest.mark.asyncio
async def test_reordered_idless_final_delivery_cannot_reuse_original_speech(wire: WireHarness) -> None:
    """Whole-delivery replay detection is structural, never a lexical consent rule."""
    await wire.prepare()
    wire.decision = "none"
    await wire.user("Perhaps later")
    await wire.drain()
    wire.decision = "approve"
    await wire.final(ORIGINAL)
    await wire.drain()
    assert wire.songs == [] and len(wire.requests) == 2


@pytest.mark.asyncio
async def test_reordered_old_final_cannot_override_new_conflicting_fragment(wire: WireHarness) -> None:
    """A replayed old final must not turn newer actual refusal speech into consent."""
    await wire.prepare()
    wire.classifier_release = asyncio.Event()
    await wire.user("Proceed")
    await asyncio.wait_for(wire.classifying.wait(), 1)
    await wire.fragment("No, wait")
    await wire.final("Proceed")
    wire.classifier_release.set()
    await wire.drain()
    assert wire.songs == []
    assert len(wire.requests) == 2


@pytest.mark.asyncio
async def test_new_fragment_during_stop_barrier_fences_approved_dispatch(wire: WireHarness) -> None:
    """The final domain consume must happen after any awaited local admission barrier."""
    await wire.prepare()
    release = asyncio.Event()

    async def stopping() -> dict[str, Any]:
        await release.wait()
        return {"status": "fixture-stop-completed"}

    barrier = asyncio.create_task(stopping())
    wire.app._stop_barrier = barrier
    try:
        await wire.user("Go ahead")
        await asyncio.wait_for(wire.classifying.wait(), 1)
        await asyncio.sleep(0)
        await wire.fragment("Actually, wait")
        release.set()
        await wire.drain()
        assert wire.songs == []
    finally:
        release.set()
        await barrier


@pytest.mark.asyncio
async def test_old_classifier_failure_does_not_narrate_over_newer_work(wire: WireHarness) -> None:
    """A newer ticket owns both execution and user-facing generation status."""
    await wire.prepare()
    old_release = asyncio.Event()
    wire.classifier_releases = {"Proceed": old_release}
    wire.classifier_errors = {"Proceed"}
    await wire.user("Proceed")
    await asyncio.wait_for(wire.classifying.wait(), 1)
    wire.generation_release = asyncio.Event()
    await wire.user("Please generate the exact offered draft")
    await asyncio.wait_for(wire.generating.wait(), 1)
    wire.messages.clear()
    old_release.set()
    await wire.app._reply_tasks[2]
    assert wire.messages == []
    wire.generation_release.set()
    await wire.drain()


@pytest.mark.asyncio
async def test_result_delivery_failure_never_claims_a_submitted_job_did_not_run(
    wire: WireHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paid-tool terminal attempt and failed narration are not cancellation proof."""
    await wire.prepare()
    original_send = wire.connection.send_event

    async def terminal_send(message: dict[str, Any]) -> None:
        if "FixtureTerminal" in json.dumps(message):
            raise ConnectionError("Fixture result context failure")
        await original_send(message)

    monkeypatch.setattr(wire.connection, "send_event", terminal_send)
    wire.messages.clear()
    await wire.user("Go ahead")
    await wire.drain()
    assert len(wire.songs) == 1
    assert "submission status is unconfirmed" in json.dumps(wire.messages)
    assert "nothing was generated" not in json.dumps(wire.messages)


@pytest.mark.asyncio
async def test_bare_final_only_reply_uses_actual_asr_without_live_delegation(wire: WireHarness) -> None:
    """A complete real final is sufficient; another Live model event is not required."""
    draft = await wire.prepare()
    await wire.final("Go ahead")
    await wire.drain()
    assert wire.songs == [draft.arguments]


@pytest.mark.asyncio
async def test_empty_none_text_does_not_duplicate_ordinary_conversation(wire: WireHarness) -> None:
    """Metadata none without clarification leaves generic voice conversation alone."""
    await wire.prepare()
    wire.decision, wire.decision_text = "none", ""
    wire.messages.clear()
    await wire.user("How are you?")
    await wire.drain()
    assert wire.messages == [] and wire.songs == []
    assert len(wire.requests) == 2


@pytest.mark.asyncio
async def test_assistant_context_is_bounded_actual_data(wire: WireHarness) -> None:
    """Only recent genuine assistant finals accompany the frozen request."""
    await wire.prepare()
    wire.decision = "none"
    words = [str(index) + "é" * 1500 for index in range(6)]
    for text in words:
        await wire.event({"type": "turn.done", "turn": {"role": "assistant", "transcript": text}})
    await wire.user("Maybe")
    await wire.drain()
    context = wire.requests[-1]["pending_confirmation"]["assistant_context"]
    assert context == words[-2:]
    assert sum(len(item.encode("utf-8")) for item in context) <= 8192


@pytest.mark.asyncio
async def test_four_actual_reply_budget_is_not_reset_by_handler(wire: WireHarness) -> None:
    """The fifth actual reply cannot obtain another assessment or execute music."""
    await wire.prepare()
    wire.decision = "none"
    for index in range(4):
        await wire.user(f"Unsure, question {index}")
        await wire.drain()
    wire.decision = "approve"
    await wire.user("Go ahead")
    await wire.drain()
    assert len(wire.requests) == 5 and wire.songs == []
    assert wire.app._pending_music.pending is None


@pytest.mark.asyncio
async def test_genuine_saved_playback_remains_available_while_draft_pending(wire: WireHarness) -> None:
    """Saved playback is not new paid generation and retains the ordinary tool path."""
    draft = await wire.prepare()
    wire.decision = "none"
    wire.normal_name, wire.normal_args = "play_saved_song", {"filename": "fixture-track.mp3"}
    await wire.user("Play my saved track")
    await wire.delegate("real-saved-id", "Play fixture-track.mp3")
    await wire.drain()
    assert wire.executed == [("play_saved_song", {"filename": "fixture-track.mp3"})]
    assert wire.app._pending_music.pending is draft


@pytest.mark.asyncio
async def test_failed_revision_preview_never_reactivates_old_or_new_request(
    wire: WireHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correction whose preview failed cannot be approved by the following reply."""
    await wire.prepare()
    wire.decision = "revise"
    wire.revision = {"prompt": "Quieter road trip", "music_length_ms": 10000}
    original_send = wire.connection.send_event

    async def terminal_send(message: dict[str, Any]) -> None:
        if "Music request (requires a fresh user reply)" in json.dumps(message):
            raise ConnectionError("Fixture revised-preview failure")
        await original_send(message)

    monkeypatch.setattr(wire.connection, "send_event", terminal_send)
    await wire.user("Yes, but quieter")
    await wire.drain()
    assert wire.app._pending_music.pending is None
    wire.decision, wire.revision = "approve", None
    await wire.user("Go ahead")
    await wire.drain()
    assert wire.songs == [] and len(wire.requests) == 2


@pytest.mark.asyncio
async def test_actual_supervised_reconnect_uses_fake_sdk_and_forgets_pending_authority(
    wire: WireHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real supervisor/restart/receiver lifecycle without SDP or devices."""
    second = subscription.SubscriptionRealtimeAdapter().connect()
    second._channel = wire.connection._channel
    second.ready.set()
    connections = iter([wire.connection, second])

    class FakeSession:
        """Replace only SDK setup; retain the production subscription iterator."""

        def __init__(self, connection: subscription.SubscriptionRealtimeConnection) -> None:
            self.connection = connection

        async def __aenter__(self) -> subscription.SubscriptionRealtimeConnection:
            return self.connection

        async def __aexit__(self, *args: Any) -> None:
            await self.connection.close()

    client = SimpleNamespace(realtime=SimpleNamespace(connect=lambda **kwargs: FakeSession(next(connections))))
    monkeypatch.setattr(wire.app, "_create_client", lambda key: setattr(wire.app, "client", client))
    running = asyncio.create_task(wire.app.start_up())

    async def wait_until(predicate: Any) -> None:
        async def wait() -> None:
            while not predicate():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), 1)

    def enqueue(value: dict[str, Any]) -> None:
        wire.connection._enqueue(SimpleNamespace(**value))

    try:
        await asyncio.wait_for(wire.app._connected_event.wait(), 1)
        enqueue({"type": "input_transcript.added", "item": {"text": ORIGINAL}})
        enqueue({"type": "turn.done", "turn": {"role": "user", "transcript": ORIGINAL}})
        enqueue(
            {
                "type": "delegation.created",
                "item": {
                    "id": "real-supervised-id",
                    "type": "delegation",
                    "target": "client",
                    "content": [{"type": "input_text", "text": ORIGINAL}],
                },
            }
        )
        await wait_until(lambda: wire.app._pending_music.pending is not None)
        wire.classifier_release = asyncio.Event()
        enqueue({"type": "input_transcript.added", "item": {"text": "Go ahead"}})
        enqueue({"type": "turn.done", "turn": {"role": "user", "transcript": "Go ahead"}})
        await asyncio.wait_for(wire.classifying.wait(), 1)
        await asyncio.wait_for(wire.app._restart_session(), 1)
        assert wire.connection._closed and wire.app.connection is second
        assert wire.app._pending_music.pending is None and wire.app._latest_user_turn == 0
        wire.classifier_release.set()
        second._enqueue(SimpleNamespace(type="turn.done", turn={"role": "user", "transcript": "Go ahead"}))
        await wait_until(lambda: wire.app._latest_user_turn == 1)
        assert wire.songs == [] and len(wire.requests) == 2
    finally:
        wire.classifier_release.set() if wire.classifier_release is not None else None
        await wire.app.shutdown()
        await asyncio.gather(running, return_exceptions=True)
        await second.close()


@pytest.mark.asyncio
async def test_approved_draft_does_not_steal_an_existing_playback_owner(wire: WireHarness) -> None:
    """A consumed approval cannot replace or unmute independently owned playback."""
    await wire.prepare()
    release = asyncio.Event()
    owner = asyncio.create_task(release.wait())
    wire.app._song_task = owner
    wire.app._music_call_active = True
    wire.app.mute_output_audio("saved-playback-owner")
    try:
        await wire.user("Go ahead")
        await wire.drain()
        assert wire.songs == [] and wire.app._song_task is owner
        assert wire.app._music_call_active and wire.app._mute_output_audio
        assert wire.app._audio_mute_owner == "saved-playback-owner"
        assert wire.app._pending_music.pending is None
        assert "busy" in json.dumps(wire.messages)
    finally:
        release.set()
        await owner
        wire.app._song_task = None
        wire.app._music_call_active = False


@pytest.mark.asyncio
async def test_matching_new_actual_fragments_allow_same_words_after_repreview(wire: WireHarness) -> None:
    """Structural replay fencing does not substitute a consent vocabulary."""
    await wire.prepare()
    wire.settings["music"]["positive_styles"] = ["swing"]
    await wire.user("Go ahead")
    await wire.drain()
    revised = wire.app._pending_music.pending
    assert revised is not None and wire.songs == []
    await wire.user("Go ahead")
    await wire.drain()
    assert wire.songs == [revised.arguments]


@pytest.mark.asyncio
async def test_repeated_idless_stop_never_loses_the_safety_control(wire: WireHarness) -> None:
    """Ambiguous repeated delivery may stop again, but must never leave work running."""
    await wire.final("stop music")
    await wire.drain()
    await wire.final("stop music")
    await wire.drain()
    assert [name for name, _ in wire.executed if name == "stop_music"] == ["stop_music", "stop_music"]
    assert wire.requests == []
    assert wire.songs == []


@pytest.mark.asyncio
async def test_current_owner_works_without_python311_task_cancelling(wire: WireHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the advertised Python 3.10 task surface instead of raising during work."""
    legacy_owner = SimpleNamespace(cancelled=lambda: False)
    current = False
    with monkeypatch.context() as patcher:
        patcher.setattr(asyncio, "current_task", lambda: legacy_owner)
        try:
            current = wire.app._current(wire.connection, wire.app._work_epoch)
        except AttributeError:
            pass
    assert current, "An active legacy task must not require Python 3.11's cancelling() method"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
async def test_confirmation_planner_calls_are_counted_without_payloads(wire: WireHarness, tmp_path: Any, outcome: str) -> None:
    """Do not hide the extra subscription round or retain private request data."""
    wire.app.diagnostics.home = tmp_path
    wire.app.diagnostics.start()
    wire.normal_args["prompt"] = "PRIVATE_DRAFT_PAYLOAD"
    await wire.prepare()
    if outcome == "failed":
        wire.classifier_errors.add("Go ahead")
    if outcome == "cancelled":
        wire.classifier_release = asyncio.Event()
    await wire.user("Go ahead")
    if outcome == "cancelled":
        await asyncio.wait_for(wire.classifying.wait(), 1)
        task = wire.app._reply_tasks[wire.app._latest_user_turn]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    else:
        await wire.drain()
    counts = wire.app.diagnostics.counts
    assert counts["planner.started"] == 2
    assert counts["planner.completed"] == (2 if outcome == "completed" else 1)
    if outcome != "completed":
        assert counts["planner." + outcome] == 1
    wire.app.diagnostics.close()
    assert "PRIVATE_DRAFT_PAYLOAD" not in (tmp_path / "logs" / "audio-trace.jsonl").read_text()
