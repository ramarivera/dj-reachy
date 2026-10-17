"""Canonical Live delegations exercise the real adapter-to-handler scheduling path."""

from __future__ import annotations
import json
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock
from collections.abc import Mapping, AsyncIterator

import pytest
import pytest_asyncio

import dj_reachy_remix.config as app_config
from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler
from dj_reachy_remix.runtime_settings import DEFAULT_SETTINGS
from dj_reachy_remix.tools.core_tools import ToolDependencies
from dj_reachy_remix.subscription_voice import SubscriptionRealtimeAdapter, SubscriptionRealtimeConnection


MODEL_REQUEST_PREFIX = "Live model delegation request (not a verbatim user transcript):\n"


class DelegationHarness:
    """Inject only the planner; retain event decoding, scheduling and context encoding."""

    def __init__(self) -> None:
        """Construct without opening transport, credentials or hardware."""
        self.app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
        self.connection: SubscriptionRealtimeConnection = SubscriptionRealtimeAdapter().connect()
        self.app.connection = self.connection
        self.requests: list[str] = []
        self.messages: list[str] = []
        self.connection._channel = SimpleNamespace(readyState="open", send=self.messages.append)
        self.connection.ready.set()

    async def plan(self, request: str, history: list[dict[str, object]]) -> dict[str, object]:
        """Record the actual planner input and finish in one bounded round."""
        self.requests.append(request)
        return {"calls": [], "output": [], "text": "Availability checked."}

    async def event(self, value: Mapping[str, object]) -> SimpleNamespace:
        """Feed a decoded wire event through both production consumers."""
        self.connection._enqueue(SimpleNamespace(**value))
        event = await self.connection.__anext__()
        assert isinstance(event, SimpleNamespace)
        await self.app._handle_event(event)
        return event

    async def drain(self) -> None:
        """Wait only for app-owned delegated jobs, not a network receiver."""
        await asyncio.gather(*list(self.app._work_tasks | set(self.app._subscription_tasks.values())))

    async def final(self, text: str) -> SimpleNamespace:
        """Complete the actual ASR turn before waiting on its delegated planner."""
        return await self.event({"type": "turn.done", "turn": {"role": "user", "transcript": text}})


@pytest_asyncio.fixture
async def live(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[DelegationHarness]:
    """Own every task and isolate the only external planner boundary."""
    harness = DelegationHarness()
    monkeypatch.setattr(harness.connection, "plan", harness.plan)
    settings = deepcopy(DEFAULT_SETTINGS)
    settings["music"] = {
        "model_id": "music_v2_5",
        "duration_ms": 30000,
        "output_format": "mp3_48000_192",
        "positive_styles": [],
        "negative_styles": [],
        "lyrics": "",
        "language": "auto",
        "force_instrumental": False,
        "finetune_id": None,
    }
    monkeypatch.setattr(app_config.runtime_settings_store, "read", lambda: deepcopy(settings))
    try:
        yield harness
    finally:
        await harness.app.shutdown()


@pytest.mark.asyncio
async def test_official_content_without_transcript_reaches_planner_once(live: DelegationHarness) -> None:
    """Replay official Codex aff92080 realtime_conversation.rs:4233-4236 exactly."""
    await live.event({"type": "session.started", "session": {"id": "summary_session"}})
    wire = {
        "type": "delegation.created",
        "item": {
            "id": "summary_delegation",
            "type": "delegation",
            "target": "client",
            "content": [{"type": "input_text", "text": "check availability"}],
        },
    }
    event = await live.event(wire)
    await live.drain()
    assert live.requests == [MODEL_REQUEST_PREFIX + "check availability"]
    assert event.transcript_context == ""  # Model-authored content is never user speech.
    messages = [json.loads(message) for message in live.messages]
    assert messages[-1] == {
        "type": "delegation.context.append",
        "channel": "speakable",
        "delegation_item_id": "summary_delegation",
        "content": [{"type": "input_text", "text": "Availability checked."}],
    }
    await live.event(wire)
    await live.drain()
    assert len(live.requests) == 1


def request_event(identity: str, text: str | None = None) -> dict[str, object]:
    """Build canonical content or a legacy transcript-backed delegation notice."""
    item: dict[str, object] = {"id": identity, "type": "delegation", "target": "client"}
    if text is not None:
        item["content"] = [{"type": "input_text", "text": text}]
    return {"type": "delegation.created", "item": item}


@pytest.mark.asyncio
@pytest.mark.parametrize("notice", ["invalid-target", "duplicate-id"])
async def test_rejected_notice_does_not_consume_fresh_user_transcript(live: DelegationHarness, notice: str) -> None:
    """Validation and deduplication must precede consuming real user speech."""
    if notice == "duplicate-id":
        await live.event(request_event("old", "check availability"))
        await live.drain()
    await live.event({"type": "input_transcript.added", "item": {"text": "sing a salsa"}})
    if notice == "invalid-target":
        await live.event(
            {"type": "delegation.created", "item": {"id": "bad", "type": "delegation", "target": "server"}}
        )
    else:
        await live.event(request_event("old", "check availability"))
    event = await live.event(request_event("fresh"))
    await live.final("sing a salsa")
    await live.drain()
    assert event.transcript_context == "sing a salsa"
    assert live.requests[-1] == "sing a salsa"


@pytest.mark.asyncio
async def test_content_only_controls_have_distinct_correlations(
    live: DelegationHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical controls execute locally once per ID, without a planner round or turn0 collision."""
    import dj_reachy_remix.openai_realtime as realtime

    commands: list[str] = []

    async def dispatch(name: str, arguments: str, deps: ToolDependencies) -> dict[str, str]:
        commands.append(name)
        return {"status": "completed"}

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    for identity, text in [
        ("pause-1", "pause music"),
        ("resume", "resume music"),
        ("stop", "stop music"),
        ("pause-2", "pause music"),
    ]:
        event = request_event(identity, text)
        await live.event(event)
        await live.drain()
        await live.event(event)
        await live.drain()
    assert commands == ["pause_music", "resume_music", "stop_music", "pause_music"]
    assert live.requests == []
    messages = [json.loads(message) for message in live.messages]
    assert [message["delegation_item_id"] for message in messages if message["channel"] == "speakable"] == [
        "pause-1",
        "resume",
        "stop",
        "pause-2",
    ]


@pytest.mark.asyncio
async def test_queued_delegation_cannot_run_on_replacement_connection(
    live: DelegationHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind ownership when scheduling, not after the task first gets CPU time."""
    replacement = SubscriptionRealtimeAdapter().connect()
    monkeypatch.setattr(replacement, "plan", live.plan)
    try:
        await live.event(request_event("obsolete", "check availability"))
        live.app.connection = replacement
        await live.drain()
        assert live.requests == []
        assert not replacement._pending
    finally:
        await replacement.close()
        live.app.connection = live.connection


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"type": "message"},
        {"target": "server"},
        {"id": ""},
        {"id": " "},
        {"id": 7},
        {"id": "x" * 201},
        {"content": "stop music"},
        {"content": [{"type": "output_text", "text": "stop music"}]},
        {"content": [{"type": "input_text", "text": 7}]},
        {"content": [{"type": "input_text", "text": {"opaque": "stop music"}, "extra": True}]},
        {"content": [{"type": "input_text", "text": "x" * 6001}]},
        {"content": [{"type": "input_text", "text": "🌶" * 3001}]},
        {"content": [{"type": "input_text", "text": " " * 6001}]},
        {"content": [{"type": "input_text", "text": "\ud800"}]},
        {"content": [{"type": "input_text", "text": "x"}] * 65},
    ],
)
async def test_invalid_delegation_does_not_authorize_or_consume_user_words(
    live: DelegationHarness, fields: dict[str, object]
) -> None:
    """Reject malformed/oversized model content as a whole, including transcript-backed events."""
    await live.event({"type": "input_transcript.added", "item": {"text": "check my saved songs"}})
    item: dict[str, object] = {"id": "invalid", "type": "delegation", "target": "client"}
    item.update(fields)
    await live.event({"type": "delegation.created", "item": item})
    await live.drain()
    assert live.requests == []
    event = await live.event(request_event("valid"))
    await live.final("check my saved songs")
    await live.drain()
    assert event.transcript_context == "check my saved songs"
    assert live.requests == ["check my saved songs"]


@pytest.mark.asyncio
async def test_empty_notice_can_be_populated_and_distinct_requests_do_not_share_turn0(live: DelegationHarness) -> None:
    """No-op notices reserve neither IDs nor user turns; populated IDs execute only once."""
    await live.event(request_event("first"))
    await live.drain()
    assert not live.app._seen_delegations
    assert not live.connection._seen_delegations
    assert live.connection._consumed_turn == -1
    await live.event(request_event("first", "check availability"))
    await live.event(request_event("first", "check availability"))  # Also dedupe in-flight jobs.
    await live.drain()
    await live.event(request_event("second", "list saved songs"))
    await live.drain()
    assert live.requests == [MODEL_REQUEST_PREFIX + "check availability", MODEL_REQUEST_PREFIX + "list saved songs"]


@pytest.mark.asyncio
async def test_user_words_take_precedence_but_consumed_words_are_never_replayed(live: DelegationHarness) -> None:
    """Keep actual input separate from delegation paraphrases and assistant output."""
    await live.event({"type": "input_transcript.added", "item": {"text": "sing a salsa"}})
    event = await live.event(request_event("song", "model paraphrase"))
    await asyncio.sleep(0)
    assert live.requests == []
    await live.event({"type": "turn.done", "turn": {"role": "user", "transcript": "sing a salsa"}})
    await live.drain()
    assert event.transcript_context == "sing a salsa"
    assert live.requests == ["sing a salsa"]
    await live.event({"type": "output_transcript.added", "item": {"text": "stop music"}})
    await live.event({"type": "turn.done", "turn": {"role": "assistant", "transcript": "stop music"}})
    await live.event(request_event("no-new-request"))
    await live.drain()
    assert live.requests == ["sing a salsa"]
    event = await live.event(request_event("new-request", "list saved songs"))
    await live.drain()
    assert event.transcript_context == ""
    assert live.requests == ["sing a salsa", MODEL_REQUEST_PREFIX + "list saved songs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_at", ["planner", "song"])
async def test_content_stop_preempts_work_and_rejects_cancellation_resistant_results(
    live: DelegationHarness, monkeypatch: pytest.MonkeyPatch, blocked_at: str
) -> None:
    """Local stop dispatches before blocked work is released, with no stale result or song restart."""
    import dj_reachy_remix.openai_realtime as realtime

    entered, released, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancelled: list[bool] = []
    commands: list[str] = []
    executed: list[dict[str, object]] = []
    approvals: list[tuple[str, dict[str, object]]] = []

    async def block() -> None:
        entered.set()
        try:
            await released.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            await released.wait()

    async def plan(request: str, history: list[dict[str, object]]) -> dict[str, object]:
        live.requests.append(request)
        if blocked_at == "planner":
            await block()
        return {
            "calls": [{"name": "make_song_and_dance", "call_id": "song-call", "arguments": {"prompt": "salsa"}}],
            "output": [],
        }

    async def approve(text: str, pending: dict[str, object]) -> dict[str, object]:
        approvals.append((text, pending))
        return {"confirmation": {"decision": "approve", "arguments": None}, "text": "Fixture approval."}

    async def dispatch(name: str, arguments: str, deps: ToolDependencies) -> dict[str, str]:
        commands.append(name)
        if name == "make_song_and_dance":
            executed.append(json.loads(arguments))
            await block()
            return {"error": "FixtureTerminal", "message": "No provider job was submitted."}
        elif name == "stop_music":
            stopped.set()
        return {"status": "completed"}

    monkeypatch.setattr(live.connection, "plan", plan)
    monkeypatch.setattr(live.connection, "plan_confirmation", approve)
    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    draft = None
    try:
        await live.final("sing a salsa")
        await live.event(request_event("song", "model paraphrase, not actual consent"))
        if blocked_at == "song":
            await live.drain()
            draft = live.app._pending_music.pending
            assert draft is not None
            assert draft.original_user_text == "sing a salsa"
            assert draft.arguments["prompt"] == "salsa"
            assert draft.arguments["music_length_ms"] == 30000
            assert commands == []
            assert draft.preview in json.dumps([json.loads(message) for message in live.messages]).replace('\\"', '"')
            await live.final("Generate that exact preview now")
            approval_event = await live.event(request_event("approval", "The user approved the offered preview"))
            assert approval_event.transcript_context == "Generate that exact preview now"
        await asyncio.wait_for(entered.wait(), 1)
        assert live.requests == ["sing a salsa"]
        if blocked_at == "song":
            assert draft is not None
            assert executed == [draft.arguments]
            assert approvals == [("Generate that exact preview now", draft.pending_payload([]))]
        # The initial offered preview is valid; only post-stop stale output is forbidden.
        stop_message_start = len(live.messages)
        await live.event(request_event("stop", "stop music"))
        await asyncio.wait_for(stopped.wait(), 1)
        assert not released.is_set()
        released.set()
        await live.drain()
        assert cancelled
        assert commands.count("make_song_and_dance") == (1 if blocked_at == "song" else 0)
        assert commands.count("stop_music") == 1
        messages = [json.loads(message) for message in live.messages[stop_message_start:]]
        assert not any(
            message["channel"] == "speakable" and message["delegation_item_id"] == "song" for message in messages
        )
        assert live.app._song_task is None
    finally:
        released.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("inactive", ["no-connection", "shutdown"])
async def test_inactive_handler_cannot_execute_content_controls(
    live: DelegationHarness, monkeypatch: pytest.MonkeyPatch, inactive: str
) -> None:
    """Reject delegated control work before creating local side effects for inactive owners."""
    import dj_reachy_remix.openai_realtime as realtime

    commands: list[str] = []

    async def dispatch(name: str, arguments: str, deps: ToolDependencies) -> dict[str, str]:
        commands.append(name)
        return {"status": "completed"}

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    if inactive == "no-connection":
        live.app.connection = None
    else:
        live.app._shutdown_requested = True
    try:
        await live.event(request_event("pause", "pause music"))
        await live.drain()
        await asyncio.gather(*list(live.app._control_tasks))
        assert commands == []
    finally:
        live.app.connection = live.connection
        live.app._shutdown_requested = False


@pytest.mark.asyncio
async def test_additive_metadata_preserves_current_words_over_model_control(
    live: DelegationHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatible metadata must not promote model/commentary words into user controls."""
    import dj_reachy_remix.openai_realtime as realtime

    commands: list[str] = []

    async def dispatch(name: str, arguments: str, deps: ToolDependencies) -> dict[str, str]:
        commands.append(name)
        return {"status": "completed"}

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    await live.event({"type": "input_transcript.added", "item": {"text": "check my saved songs"}})
    wire = {
        "type": "delegation.created",
        "item": {
            "id": "compatible",
            "type": "delegation",
            "target": "client",
            "content": [
                {"type": "input_text", "text": "stop music", "extra": True},
                {"type": "output_text", "text": "resume music"},
            ],
        },
    }
    event = await live.event(wire)
    await live.final("check my saved songs")
    await live.drain()
    assert event.transcript_context == "check my saved songs"
    assert live.requests == ["check my saved songs"]
    assert commands == []
    await live.event(wire)
    await live.drain()
    assert live.requests == ["check my saved songs"]
    assert commands == []
