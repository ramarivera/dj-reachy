"""Pending actual user speech stays contextual, bounded and connection-local."""

from __future__ import annotations
import json
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

import dj_reachy_remix.config as app_config
import dj_reachy_remix.openai_realtime as realtime
import dj_reachy_remix.subscription_voice as voice
from dj_reachy_remix.voice_planner import DELEGATION_CONTEXT_PREFIX, PENDING_USER_CONTEXT_PREFIX
from dj_reachy_remix.runtime_settings import DEFAULT_SETTINGS
from dj_reachy_remix.tools.core_tools import ToolDependencies
from dj_reachy_remix.subscription_voice import SubscriptionRealtimeAdapter, SubscriptionRealtimeConnection


async def emit(connection: SubscriptionRealtimeConnection, kind: str, **fields: object) -> SimpleNamespace:
    """Exercise the real enqueue and async iterator without opening transport."""
    connection._enqueue(SimpleNamespace(type=kind, **fields))
    event = await anext(connection)
    assert isinstance(event, SimpleNamespace)
    return event


async def user(connection: SubscriptionRealtimeConnection, text: str) -> SimpleNamespace:
    """Complete one real user turn."""
    return await emit(connection, "turn.done", turn={"role": "user", "transcript": text})


async def delegate(connection: SubscriptionRealtimeConnection, identity: str, text: str = "") -> SimpleNamespace:
    """Admit canonical model content independently of actual user speech."""
    return await emit(
        connection,
        "delegation.created",
        item={
            "type": "delegation",
            "target": "client",
            "id": identity,
            "content": [{"type": "input_text", "text": text}],
        },
    )


@pytest.mark.asyncio
async def test_original_request_survives_without_initial_delegation() -> None:
    """A latest-confirmation-only canonical request must retain the original as context."""
    connection = SubscriptionRealtimeAdapter().connect()
    original = "Make a 10 second roadtrip song with acoustic guitar."
    try:
        first = await user(connection, original)
        assert not hasattr(first, "prior_user_context")
        await emit(connection, "turn.done", turn={"role": "assistant", "transcript": "Shall I make it?"})
        await user(connection, "Go ahead and make that song.")
        event = await delegate(connection, "confirmation", "Go ahead and make that song.")
        assert event.transcript_context == "Go ahead and make that song."
        assert event.prior_user_context == (original,)
        assert isinstance(event.prior_user_context, tuple)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_clarification_persists_until_explicit_clear_without_recaching_final() -> None:
    """No-call delegations retain context; app completion clears even an awaiting final."""
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        await user(connection, "Make a 10 second roadtrip song")
        await delegate(connection, "clarify")
        await user(connection, "With acoustic guitar")
        event = await delegate(connection, "clarify-again")
        assert event.prior_user_context == ("Make a 10 second roadtrip song",)
        await emit(connection, "input_transcript.added", item={"text": "Go ahead"})
        event = await delegate(connection, "confirmed", "Go ahead")
        assert event.prior_user_context == ("Make a 10 second roadtrip song", "With acoustic guitar")
        connection.clear_pending_user_context()
        await user(connection, "Go ahead")  # Final transcript for already consumed streaming input.
        stale = await delegate(connection, "stale", "model content")
        assert stale.transcript_context == ""
        assert not hasattr(stale, "prior_user_context")
        await user(connection, "A genuinely new request")
        fresh = await delegate(connection, "new")
        assert fresh.prior_user_context == ()
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["stop music", "cancel", "Please stop the song now!"])
async def test_current_stop_clears_pending_before_any_notice(stop: str) -> None:
    """Use the existing complete-control recognizer, not confirmation word heuristics."""
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        await user(connection, "Make a roadtrip song")
        await user(connection, stop)
        event = await delegate(connection, "stop", "make the roadtrip song")
        assert event.transcript_context == stop
        assert event.prior_user_context == ()
        await user(connection, "Go ahead")
        assert (await delegate(connection, "after-stop")).prior_user_context == ()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rejected_notices_and_no_fresh_speech_cannot_expose_or_consume_context() -> None:
    """Context never supplies admission or substitutes for a current transcript."""
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        await delegate(connection, "used", "model request")
        await user(connection, "Original user request")
        await user(connection, "Fresh confirmation")
        invalid = await emit(connection, "delegation.created", item={"type": "delegation", "target": "server"})
        duplicate = await delegate(connection, "used", "model request")
        assert not hasattr(invalid, "prior_user_context")
        assert not hasattr(duplicate, "prior_user_context")
        accepted = await delegate(connection, "valid")
        assert accepted.transcript_context == "Fresh confirmation"
        assert accepted.prior_user_context == ("Original user request",)
        for identity, request in [("empty", ""), ("content-only", "model request")]:
            event = await delegate(connection, identity, request)
            assert event.transcript_context == ""
            assert not hasattr(event, "prior_user_context")
        forged = await emit(connection, "output_transcript.added", prior_user_context=("not user speech",))
        assert not hasattr(forged, "prior_user_context")
        await user(connection, "Further details")
        assert (await delegate(connection, "further")).prior_user_context == (
            "Original user request",
            "Fresh confirmation",
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turns", "expected"),
    [
        ([str(index) for index in range(7)], ("3", "4", "5", "6")),
        (["é" * 2048, "界" * 1365, "zz"], ("界" * 1365, "zz")),
        (["é" * 4097, "intact"], ("intact",)),
        (["a" * 4096, "b" * 4096], ("a" * 4096, "b" * 4096)),
        (["bad\ud800", "valid"], ("valid",)),
        (["a" * 7000], ("a" * 7000,)),
    ],
)
async def test_count_and_utf8_bounds_drop_whole_oldest_turns(turns: list[str], expected: tuple[str, ...]) -> None:
    """Never truncate oversized pending speech or rewrite text to fit the cap."""
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        for text in turns:
            await user(connection, text)
        await user(connection, "current")
        event = await delegate(connection, "bounded")
        assert event.prior_user_context == expected
        assert len(event.prior_user_context) <= 4
        assert sum(len(text.encode("utf-8")) for text in event.prior_user_context) <= 8192
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_monotonic_expiry_including_current_turn_age(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expiry uses original observation age, not delegation or finalization time."""
    clock = [100.0]
    monkeypatch.setattr(voice, "monotonic", lambda: clock[0])
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        await emit(connection, "input_transcript.added", item={"text": "old"})
        clock[0] = 189.0
        await user(connection, "old finalized")
        await user(connection, "recent")
        clock[0] = 190.0
        assert (await delegate(connection, "boundary")).prior_user_context == ("old finalized",)
        clock[0] = 190.001
        await user(connection, "fresh")
        assert (await delegate(connection, "expired")).prior_user_context == ("recent",)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_close_and_new_connection_prevent_replay() -> None:
    """Late queued notices cannot reopen a closed carrier or cross connection ownership."""
    connection = SubscriptionRealtimeAdapter().connect()
    await user(connection, "old request")
    await user(connection, "confirmation")
    await connection.close()
    connection._enqueue(SimpleNamespace(type="turn.done", turn={"role": "user", "transcript": "late"}))
    with pytest.raises(StopAsyncIteration):
        await anext(connection)
    with pytest.raises(StopAsyncIteration):
        await anext(connection)
    fresh = SubscriptionRealtimeAdapter().connect()
    try:
        await user(fresh, "new speech")
        assert (await delegate(fresh, "new")).prior_user_context == ()
    finally:
        await fresh.close()


@pytest.mark.asyncio
async def test_expiry_while_iterator_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A receiver waiting on media must recheck context age when a notice arrives."""
    clock = [100.0]
    monkeypatch.setattr(voice, "monotonic", lambda: clock[0])
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        await user(connection, "old request")
        await user(connection, "fresh at the time")
        waiting = asyncio.create_task(anext(connection))
        await asyncio.sleep(0)
        clock[0] = 191.0
        connection._enqueue(
            SimpleNamespace(
                type="delegation.created",
                item={
                    "type": "delegation",
                    "target": "client",
                    "id": "late",
                    "content": [],
                },
            )
        )
        assert (await waiting).prior_user_context == ()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_clear_through_executed_turn_preserves_newer_speech() -> None:
    """Late app completion cannot erase speech received while its tool awaited."""
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        await user(connection, "old pending request")
        executed = await delegate(connection, "executed")
        await user(connection, "new request while tool awaited")
        await emit(connection, "input_transcript.added", item={"text": "new current details"})
        connection.clear_pending_user_context(through_turn=executed.turn_id)
        event = await delegate(connection, "newer")
        assert event.transcript_context == "new current details"
        assert event.prior_user_context == ("new request while tool awaited",)
        connection.clear_pending_user_context(through_turn=executed.turn_id)
        await user(connection, "new current details")
        await user(connection, "confirm newer request")
        assert (await delegate(connection, "newest")).prior_user_context == (
            "new request while tool awaited",
            "new current details",
        )
    finally:
        await connection.close()


class PendingHandlerHarness:
    """Retain the actual decoder, carrier, handler and domain around fake IO."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject owned settings, planner metadata and terminal dispatch only."""
        self.connection = SubscriptionRealtimeAdapter().connect()
        self.app = realtime.OpenaiRealtimeHandler(
            ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        )
        self.app.connection = self.connection
        self.messages: list[str] = []
        self.connection._channel = SimpleNamespace(readyState="open", send=self.messages.append)
        self.connection.ready.set()
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
        self.requests: list[tuple[str, list[dict[str, object]]]] = []
        self.assessments: list[tuple[str, dict[str, object]]] = []
        self.executed: list[tuple[str, dict[str, object]]] = []
        self.decision = "approve"
        self.tool = "make_song_and_dance"
        self.arguments: dict[str, object] = {
            "prompt": "Roadtrip song with acoustic guitar",
            "music_length_ms": 10000,
        }
        self.planning = asyncio.Event()
        self.planner_release: asyncio.Event | None = None

        async def plan(text: str, history: list[dict[str, object]]) -> dict[str, object]:
            self.requests.append((text, deepcopy(history)))
            if any(item.get("type") == "function_call_output" for item in history):
                return {"calls": [], "output": [], "text": "Fixture terminal result received."}
            self.planning.set()
            if self.planner_release is not None:
                await self.planner_release.wait()
            return {
                "calls": [{"name": self.tool, "call_id": "fixture-call", "arguments": deepcopy(self.arguments)}],
                "output": [],
            }

        async def assess(text: str, pending: dict[str, object]) -> dict[str, object]:
            self.assessments.append((text, deepcopy(pending)))
            return {
                "confirmation": {"decision": self.decision, "arguments": None},
                "text": "Fixture metadata decision; not real-model semantic proof.",
            }

        async def dispatch(name: str, arguments: str, deps: ToolDependencies) -> dict[str, str]:
            self.executed.append((name, json.loads(arguments)))
            return {"error": "FixtureTerminal", "message": "No provider job or device action was submitted."}

        monkeypatch.setattr(self.connection, "plan", plan)
        monkeypatch.setattr(self.connection, "plan_confirmation", assess)
        monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)

    async def user(self, text: str) -> SimpleNamespace:
        """Deliver a real final through the adapter and handler."""
        event = await user(self.connection, text)
        await self.app._handle_event(event)
        return event

    async def delegate(self, identity: str, text: str = "") -> SimpleNamespace:
        """Deliver a genuine Live delegation, never a hand-built decoded event."""
        event = await delegate(self.connection, identity, text)
        await self.app._handle_event(event)
        return event

    async def drain(self) -> None:
        """Join all currently owned delegation and confirmation work."""
        await asyncio.gather(*list(self.app._work_tasks | set(self.app._subscription_tasks.values())))


@pytest_asyncio.fixture
async def pending_handler(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[PendingHandlerHarness]:
    """Release blocked fake work and close every handler-owned task on failure."""
    harness = PendingHandlerHarness(monkeypatch)
    try:
        yield harness
    finally:
        if harness.planner_release is not None:
            harness.planner_release.set()
        await harness.app.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "decision"),
    [("Generate that exact preview", "approve"), ("No, do not make it", "decline"), ("Maybe later", "none")],
)
async def test_carried_original_intent_becomes_frozen_unbilled_draft_before_fresh_reply(
    pending_handler: PendingHandlerHarness,
    reply: str,
    decision: str,
) -> None:
    """Earlier intent is actual planner context, not consent from a model summary."""
    harness = pending_handler
    original = "Make a 10 second roadtrip song with acoustic guitar."
    current = "Go ahead and make that song."
    model_summary = "Make a long electronic song instead"
    await harness.user(original)
    await harness.user(current)
    delegated = await harness.delegate("candidate", model_summary)
    await harness.drain()
    assert delegated.transcript_context == current
    assert delegated.prior_user_context == (original,)
    assert harness.requests == [
        (
            current,
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": PENDING_USER_CONTEXT_PREFIX + json.dumps((original,)),
                },
                {"type": "message", "role": "assistant", "content": DELEGATION_CONTEXT_PREFIX + model_summary},
            ],
        )
    ]
    draft = harness.app._pending_music.pending
    assert draft is not None
    assert draft.original_user_text == current
    assert draft.original_turn == delegated.turn_id
    assert json.loads(draft.source_arguments_json) == {
        "prompt": "Roadtrip song with acoustic guitar",
        "music_length_ms": 10000,
    }
    assert draft.arguments == {
        "prompt": "Roadtrip song with acoustic guitar",
        "music_length_ms": 10000,
        "model_id": "music_v2_5",
        "output_format": "mp3_48000_192",
        "positive_styles": [],
        "negative_styles": [],
        "lyrics": "",
        "language": "auto",
        "force_instrumental": False,
    }
    assert "finetune_id" not in draft.arguments  # Unset optional fine-tuning is not executable data.
    assert harness.executed == []
    assert harness.assessments == []  # The original "Go ahead" is not fresh approval.
    assert draft.preview in "".join(
        part["text"] for message in map(json.loads, harness.messages) for part in message["content"]
    )
    harness.decision = decision
    await harness.user(reply)
    await harness.drain()
    assert harness.assessments == [(reply, draft.pending_payload([]))]
    assert harness.executed == ([("make_song_and_dance", draft.arguments)] if decision == "approve" else [])
    assert (harness.app._pending_music.pending is draft) == (decision == "none")
    assert len(harness.requests) == 1


@pytest.mark.asyncio
async def test_actual_handler_clear_uses_captured_turn_while_later_speech_arrives(
    pending_handler: PendingHandlerHarness,
) -> None:
    """A blocked real planner path must not clear newer carrier context on completion."""
    harness = pending_handler
    harness.tool, harness.arguments = "pause_music", {}
    harness.planner_release = asyncio.Event()
    await harness.user("Check my current saved-song request")
    captured = await harness.delegate("blocked", "Check saved music")
    await asyncio.wait_for(harness.planning.wait(), 1)
    await harness.user("New request while the planner awaited")
    fragment = await emit(harness.connection, "input_transcript.added", item={"text": "New current details"})
    await harness.app._handle_event(fragment)
    assert harness.executed == []
    harness.planner_release.set()
    await harness.drain()
    assert harness.executed == [("pause_music", {})]
    assert [request for request, _ in harness.requests] == [
        "Check my current saved-song request",
        "Check my current saved-song request",
    ]
    event = await delegate(harness.connection, "inspect-newer")
    assert event.turn_id > captured.turn_id
    assert event.transcript_context == "New current details"
    assert event.prior_user_context == ("New request while the planner awaited",)
    harness.connection.clear_pending_user_context(through_turn=captured.turn_id)
    await harness.user("New current details")
    await harness.user("Confirm the newer request")
    newest = await delegate(harness.connection, "inspect-newest")
    assert newest.prior_user_context == ("New request while the planner awaited", "New current details")
