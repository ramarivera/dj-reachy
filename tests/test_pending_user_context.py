"""Pending actual user speech stays contextual, bounded and connection-local."""

from __future__ import annotations
import asyncio
from types import SimpleNamespace

import pytest

import dj_reachy.subscription_voice as voice
from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter, SubscriptionRealtimeConnection


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
