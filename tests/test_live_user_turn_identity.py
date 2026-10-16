"""Local user-turn identity follows transcript finalization, not delegation consumption."""

from __future__ import annotations
from types import SimpleNamespace

import pytest

from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter, SubscriptionRealtimeConnection


async def emit(connection: SubscriptionRealtimeConnection, kind: str, **fields: object) -> SimpleNamespace:
    """Exercise the real queue and async iterator without opening transport."""
    connection._enqueue(SimpleNamespace(type=kind, **fields))
    event = await anext(connection)
    assert isinstance(event, SimpleNamespace)
    return event


async def delegate(connection: SubscriptionRealtimeConnection, identity: str, request: str = "") -> SimpleNamespace:
    """Submit a valid notice, optionally including canonical model-authored input text."""
    return await emit(
        connection,
        "delegation.created",
        item={
            "type": "delegation",
            "target": "client",
            "id": identity,
            "content": [{"type": "input_text", "text": request}] if request else [],
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_request", ["", "Make a roadtrip song"], ids=["transcript-backed", "canonical-content"])
async def test_early_delegation_does_not_split_or_reconsume_open_user_turn(model_request: str) -> None:
    """Two fragments separated by a handoff remain one turn through its final."""
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        first = await emit(connection, "input_transcript.added", item={"text": "Make a roadtrip song"})
        accepted = await delegate(connection, "early", model_request)
        assert accepted.transcript_context == "Make a roadtrip song"
        assert accepted.prior_user_context == ()
        trailing = await emit(connection, "input_transcript.added", item={"text": " with acoustic guitar"})
        second = await delegate(connection, "after-trailing")
        final = await emit(
            connection,
            "turn.done",
            turn={"role": "user", "transcript": "Make a roadtrip song with acoustic guitar"},
        )
        after_final = await delegate(connection, "after-final")

        assert [event.turn_id for event in (accepted, trailing, second, final, after_final)] == [first.turn_id] * 5
        assert second.transcript_context == ""
        assert after_final.transcript_context == ""
        assert not hasattr(second, "prior_user_context")
        assert not hasattr(after_final, "prior_user_context")
        assert connection._consumed_turn == first.turn_id
        assert connection._seen_delegations == {"early"}
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["final-only", "streamed"])
async def test_next_utterance_after_final_gets_fresh_identity_and_context(streamed: bool) -> None:
    """A final closes the old identity; retained context is one complete old utterance."""
    connection = SubscriptionRealtimeAdapter().connect()
    original = "Make a roadtrip song with acoustic guitar"
    try:
        first = await emit(connection, "input_transcript.added", item={"text": "Make a roadtrip song"})
        accepted = await delegate(connection, "early")
        await emit(connection, "input_transcript.added", item={"text": " with acoustic guitar"})
        final = await emit(connection, "turn.done", turn={"role": "user", "transcript": original})
        assert final.turn_id == accepted.turn_id == first.turn_id

        if streamed:
            next_first = await emit(connection, "input_transcript.added", item={"text": "Use a warm "})
            next_last = await emit(connection, "input_transcript.added", item={"text": "voice"})
            assert next_first.turn_id == next_last.turn_id == first.turn_id + 1
        next_final = await emit(connection, "turn.done", turn={"role": "user", "transcript": "Use a warm voice"})
        fresh = await delegate(connection, "next")
        assert fresh.turn_id == next_final.turn_id == first.turn_id + 1
        assert fresh.transcript_context == "Use a warm voice"
        assert fresh.prior_user_context == (original,)
        assert connection._seen_delegations == {"early", "next"}
        assert (await delegate(connection, "next-again")).transcript_context == ""
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_at", ["before-trailing", "before-final", "after-final"])
async def test_clear_through_early_handoff_never_recaches_its_trailing_speech(clear_at: str) -> None:
    """The clear-through floor covers late fragments and the old final, not fresh speech."""
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        await emit(connection, "turn.done", turn={"role": "user", "transcript": "Earlier request"})
        first = await emit(connection, "input_transcript.added", item={"text": "Make a roadtrip song"})
        accepted = await delegate(connection, "early")
        assert accepted.prior_user_context == ("Earlier request",)
        if clear_at == "before-trailing":
            connection.clear_pending_user_context(through_turn=accepted.turn_id)
        trailing = await emit(connection, "input_transcript.added", item={"text": " with acoustic guitar"})
        if clear_at == "before-final":
            connection.clear_pending_user_context(through_turn=accepted.turn_id)
        final = await emit(
            connection,
            "turn.done",
            turn={"role": "user", "transcript": "Make a roadtrip song with acoustic guitar"},
        )
        if clear_at == "after-final":
            connection.clear_pending_user_context(through_turn=accepted.turn_id)
        assert first.turn_id == accepted.turn_id == trailing.turn_id == final.turn_id
        assert (await delegate(connection, "stale")).transcript_context == ""

        await emit(connection, "input_transcript.added", item={"text": "A genuinely new request"})
        await emit(connection, "turn.done", turn={"role": "user", "transcript": "A genuinely new request"})
        fresh = await delegate(connection, "fresh")
        assert fresh.turn_id == accepted.turn_id + 1
        assert fresh.transcript_context == "A genuinely new request"
        assert fresh.prior_user_context == ()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_delayed_clear_after_early_handoff_preserves_genuinely_newer_speech() -> None:
    """Tool completion may arrive after the old final and several newer speech events."""
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        await emit(connection, "input_transcript.added", item={"text": "Old request"})
        accepted = await delegate(connection, "early")
        await emit(connection, "input_transcript.added", item={"text": " with details"})
        await emit(connection, "turn.done", turn={"role": "user", "transcript": "Old request with details"})
        newer = await emit(connection, "turn.done", turn={"role": "user", "transcript": "New request while awaiting"})
        current = await emit(connection, "input_transcript.added", item={"text": "New current details"})
        assert newer.turn_id == accepted.turn_id + 1
        assert current.turn_id == newer.turn_id + 1
        connection.clear_pending_user_context(through_turn=accepted.turn_id)
        fresh = await delegate(connection, "newer")
        assert fresh.transcript_context == "New current details"
        assert fresh.prior_user_context == ("New request while awaiting",)
        connection.clear_pending_user_context(through_turn=accepted.turn_id)
        final = await emit(connection, "turn.done", turn={"role": "user", "transcript": "New current details"})
        assert final.turn_id == current.turn_id
        await emit(connection, "turn.done", turn={"role": "user", "transcript": "Confirm newer request"})
        assert (await delegate(connection, "newest")).prior_user_context == (
            "New request while awaiting",
            "New current details",
        )
    finally:
        await connection.close()
