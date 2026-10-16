"""Canonical Codex decoding compatibility within local authorization bounds."""
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import dj_reachy.subscription_voice as voice
from dj_reachy.openai_realtime import OpenaiRealtimeHandler
from dj_reachy.tools.core_tools import ToolDependencies
from dj_reachy.subscription_voice import subscription_delegation


@pytest.mark.parametrize("text", [None, 7, [], {}, "\ud800", "x" * 6001, "🌶" * 3001])
def test_actual_input_text_remains_fail_closed(text: object) -> None:
    """Reject malformed actual input text even alongside a valid part."""
    assert subscription_delegation(item([
        {"type": "input_text", "text": "valid"},
        {"type": "input_text", "text": text, "extra": True},
    ])) is None


@pytest.mark.parametrize("fields", [
    {"type": "message"}, {"target": "server"}, {"id": " "}, {"id": 7},
    {"id": "x" * 201}, {"content": "text"},
    {"content": [{"type": "input_text", "text": ""}] * 65},
])
def test_local_authorization_bounds_remain(fields: dict[str, object]) -> None:
    """Keep item identity, target and content container validation unchanged."""
    value = item([{"type": "input_text", "text": "check availability"}])
    value.update(fields)
    assert subscription_delegation(value) is None


@pytest.mark.parametrize("text", ["x" * 6000, "🌶" * 3000])
def test_aggregate_limits_accept_exact_boundary(text: str) -> None:
    """Apply both local aggregate limits without counting invented separators."""
    assert subscription_delegation(item([
        {"type": "input_text", "text": text[:1500], "extra": True},
        {"type": "input_text", "text": text[1500:]},
    ])) == ("decoder-1", text)


@pytest.mark.asyncio
async def test_mixed_content_reaches_real_handler_and_broker_envelope_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain production scheduling and plan encoding with only network replaced."""
    connection = voice.SubscriptionRealtimeAdapter().connect()
    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    app.connection = connection
    envelopes: list[dict[str, Any]] = []

    async def broker(body: dict[str, Any], *, operation: str = "session") -> dict[str, Any]:
        """Capture the actual plan envelope before external I/O."""
        assert operation == "plan"
        envelopes.append(body)
        return {"calls": [], "output": [], "text": "Availability checked."}

    monkeypatch.setattr(voice, "broker_request", broker)
    wire = item([
        {"type": "input_text", "text": "check avail", "extra": True},
        {"type": "output_text", "text": "stop music"},
        {"type": "input_text", "text": "ability"},
    ])
    try:
        for _ in range(2):
            connection._enqueue(SimpleNamespace(type="delegation.created", item=wire))
            event = await anext(connection)
            assert event.transcript_context == ""
            await app._handle_event(event)
            await asyncio.gather(*list(app._subscription_tasks.values()))
        assert len(envelopes) == 1
        assert envelopes[0]["billing"] == "subscription"
        assert envelopes[0]["transcript"] == (
            "Live model delegation request (not a verbatim user transcript):\ncheck availability"
        )
        assert isinstance(envelopes[0]["history"], list)
        assert connection._consumed_turn == -1
    finally:
        await app.shutdown()


@pytest.mark.parametrize("text", ["x" * 6001, "🌶" * 3001])
def test_aggregate_limits_reject_split_overflow(text: str) -> None:
    """Reject overflow across individually valid parts, not merely per part."""
    assert subscription_delegation(item([
        {"type": "input_text", "text": text[:1500]},
        {"type": "input_text", "text": text[1500:]},
    ])) is None


def test_missing_actual_text_is_not_silently_filtered() -> None:
    """Deliberately retain stricter local handling than official filter_map."""
    assert subscription_delegation(item([
        {"type": "input_text", "text": "valid"}, {"type": "input_text"},
    ])) is None


def test_maximum_id_and_content_count_still_accepted() -> None:
    """Retain inclusive local identity and list bounds."""
    value = item([{"type": "input_text", "text": "x"}] * 64)
    value["id"] = "d" * 200
    assert subscription_delegation(value) == ("d" * 200, "x" * 64)


def item(content: object) -> dict[str, object]:
    """Build a canonical client item, including intentionally invalid content."""
    return {"type": "delegation", "target": "client", "id": "decoder-1", "content": content}


def test_input_text_concatenates_without_invented_separators() -> None:
    """Match pinned Codex collect::<String>() semantics."""
    assert subscription_delegation(item([
        {"type": "input_text", "text": "check avail"},
        {"type": "input_text", "text": "ability"},
    ])) == ("decoder-1", "check availability")


def test_unrelated_parts_never_promote_their_text() -> None:
    """Filter unrelated JSON values without turning commentary into a command."""
    assert subscription_delegation(item([
        {"type": "output_text", "text": "stop music"},
        None, 17, "unrelated", {"text": "resume music"},
        {"type": "input_text", "text": "check availability"},
    ])) == ("decoder-1", "check availability")


def test_unrelated_only_content_cannot_authorize_user_transcript() -> None:
    """Do not broaden the local legacy transcript-backed admission path."""
    assert subscription_delegation(item([
        {"type": "output_text", "text": "stop music"},
    ])) is None


def test_input_text_allows_additive_metadata() -> None:
    """Accept additive fields rather than requiring an exact key set."""
    assert subscription_delegation(item([
        {"type": "input_text", "text": "check availability", "metadata": {"source": "live"}},
    ])) == ("decoder-1", "check availability")
