from __future__ import annotations

import pytest

from dj_reachy.subscription_voice import build_context_messages, plan_delegation, require_subscription_broker


def test_subscription_context_chunks_utf8_at_500_bytes() -> None:
    messages = build_context_messages("🌶️ música " * 100, channel="commentary", delegation_id="dlg-1")
    assert messages
    assert all(len(message["content"][0]["text"].encode("utf-8")) <= 500 for message in messages)
    assert all(message["type"] == "delegation.context.append" for message in messages)
    assert all(message["delegation_item_id"] == "dlg-1" for message in messages)


def test_delegated_work_uses_transcript_not_empty_event_payload() -> None:
    action = plan_delegation("", "Please pause the music now")
    assert action == ("pause_music", {})
    action = plan_delegation("", "Make an upbeat funk song")
    assert action is not None and action[0] == "make_song_and_dance"
    assert action[1]["prompt"] == "Make an upbeat funk song"


def test_subscription_backend_fails_closed_without_scoped_broker(monkeypatch) -> None:
    monkeypatch.delenv("DJ_REACHY_SUBSCRIPTION_BROKER_URL", raising=False)
    monkeypatch.delenv("DJ_REACHY_SUBSCRIPTION_BROKER_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="broker URL"):
        require_subscription_broker()
