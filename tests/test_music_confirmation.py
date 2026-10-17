"""Offline contract tests for app-owned music confirmation authority."""

from __future__ import annotations
import copy
import json
import socket
import importlib.util
from typing import Any, cast
from dataclasses import FrozenInstanceError

import pytest

from dj_reachy_remix.runtime_settings import DEFAULT_SETTINGS, validate_generation_request


@pytest.fixture(autouse=True)
def offline_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prohibit network and real SDK construction even on a regression."""
    import reachy_mini

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("offline domain test attempted external IO")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(reachy_mini, "ReachyMini", forbidden)


def test_prepare_freezes_expanded_request_without_activating() -> None:
    """Build a detached immutable request without interpreting consent or doing IO."""
    assert importlib.util.find_spec("dj_reachy_remix.music_confirmation") is not None
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings["music"].update(positive_styles=["jazz"], lyrics="private lyrics")
    request = {"prompt": "private musical idea"}
    expected = copy.deepcopy(validate_generation_request(request, settings))
    gate = PendingMusicConfirmation(clock=lambda: 10.0)
    draft = gate.prepare(request, settings, original_turn=1, original_user_text="actual user's request")
    assert gate.pending is None
    assert draft.arguments == expected
    assert draft.original_turn == 1
    assert draft.original_user_text == "actual user's request"
    assert draft.created_at == 10.0
    assert draft.id.isascii() and not any(c.isspace() for c in draft.id)
    assert len(draft.id) <= 200
    assert draft.revision == 0
    assert "private musical idea" in draft.preview
    assert "30000" in draft.preview and "music_v2_5" in draft.preview
    request["prompt"] = "mutated"
    settings["music"]["positive_styles"].append("mutated")
    exposed = draft.arguments
    exposed["positive_styles"].append("mutated")
    assert draft.arguments == expected
    with pytest.raises(FrozenInstanceError):
        setattr(draft, "id", "replacement")
    assert "private" not in repr(draft)
    assert "actual user's request" not in repr(draft)
    assert "private" not in repr(gate)


def test_pending_payload_is_exact_bounded_detached_rpc_data() -> None:
    """Export only local metadata and enforce UTF8 byte budgets."""
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    draft = PendingMusicConfirmation().prepare(
        {"prompt": "jazz"}, copy.deepcopy(DEFAULT_SETTINGS), original_turn=1, original_user_text="mi canción"
    )
    context = ["I offered this request."]
    payload = draft.pending_payload(context)
    assert payload == {
        "draft_id": draft.id,
        "request": draft.arguments,
        "preview": draft.preview,
        "original_user_text": "mi canción",
        "assistant_context": context,
    }
    assert draft.pending_payload()["assistant_context"] == []
    assert draft.pending_payload(None)["assistant_context"] == []
    context.append("later")
    payload["request"]["prompt"] = "tampered"
    assert draft.pending_payload()["request"]["prompt"] == "jazz"
    assert payload["assistant_context"] == ["I offered this request."]
    assert len(json.dumps(payload).encode()) <= 65536
    for invalid in (["x"] * 5, ["é" * 4097], ["\ud800"], [1], "not a list"):
        with pytest.raises(ValueError, match="confirmation"):
            draft.pending_payload(cast(list[str], invalid))


@pytest.mark.parametrize("reply", ["Haz exactamente esa canción", "それでお願いします", "no fixed approval phrase"])
def test_offered_request_consumes_once_only_on_injected_approval(reply: str) -> None:
    """A real fresh ticket, not any particular word, grants one frozen execution."""
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    settings = copy.deepcopy(DEFAULT_SETTINGS)
    gate = PendingMusicConfirmation()
    draft = gate.prepare({"prompt": "jazz"}, settings, original_turn=1, original_user_text="music please")
    assert gate.pending is None
    gate.offer(draft, epoch=0, presented_turn=1)
    assert gate.pending is draft
    ticket = gate.begin_reply(turn=2, text=reply, epoch=0)
    assert ticket is not None
    assert ticket.draft_id == draft.id and ticket.draft_revision == draft.revision
    assert ticket.reply_turn == 2 and ticket.reply_text == reply and ticket.epoch == 0
    assert reply not in repr(ticket)
    with pytest.raises(FrozenInstanceError):
        ticket.epoch = 1
    approved = {"decision": "approve", "arguments": None, "text": "private result"}
    result = gate.apply(ticket, approved, epoch=0, current_turn=2, settings=settings)
    assert result.kind == "approved"
    assert result.arguments == draft.arguments
    assert result.draft is None and result.text == "private result"
    assert "private result" not in repr(result)
    assert "jazz" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.kind = "none"
    assert gate.pending is None
    assert gate.apply(ticket, approved, epoch=0, current_turn=2, settings=settings).kind == "stale"


@pytest.mark.parametrize(
    "options,turn,text",
    [
        ({"prompt": "jazz"}, 0, "real"),
        ({"prompt": "jazz"}, True, "real"),
        ({"prompt": "jazz"}, 2**63, "real"),
        ({"prompt": "jazz"}, 1, "  "),
        ({"prompt": "jazz"}, 1, "é" * 6001),
        ({"prompt": "jazz"}, 1, "\ud800"),
        ({"prompt": "\ud800"}, 1, "real"),
        ({"prompt": "é" * 5000}, 1, "real"),
        ({"prompt": "jazz", "provider_id": "not-local"}, 1, "real"),
        ({"prompt": "jazz", "music_length_ms": True}, 1, "real"),
        ({"prompt": "jazz", "music_length_ms": 2999}, 1, "real"),
        ({"prompt": "jazz", "music_length_ms": 600001}, 1, "real"),
        ({"prompt": "jazz", "voice_id": "unsupported"}, 1, "real"),
    ],
    ids=[
        "zero-turn",
        "bool-turn",
        "huge-turn",
        "blank",
        "utf8-size",
        "surrogate-origin",
        "surrogate-prompt",
        "unpresentable-preview",
        "provider-id",
        "bool-duration",
        "short-duration",
        "long-duration",
        "voice-id",
    ],
)
def test_prepare_rejects_invalid_authority_or_unpresentable_request(options: Any, turn: Any, text: Any) -> None:
    """Reject malformed origins and requests before they can become approvable."""
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    gate = PendingMusicConfirmation()
    with pytest.raises(ValueError, match="confirmation"):
        gate.prepare(options, copy.deepcopy(DEFAULT_SETTINGS), original_turn=turn, original_user_text=text)
    assert gate.pending is None


@pytest.fixture
def offered() -> tuple[Any, Any, dict[str, Any], list[float]]:
    """Offer a real frozen request with a deterministic monotonic lifetime."""
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    now = [10.0]
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    gate = PendingMusicConfirmation(clock=lambda: now[0])
    draft = gate.prepare({"prompt": "jazz"}, settings, original_turn=1, original_user_text="actual music request")
    gate.offer(draft, epoch=7, presented_turn=2)
    return gate, draft, settings, now


@pytest.mark.parametrize(
    "turn,text,epoch",
    [
        (1, "yes", 7),
        (2, "yes", 7),
        (True, "yes", 7),
        (2**63, "yes", 7),
        (3, "", 7),
        (3, "   ", 7),
        (3, "é" * 6001, 7),
        (3, "\ud800", 7),
        (3, "yes", True),
        (3, "yes", -1),
        (3, "yes", 2**63),
        (3, "yes", 6),
        (3, "yes", 8),
    ],
    ids=[
        "original",
        "presentation",
        "bool-turn",
        "huge-turn",
        "empty",
        "blank",
        "oversize",
        "surrogate",
        "bool-epoch",
        "negative-epoch",
        "huge-epoch",
        "old-epoch",
        "new-epoch",
    ],
)
def test_begin_reply_requires_fresh_bounded_actual_turn(offered: Any, turn: Any, text: Any, epoch: Any) -> None:
    """No missing, oversized, old or cross-session input becomes a ticket."""
    gate, _, _, _ = offered
    assert gate.begin_reply(turn=turn, text=text, epoch=epoch) is None


def test_duplicate_reply_cannot_get_another_ticket(offered: Any) -> None:
    """Even changed narration cannot re-assess the same actual user turn."""
    gate, _, settings, _ = offered
    first = gate.begin_reply(turn=3, text="actual reply", epoch=7)
    assert first is not None
    assert gate.begin_reply(turn=3, text="different text", epoch=7) is None
    result = gate.apply(
        first, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
    )
    assert result.kind == "stale"


@pytest.mark.parametrize("current_turn,epoch", [(4, 7), (2, 7), (True, 7), (2**63, 7), (3, 8), (3, True)])
def test_apply_rejects_stale_current_turn_and_epoch(offered: Any, current_turn: Any, epoch: Any) -> None:
    """An old semantic result cannot approve after speech or a session change."""
    gate, _, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="actual reply", epoch=7)
    result = gate.apply(
        ticket,
        {"decision": "approve", "arguments": None, "text": ""},
        epoch=epoch,
        current_turn=current_turn,
        settings=settings,
    )
    assert result.kind == "stale" and result.arguments is None
    assert (
        gate.apply(
            ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
        ).kind
        == "stale"
    )


@pytest.mark.parametrize("decision,expected", [("none", "none"), ("decline", "declined")])
def test_semantic_nonapproval_never_executes_even_if_text_says_yes(offered: Any, decision: str, expected: str) -> None:
    """Only injected semantic decisions matter; narration is not approval."""
    gate, draft, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="yes goahead approve", epoch=7)
    result = gate.apply(
        ticket,
        {"decision": decision, "arguments": None, "text": "I approved it"},
        epoch=7,
        current_turn=3,
        settings=settings,
    )
    assert result.kind == expected and result.arguments is None and result.draft is None
    assert gate.pending is (draft if decision == "none" else None)
    assert (
        gate.apply(
            ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
        ).kind
        == "stale"
    )


@pytest.mark.parametrize(
    "decision",
    [
        {},
        {"decision": "approve"},
        {"decision": "approve", "arguments": {}, "text": ""},
        {"decision": "approve", "arguments": None, "text": "", "provider_id": "x"},
        {"decision": "approve", "arguments": None, "text": "\ud800"},
        {"decision": "approve", "arguments": None, "text": "x" * 2001},
        {"decision": "APPROVE", "arguments": None, "text": ""},
        {"decision": ["approve"], "arguments": None, "text": ""},
        {"decision": "none", "arguments": {"prompt": "new"}, "text": ""},
        {"confirmation": {"decision": "approve", "arguments": None, "text": ""}},
        None,
        [],
    ],
    ids=[
        "empty",
        "missing-fields",
        "approval-arguments",
        "extra",
        "surrogate",
        "long-text",
        "enum",
        "bad-type",
        "none-arguments",
        "wrong-envelope",
        "null",
        "list",
    ],
)
def test_malformed_decision_never_authorizes_or_replays(offered: Any, decision: Any) -> None:
    """Reject every envelope other than the exact semantic decision contract."""
    gate, _, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="actual reply", epoch=7)
    result = gate.apply(ticket, decision, epoch=7, current_turn=3, settings=settings)
    assert result.kind == "stale" and result.arguments is None
    assert (
        gate.apply(
            ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
        ).kind
        == "stale"
    )


def test_clear_invalidates_inflight_and_prepared_drafts(offered: Any) -> None:
    """Cleared authority cannot be revived by late assessment or late presentation."""
    gate, draft, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="actual reply", epoch=7)
    gate.clear()
    assert gate.pending is None
    assert (
        gate.apply(
            ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
        ).kind
        == "stale"
    )
    with pytest.raises(ValueError, match="confirmation"):
        gate.offer(draft, epoch=7, presented_turn=3)
    detached = gate.prepare({"prompt": "new"}, settings, original_turn=4, original_user_text="new actual request")
    gate.clear()
    with pytest.raises(ValueError, match="confirmation"):
        gate.offer(detached, epoch=7, presented_turn=4)


@pytest.mark.parametrize("boundary", ["pending", "reply", "apply", "offer"])
def test_original_lifetime_expires_at_exactly_ninety_seconds(offered: Any, boundary: str) -> None:
    """All authority boundaries prune both active and unoffered expired requests."""
    gate, _, settings, now = offered
    ticket = gate.begin_reply(turn=3, text="actual reply", epoch=7)
    if boundary == "offer":
        draft = gate.prepare({"prompt": "new"}, settings, original_turn=4, original_user_text="new actual request")
    now[0] = 100.0
    if boundary == "reply":
        assert gate.begin_reply(turn=4, text="yes", epoch=7) is None
    elif boundary == "apply":
        assert (
            gate.apply(
                ticket,
                {"decision": "approve", "arguments": None, "text": ""},
                epoch=7,
                current_turn=3,
                settings=settings,
            ).kind
            == "stale"
        )
    elif boundary == "offer":
        with pytest.raises(ValueError, match="confirmation"):
            gate.offer(draft, epoch=7, presented_turn=4)
    assert gate.pending is None


@pytest.mark.parametrize("epoch,turn", [(True, 1), (-1, 1), (2**63, 1), (0, 0), (0, True), (0, 2**63)])
def test_offer_rejects_invalid_presentation_counters(epoch: Any, turn: Any) -> None:
    """Activation requires bounded local session and actual presentation counters."""
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    gate = PendingMusicConfirmation()
    draft = gate.prepare(
        {"prompt": "jazz"}, copy.deepcopy(DEFAULT_SETTINGS), original_turn=1, original_user_text="actual request"
    )
    with pytest.raises(ValueError, match="confirmation"):
        gate.offer(draft, epoch=epoch, presented_turn=turn)
    assert gate.pending is None


@pytest.mark.parametrize("action", ["copy", "foreign", "reoffer", "replace"])
def test_only_latest_locally_prepared_draft_can_be_offered(offered: Any, action: str) -> None:
    """Equal-looking values and delayed presentations cannot manufacture authority."""
    from dataclasses import replace

    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    gate, draft, settings, _ = offered
    if action == "copy":
        draft = replace(draft)
    elif action == "foreign":
        draft = PendingMusicConfirmation().prepare(
            {"prompt": "jazz"}, settings, original_turn=1, original_user_text="real"
        )
    elif action == "replace":
        gate.prepare({"prompt": "new"}, settings, original_turn=4, original_user_text="new request")
    with pytest.raises(ValueError, match="confirmation"):
        gate.offer(draft, epoch=7, presented_turn=4)


@pytest.mark.parametrize("last", ["none", "approve"])
def test_four_distinct_reply_budget_is_bounded_without_blocking_fourth_approval(offered: Any, last: str) -> None:
    """Three ambiguous replies leave one chance, but never an unlimited conversation."""
    gate, draft, settings, _ = offered
    for turn in range(3, 6):
        ticket = gate.begin_reply(turn=turn, text="ordinary conversation", epoch=7)
        assert ticket is not None
        assert (
            gate.apply(
                ticket,
                {"decision": "none", "arguments": None, "text": ""},
                epoch=7,
                current_turn=turn,
                settings=settings,
            ).kind
            == "none"
        )
        assert gate.pending is draft
    ticket = gate.begin_reply(turn=6, text="actual final reply", epoch=7)
    assert ticket is not None
    result = gate.apply(
        ticket, {"decision": last, "arguments": None, "text": ""}, epoch=7, current_turn=6, settings=settings
    )
    assert result.kind == ("approved" if last == "approve" else "none")
    assert result.arguments == (draft.arguments if last == "approve" else None)
    assert gate.pending is None
    assert gate.begin_reply(turn=7, text="fifth reply", epoch=7) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt": "salsa", "music_length_ms": 30000},
        {"prompt": "jazz", "music_length_ms": 10000},
        {"prompt": "salsa", "music_length_ms": 10000},
    ],
)
def test_revision_is_an_unoffered_frozen_preview_requiring_a_new_reply(offered: Any, changes: Any) -> None:
    """Corrections never execute or reuse approval for a different request."""
    gate, original, settings, now = offered
    now[0] = 30.0
    ticket = gate.begin_reply(turn=3, text="make that different", epoch=7)
    result = gate.apply(
        ticket,
        {"decision": "revise", "arguments": changes, "text": "updated preview"},
        epoch=7,
        current_turn=3,
        settings=settings,
    )
    assert result.kind == "revised" and result.arguments is None
    draft = result.draft
    assert draft is not None
    assert draft.id == original.id and draft.revision == original.revision + 1
    assert draft.created_at == original.created_at
    assert draft.original_turn == original.original_turn and draft.original_user_text == original.original_user_text
    expected = original.arguments | changes
    assert draft.arguments == expected
    assert gate.pending is None
    gate.offer(draft, epoch=7, presented_turn=3)
    assert gate.begin_reply(turn=3, text="approve", epoch=7) is None
    assert (
        gate.apply(
            ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
        ).kind
        == "stale"
    )
    fresh = gate.begin_reply(turn=4, text="that request exactly", epoch=7)
    result = gate.apply(
        fresh, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=4, settings=settings
    )
    assert result.kind == "approved" and result.arguments == expected


@pytest.mark.parametrize(
    "changes",
    [
        None,
        {},
        {"prompt": "", "music_length_ms": 30000},
        {"prompt": "jazz", "music_length_ms": True},
        {"prompt": "jazz", "music_length_ms": 1000},
        {"prompt": "jazz", "music_length_ms": 30000, "model_id": "music_v1"},
        {"prompt": "new", "music_length_ms": 30000, "lyrics": "not-authorized"},
        {"prompt": "\ud800", "music_length_ms": 30000},
    ],
)
def test_revision_cannot_smuggle_execution_options(offered: Any, changes: Any) -> None:
    """The classifier may correct only validated prompt/duration, never execute."""
    gate, _, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="correction", epoch=7)
    result = gate.apply(
        ticket, {"decision": "revise", "arguments": changes, "text": ""}, epoch=7, current_turn=3, settings=settings
    )
    assert result.kind == "stale" and result.arguments is None and result.draft is None


@pytest.mark.parametrize("choice", ["approve", "none", "revise"])
def test_music_settings_change_repreviews_instead_of_executing(offered: Any, choice: str) -> None:
    """Consent is invalid for newly resolved defaults, even with the same prompt."""
    gate, original, settings, now = offered
    ticket = gate.begin_reply(turn=3, text="the exact request", epoch=7)
    settings["music"].update(
        model_id="music_v2",
        duration_ms=15000,
        positive_styles=["warm"],
        lyrics="new private lyrics",
        output_format="mp3_44100_128",
    )
    now[0] = 50
    decision = {
        "decision": choice,
        "arguments": {"prompt": "salsa", "music_length_ms": 15000} if choice == "revise" else None,
        "text": "",
    }
    result = gate.apply(ticket, decision, epoch=7, current_turn=3, settings=settings)
    assert result.kind == "revised" and result.arguments is None
    draft = result.draft
    assert draft is not None and gate.pending is None
    assert draft.created_at == original.created_at and draft.original_user_text == original.original_user_text
    expected = validate_generation_request({"prompt": "salsa" if choice == "revise" else "jazz"}, settings)
    assert draft.arguments == expected
    gate.offer(draft, epoch=7, presented_turn=3)
    fresh = gate.begin_reply(turn=4, text="the new request", epoch=7)
    result = gate.apply(
        fresh, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=4, settings=settings
    )
    assert result.kind == "approved" and result.arguments == expected


def test_stale_current_turn_clears_its_authority_but_not_a_newer_ticket(offered: Any) -> None:
    """Late completions may neither revive old speech nor cancel a newer assessment."""
    gate, _, settings, _ = offered
    old = gate.begin_reply(turn=3, text="old real reply", epoch=7)
    fresh = gate.begin_reply(turn=4, text="new actual correction", epoch=7)
    approve = {"decision": "approve", "arguments": None, "text": ""}
    assert gate.apply(old, approve, epoch=7, current_turn=4, settings=settings).kind == "stale"
    assert gate.pending is not None
    assert gate.apply(fresh, approve, epoch=7, current_turn=5, settings=settings).kind == "stale"
    assert gate.pending is None


@pytest.mark.parametrize("boundary", ["begin", "apply"])
def test_cross_epoch_cannot_return_to_old_authority(offered: Any, boundary: str) -> None:
    """A different current session invalidates drafts, even if clear was missed."""
    gate, _, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="old reply", epoch=7)
    if boundary == "begin":
        assert gate.begin_reply(turn=4, text="new epoch", epoch=8) is None
    else:
        assert (
            gate.apply(
                ticket,
                {"decision": "approve", "arguments": None, "text": ""},
                epoch=8,
                current_turn=3,
                settings=settings,
            ).kind
            == "stale"
        )
    assert gate.pending is None
    assert gate.begin_reply(turn=5, text="old epoch", epoch=7) is None


@pytest.mark.parametrize("forged", [None, {}, "ticket"])
def test_non_ticket_never_bypasses_offer_state(offered: Any, forged: Any) -> None:
    """Absent tickets are not accidentally equal to the idle ticket slot."""
    gate, _, settings, _ = offered
    result = gate.apply(
        forged, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
    )
    assert result.kind == "stale" and result.arguments is None


def test_speech_before_preview_offer_invalidates_the_prepared_authority(offered: Any) -> None:
    """An in-flight preview cannot be activated after a newer genuine user turn."""
    gate, _, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="make it salsa", epoch=7)
    revised = gate.apply(
        ticket,
        {"decision": "revise", "arguments": {"prompt": "salsa", "music_length_ms": 30000}, "text": ""},
        epoch=7,
        current_turn=3,
        settings=settings,
    ).draft
    assert revised is not None
    assert gate.begin_reply(turn=4, text="actually stop that", epoch=7) is None
    with pytest.raises(ValueError, match="confirmation"):
        gate.offer(revised, epoch=7, presented_turn=3)


def test_builder_receives_a_detached_settings_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The validated execution options and settings fingerprint share one snapshot."""
    import dj_reachy_remix.music_confirmation as domain

    settings = copy.deepcopy(DEFAULT_SETTINGS)
    expected = copy.deepcopy(validate_generation_request({"prompt": "jazz"}, settings))

    def validate(arguments: Any, full_settings: Any) -> Any:
        settings["music"]["duration_ms"] = 10000
        return validate_generation_request(arguments, full_settings)

    monkeypatch.setattr(domain, "validate_generation_request", validate)
    draft = domain.PendingMusicConfirmation().prepare(
        {"prompt": "jazz"}, settings, original_turn=1, original_user_text="actual request"
    )
    assert draft.arguments == expected
    assert json.loads(draft.music_settings_json)["duration_ms"] == expected["music_length_ms"]


@pytest.mark.parametrize("now", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_clock_cannot_create_a_draft(now: float) -> None:
    """Invalid lifetime origins fail before publishing any preview value."""
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    gate = PendingMusicConfirmation(clock=lambda: now)
    with pytest.raises(ValueError, match="confirmation"):
        gate.prepare(
            {"prompt": "jazz"}, copy.deepcopy(DEFAULT_SETTINGS), original_turn=1, original_user_text="real request"
        )
    assert gate.pending is None


def test_concurrent_approvals_consume_exactly_once(offered: Any) -> None:
    """Actual simultaneous callers cannot extract multiple execution grants."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    gate, draft, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="the exact request", epoch=7)
    ready = threading.Barrier(8)

    def approve(_: int) -> Any:
        ready.wait(timeout=5)
        return gate.apply(
            ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(approve, range(8)))
    assert sum(item.kind == "approved" for item in results) == 1
    assert sum(item.kind == "stale" for item in results) == 7
    winner = next(item for item in results if item.kind == "approved")
    assert winner.arguments == draft.arguments
    winner.arguments["positive_styles"].append("mutation")
    assert winner.arguments == draft.arguments
    assert gate.pending is None


@pytest.mark.parametrize("boundary", ["revision", "settings"])
def test_budget_survives_every_new_preview(offered: Any, boundary: str) -> None:
    """A corrected request cannot reset the four-distinct-turn allowance."""
    gate, original, settings, _ = offered
    for turn in range(3, 7):
        ticket = gate.begin_reply(turn=turn, text="a correction", epoch=7)
        if boundary == "settings":
            settings["music"]["duration_ms"] += 1000
        decision = {
            "decision": "revise" if boundary == "revision" else "approve",
            "arguments": {"prompt": f"revision {turn}", "music_length_ms": 30000} if boundary == "revision" else None,
            "text": "",
        }
        result = gate.apply(ticket, decision, epoch=7, current_turn=turn, settings=settings)
        assert result.kind == "revised" and result.arguments is None
        assert result.draft.created_at == original.created_at
        if turn < 6:
            gate.offer(result.draft, epoch=7, presented_turn=turn)
        else:
            with pytest.raises(ValueError, match="confirmation"):
                gate.offer(result.draft, epoch=7, presented_turn=turn)
    assert gate.pending is None


def test_revised_request_does_not_extend_original_expiry(offered: Any) -> None:
    """A late correction still expires ninety seconds after the original prepare."""
    gate, _, settings, now = offered
    now[0] = 99.999
    ticket = gate.begin_reply(turn=3, text="a correction", epoch=7)
    result = gate.apply(
        ticket,
        {"decision": "revise", "arguments": {"prompt": "jazz", "music_length_ms": 10000}, "text": ""},
        epoch=7,
        current_turn=3,
        settings=settings,
    )
    assert result.kind == "revised" and result.draft.created_at == 10
    gate.offer(result.draft, epoch=7, presented_turn=3)
    fresh = gate.begin_reply(turn=4, text="the corrected request", epoch=7)
    now[0] = 100
    assert (
        gate.apply(
            fresh, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=4, settings=settings
        ).kind
        == "stale"
    )
    assert gate.pending is None


def test_invalid_settings_clear_instead_of_retaining_old_approval(offered: Any) -> None:
    """Broken current settings cannot authorize either the old or a new request."""
    gate, _, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="the exact request", epoch=7)
    settings["music"]["duration_ms"] = True
    result = gate.apply(
        ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
    )
    assert result.kind == "stale" and result.arguments is None and gate.pending is None


def test_unrelated_settings_changes_do_not_rewrite_music(offered: Any) -> None:
    """Voice and motion configuration are deliberately separate from music consent."""
    gate, draft, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="the exact request", epoch=7)
    settings["voice"]["language"] = "es"
    settings["motion"]["intensity"] = 0.1
    result = gate.apply(
        ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
    )
    assert result.kind == "approved" and result.arguments == draft.arguments


def test_explicit_source_overrides_survive_new_defaults() -> None:
    """Repreview uses the source request, preserving explicit user-selected options."""
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    settings = copy.deepcopy(DEFAULT_SETTINGS)
    source = {"prompt": "jazz", "music_length_ms": 10000, "positive_styles": ["warm"]}
    gate = PendingMusicConfirmation()
    original = gate.prepare(source, settings, original_turn=1, original_user_text="my actual request")
    gate.offer(original, epoch=0, presented_turn=1)
    ticket = gate.begin_reply(turn=2, text="the request", epoch=0)
    assert ticket is not None
    settings["music"].update(duration_ms=60000, positive_styles=["cold"], model_id="music_v2")
    result = gate.apply(
        ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=0, current_turn=2, settings=settings
    )
    assert result.kind == "revised" and result.arguments is None
    assert result.draft is not None
    assert result.draft.arguments == validate_generation_request(source, settings)
    assert result.draft.arguments["positive_styles"] == ["warm"]
    assert result.draft.arguments["music_length_ms"] == 10000


def test_forged_equal_ticket_cannot_consume_real_ticket(offered: Any) -> None:
    """Only the actual issued capability works, not a reconstructed equal value."""
    from dataclasses import replace

    gate, draft, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="the exact request", epoch=7)
    decision = {"decision": "approve", "arguments": None, "text": ""}
    assert gate.apply(replace(ticket), decision, epoch=7, current_turn=3, settings=settings).kind == "stale"
    result = gate.apply(ticket, decision, epoch=7, current_turn=3, settings=settings)
    assert result.kind == "approved" and result.arguments == draft.arguments


def test_domain_never_logs_private_content(offered: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Semantic payload, lyrics, and transcripts never become diagnostics."""
    gate, _, settings, _ = offered
    settings["music"]["lyrics"] = "private lyric marker"
    draft = gate.prepare(
        {"prompt": "private prompt marker"}, settings, original_turn=5, original_user_text="private origin marker"
    )
    gate.offer(draft, epoch=7, presented_turn=5)
    ticket = gate.begin_reply(turn=6, text="private reply marker", epoch=7)
    result = gate.apply(
        ticket,
        {"decision": "approve", "arguments": None, "text": "private result marker"},
        epoch=7,
        current_turn=6,
        settings=settings,
    )
    assert result.kind == "approved"
    assert "private" not in repr((draft, ticket, result, gate))
    assert "marker" not in caplog.text


def test_expiry_during_offline_validation_cannot_return_a_late_approval(
    offered: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final consume boundary rechecks time, not just assessment arrival."""
    import dj_reachy_remix.music_confirmation as domain

    gate, _, settings, now = offered
    ticket = gate.begin_reply(turn=3, text="the exact request", epoch=7)

    def validate(arguments: Any, full_settings: Any) -> Any:
        result = validate_generation_request(arguments, full_settings)
        now[0] = 100
        return result

    monkeypatch.setattr(domain, "validate_generation_request", validate)
    result = gate.apply(
        ticket, {"decision": "approve", "arguments": None, "text": ""}, epoch=7, current_turn=3, settings=settings
    )
    assert result.kind == "stale" and result.arguments is None and gate.pending is None


def test_same_original_turn_before_offer_is_not_a_fresh_reply() -> None:
    """Delayed original ASR is neither approval nor newer speech cancelling the draft."""
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    gate = PendingMusicConfirmation()
    draft = gate.prepare(
        {"prompt": "jazz"}, copy.deepcopy(DEFAULT_SETTINGS), original_turn=2, original_user_text="actual music request"
    )
    assert gate.begin_reply(turn=2, text="actual music request", epoch=0) is None
    gate.offer(draft, epoch=0, presented_turn=3)
    assert gate.begin_reply(turn=3, text="pre-preview speech", epoch=0) is None
    assert gate.begin_reply(turn=4, text="fresh actual reply", epoch=0) is not None


@pytest.mark.parametrize(
    "changes",
    [{"prompt": "salsa"}, {"music_length_ms": 10000}, {"prompt": "x" * 2001, "music_length_ms": 10000}],
    ids=["missing-duration", "missing-prompt", "oversize-normal-prompt"],
)
def test_revision_requires_the_exact_normal_prompt_duration_shape(offered: Any, changes: Any) -> None:
    """Domain decisions obey the same complete correction shape as the semantic RPC."""
    gate, _, settings, _ = offered
    ticket = gate.begin_reply(turn=3, text="a correction", epoch=7)
    result = gate.apply(
        ticket, {"decision": "revise", "arguments": changes, "text": ""}, epoch=7, current_turn=3, settings=settings
    )
    assert result.kind == "stale" and result.arguments is None and result.draft is None
