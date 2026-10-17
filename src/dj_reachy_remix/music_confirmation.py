"""Pure, app-owned music request snapshots; never a consent word matcher.

Callers alone establish genuine user-turn provenance and successful presentation.
This module performs no provider calls, file access, logging, or tool execution.
"""

from __future__ import annotations
import json
import math
import time
import threading
from uuid import uuid4
from typing import Any
from dataclasses import replace, dataclass
from collections.abc import Callable

from dj_reachy_remix.runtime_settings import validate_generation_request


def _text(value: Any, limit: int, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid confirmation text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise ValueError("invalid confirmation UTF8") from None
    if size > limit or (nonempty and not value.strip()):
        raise ValueError("invalid confirmation text size")
    return value


def _counter(value: Any, *, positive: bool = False) -> bool:
    return type(value) is int and (1 if positive else 0) <= value < 2**63


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, repr=False)
class MusicDraft:
    """Immutable local snapshot, with no provider delegation identity."""

    id: str
    revision: int
    original_turn: int
    original_user_text: str
    arguments_json: str
    source_arguments_json: str
    music_settings_json: str
    preview: str
    created_at: float

    @property
    def arguments(self) -> dict[str, Any]:
        """Return a fresh deep copy of the concrete execution options."""
        result: dict[str, Any] = json.loads(self.arguments_json)
        return result

    def pending_payload(self, assistant_context: list[str] | None = None) -> dict[str, Any]:
        """Export bounded RPC data; assistant narration is context, never consent."""
        if assistant_context is None:
            assistant_context = []
        if not isinstance(assistant_context, list) or len(assistant_context) > 4:
            raise ValueError("invalid confirmation context")
        for item in assistant_context:
            _text(item, 8192)
        if sum(len(item.encode("utf-8")) for item in assistant_context) > 8192:
            raise ValueError("invalid confirmation context size")
        _text(self.arguments_json, 32768)
        _text(self.preview, 8192, nonempty=True)
        _text(self.original_user_text, 12000, nonempty=True)
        if not self.id or len(self.id) > 200 or not self.id.isascii() or any(c.isspace() for c in self.id):
            raise ValueError("invalid confirmation draft id")
        payload = {
            "draft_id": self.id,
            "request": self.arguments,
            "preview": self.preview,
            "original_user_text": self.original_user_text,
            "assistant_context": list(assistant_context),
        }
        _text(_json(payload), 65536)
        return payload


@dataclass(frozen=True, repr=False)
class ConfirmationTicket:
    """An app-issued single-use snapshot for one genuine reply assessment."""

    draft_id: str
    draft_revision: int
    epoch: int
    reply_turn: int
    reply_text: str
    state_revision: int


@dataclass(frozen=True, repr=False)
class ConfirmationResolution:
    """A non-executing result; only approval carries execution arguments."""

    kind: str
    arguments_json: str | None = None
    draft: MusicDraft | None = None
    text: str | None = None

    @property
    def arguments(self) -> dict[str, Any] | None:
        """Return detached options only for an approved resolution."""
        if self.arguments_json is None:
            return None
        result: dict[str, Any] = json.loads(self.arguments_json)
        return result


def _snapshot(arguments: dict[str, Any], settings: dict[str, Any]) -> tuple[str, str, str, str]:
    try:
        source = _text(_json(arguments), 32768)
        frozen_settings = json.loads(_json(settings))
        music_settings = _text(_json(frozen_settings["music"]), 32768)
        frozen = _text(_json(validate_generation_request(json.loads(source), frozen_settings)), 32768)
        preview = _text("Music request (requires a fresh user reply): " + frozen, 8192)
        return source, music_settings, frozen, preview
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise ValueError("invalid confirmation music request") from None


class PendingMusicConfirmation:
    """Hold local request state, without granting semantic authority to text.

    Prepare only from a genuine caller delegation and the actual original user
    text. Offer only after preview delivery succeeds. The caller must advance
    ``current_turn`` on the FIRST input fragment, not just final transcripts,
    and clear synchronously on stop, disconnect, cancellation, or replacement.
    No pure string API can establish audio/user provenance on the caller's behalf.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Use an injectable monotonic clock, without reading user configuration."""
        self._clock = clock
        self._lock = threading.RLock()
        self._prepared: MusicDraft | None = None
        self._pending: MusicDraft | None = None
        self._ticket: ConfirmationTicket | None = None
        self._epoch: int | None = None
        self._presented_turn = 0
        self._state_revision = 0
        self._last_turn = 0
        self._reply_count = 0

    @property
    def pending(self) -> MusicDraft | None:
        """Return the currently offered draft, if any."""
        with self._lock:
            self._prune()
            return self._pending

    def clear(self) -> None:
        """Invalidate active, prepared, and in-flight authority synchronously."""
        with self._lock:
            self._pending = None
            self._prepared = None
            self._ticket = None
            self._state_revision += 1
            self._epoch = None
            self._last_turn = 0
            self._reply_count = 0
            self._presented_turn = 0

    def _prune(self) -> None:
        draft = self._pending or self._prepared
        if draft is not None and (
            not 0 <= self._clock() - draft.created_at < 90 or (self._reply_count >= 4 and self._ticket is None)
        ):
            self.clear()

    def offer(self, draft: MusicDraft, *, epoch: int, presented_turn: int) -> None:
        """Activate only after the caller successfully sent the full preview."""
        with self._lock:
            self._prune()
            if (
                self._prepared is not draft
                or not _counter(epoch)
                or not _counter(presented_turn, positive=True)
                or presented_turn < max(draft.original_turn, self._last_turn)
                or (self._epoch is not None and epoch != self._epoch)
            ):
                self.clear()
                raise ValueError("invalid confirmation presentation")
            self._pending = draft
            self._prepared = None
            self._epoch = epoch
            self._presented_turn = presented_turn
            self._last_turn = presented_turn
            self._ticket = None
            self._state_revision += 1

    def begin_reply(self, *, turn: int, text: str, epoch: int) -> ConfirmationTicket | None:
        """Issue a snapshot for semantic assessment of the caller's actual reply."""
        with self._lock:
            self._prune()
            draft = self._pending
            if draft is None:
                if (
                    self._prepared is not None
                    and _counter(turn, positive=True)
                    and turn > max(self._last_turn, self._prepared.original_turn)
                ):
                    self.clear()
                return None
            self._ticket = None
            self._state_revision += 1
            if self._reply_count >= 4:
                self.clear()
                return None
            if not _counter(epoch) or epoch != self._epoch:
                self.clear()
                return None
            if not _counter(turn, positive=True) or turn <= max(
                self._presented_turn, draft.original_turn, self._last_turn
            ):
                return None
            self._last_turn = turn
            try:
                _text(text, 12000, nonempty=True)
            except ValueError:
                return None
            self._reply_count += 1
            self._ticket = ConfirmationTicket(draft.id, draft.revision, epoch, turn, text, self._state_revision)
            return self._ticket

    def apply(
        self,
        ticket: ConfirmationTicket,
        decision: dict[str, Any],
        *,
        epoch: int,
        current_turn: int,
        settings: dict[str, Any],
    ) -> ConfirmationResolution:
        """Consume an eligible approval synchronously before returning options."""
        with self._lock:
            self._prune()
            draft = self._pending
            if not isinstance(ticket, ConfirmationTicket) or self._ticket is not ticket or draft is None:
                return ConfirmationResolution("stale")
            self._ticket = None
            self._state_revision += 1
            if (
                not _counter(epoch)
                or epoch != self._epoch
                or epoch != ticket.epoch
                or not _counter(current_turn, positive=True)
                or current_turn != ticket.reply_turn
                or current_turn != self._last_turn
                or ticket.state_revision != self._state_revision - 1
                or ticket.draft_id != draft.id
                or ticket.draft_revision != draft.revision
            ):
                self.clear()
                return ConfirmationResolution("stale")
            if not isinstance(decision, dict) or set(decision) != {"decision", "arguments", "text"}:
                return ConfirmationResolution("stale")
            choice = decision["decision"]
            try:
                reply = _text(decision["text"], 8000)
                if len(reply) > 2000 or choice not in ("approve", "revise", "decline", "none"):
                    return ConfirmationResolution("stale")
            except ValueError:
                return ConfirmationResolution("stale")
            source_arguments = json.loads(draft.source_arguments_json)
            if choice == "revise":
                changes = decision["arguments"]
                if not isinstance(changes, dict) or set(changes) != {"prompt", "music_length_ms"}:
                    return ConfirmationResolution("stale")
                try:
                    prompt = _text(changes["prompt"], 8000, nonempty=True)
                    if len(prompt) > 2000:
                        return ConfirmationResolution("stale")
                except ValueError:
                    return ConfirmationResolution("stale")
                source_arguments.update(changes)
                if "prompt" in changes:
                    source_arguments.pop("composition_plan", None)
            elif decision["arguments"] is not None:
                return ConfirmationResolution("stale")
            if choice == "decline":
                self.clear()
                return ConfirmationResolution("declined", text=reply)
            try:
                source, music_settings, frozen, preview = _snapshot(source_arguments, settings)
                if not 0 <= self._clock() - draft.created_at < 90:
                    self.clear()
                    return ConfirmationResolution("stale")
                if choice == "revise" or music_settings != draft.music_settings_json:
                    revised = replace(
                        draft,
                        revision=draft.revision + 1,
                        arguments_json=frozen,
                        source_arguments_json=source,
                        music_settings_json=music_settings,
                        preview=preview,
                    )
                    revised.pending_payload()
                    self._pending = None
                    self._prepared = revised
                    return ConfirmationResolution("revised", draft=revised, text=reply)
            except ValueError:
                self.clear()
                return ConfirmationResolution("stale")
            if choice == "none":
                return ConfirmationResolution("none", text=reply)
            self.clear()
            return ConfirmationResolution("approved", arguments_json=draft.arguments_json, text=reply)

    def prepare(
        self, arguments: dict[str, Any], settings: dict[str, Any], *, original_turn: int, original_user_text: str
    ) -> MusicDraft:
        """Freeze validated execution options without activating a request."""
        with self._lock:
            self.clear()
            created_at = self._clock()
            if not math.isfinite(created_at):
                raise ValueError("invalid confirmation lifetime origin")
            if not _counter(original_turn, positive=True):
                raise ValueError("invalid confirmation original turn")
            _text(original_user_text, 12000, nonempty=True)
            source, music_settings, frozen, preview = _snapshot(arguments, settings)
            draft = MusicDraft(
                id=uuid4().hex,
                revision=0,
                original_turn=original_turn,
                original_user_text=original_user_text,
                arguments_json=frozen,
                source_arguments_json=source,
                music_settings_json=music_settings,
                preview=preview,
                created_at=created_at,
            )
            draft.pending_payload()
            self._prepared = draft
            return draft
