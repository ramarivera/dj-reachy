"""App voice handler with independent, correlated and cancellable tool ownership."""

from __future__ import annotations
import json
import base64
import asyncio
import logging
from time import monotonic
from typing import Any, Final, Literal, TypeVar, Protocol, cast
from hashlib import sha256
from datetime import datetime
from collections.abc import Callable, Awaitable

import numpy as np
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from dj_reachy_remix.config import config, runtime_settings_store
from dj_reachy_remix.prompts import get_session_voice, get_session_instructions
from dj_reachy_remix.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call
from dj_reachy_remix.audio_diagnostics import Phase, AudioDiagnostics
from dj_reachy_remix.music_confirmation import MusicDraft, PendingMusicConfirmation
from dj_reachy_remix.subscription_voice import (
    SubscriptionRealtimeAdapter,
    SubscriptionRealtimeConnection,
    fast_control,
    subscription_delegation,
)


logger = logging.getLogger(__name__)
OPEN_AI_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OPEN_AI_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
SONG_TOOLS = {"make_song_and_dance", "play_saved_song"}
CONTROL_TOOLS = {"pause_music", "resume_music", "stop_music"}


_WorkResult = TypeVar("_WorkResult")


class _WorkTask(asyncio.Task[_WorkResult]):
    """Remember cancellation even when a transport suppresses CancelledError.

    Task.cancelling exists only on Python 3.11+. Keep the same authority fence
    on every supported Python version, without touching the loop's task factory.
    """

    cancel_requested = False

    def cancel(self, msg: Any = None) -> bool:
        """Fence this task before requesting cancellation."""
        self.cancel_requested = True
        return super().cancel(msg)

    def uncancel(self) -> int:
        """Release internal Python 3.11+ timeout cancellation like the base task."""
        # Older wait_for cancels its child instead and never calls this method.
        remaining = cast(Callable[[], int], getattr(super(), "uncancel"))()
        self.cancel_requested = remaining > 0
        return remaining


class VoiceConfiguration(Protocol):
    """Typed view of the fields config initializes from persisted runtime settings."""

    VOICE_BACKEND: str
    VOICE_MODEL: str
    VOICE_NAME: str
    CONVERSATION_LANGUAGE: str


voice_config = cast(VoiceConfiguration, config)


class VoiceTransportError(ConnectionError):
    """A sanitized input-transport failure, distinct from local PCM/media errors."""


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """Keep the voice receiver live while independently owned app tasks execute."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: str | None = None):
        """Initialize lifecycle state without opening providers or hardware."""
        super().__init__(expected_layout="mono", output_sample_rate=24000, input_sample_rate=24000)
        self.deps = deps
        self.diagnostics = AudioDiagnostics()
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self.connection: Any = None
        self.client: Any = None
        self.output_queue: asyncio.Queue[tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        self._drain_queue: Callable[[], Awaitable[bool]] | None = None
        self.last_activity_time = monotonic()
        self.start_time = self.last_activity_time
        self.is_idle_tool_call = False
        self._shutdown_requested = False
        self._connected_event = asyncio.Event()
        self._mute_output_audio = False
        self._audio_mute_owner: str | None = None
        self._response_done_event = asyncio.Event()
        self._waiting_for_response = False
        self._music_ready_notice: asyncio.Event | None = None
        self._tool_tasks: dict[str, asyncio.Task[None]] = {}
        self._subscription_tasks: dict[str, asyncio.Task[None]] = {}
        self._work_tasks: set[asyncio.Task[Any]] = set()
        self._seen_calls: set[str] = set()
        self._seen_delegations: set[str] = set()
        self._delegated_turns: set[int] = set()
        self._turn_controls: dict[int | str, asyncio.Task[dict[str, Any]]] = {}
        self._control_tasks: set[asyncio.Task[dict[str, Any]]] = set()
        self._song_task: asyncio.Task[Any] | None = None
        self._music_call_active = False
        self._work_epoch = 0
        self._stop_barrier: asyncio.Task[dict[str, Any]] | None = None
        self._supervisor: asyncio.Task[Any] | None = None
        self._restart_requested = False
        self._pending_music = PendingMusicConfirmation()
        self._latest_user_turn = 0
        self._final_user_turns: dict[int, str] = {}
        self._assistant_context: list[str] = []
        self._reply_tasks: dict[int, asyncio.Task[str]] = {}
        self._reply_identities: dict[int, str] = {}
        self._final_events: dict[int, asyncio.Event] = {}
        self._fragment_turn = 0
        self._fragment_text: str | None = ""
        self._seen_final_payloads: set[bytes] = set()
        self._seen_user_finals: set[str] = set()
        self._preview_draft: MusicDraft | None = None

    def copy(self) -> OpenaiRealtimeHandler:
        """Create a fresh connection owner sharing only injected app dependencies."""
        return OpenaiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    def mute_output_audio(self, owner: str = "external") -> None:
        """Fence prior PCM and filter history once per independent playback owner."""
        if self._audio_mute_owner != owner:
            clear = getattr(self, "_clear_queue", None)
            if callable(clear):
                clear()
        self._audio_mute_owner = owner
        self._mute_output_audio = True

    def unmute_output_audio(self, owner: str = "external") -> None:
        """Never let an old playback task unmute its replacement."""
        if self._audio_mute_owner == owner:
            self._audio_mute_owner = None
            self._mute_output_audio = False

    def _current(self, connection: Any, epoch: int) -> bool:
        owner = asyncio.current_task()
        return (
            self.connection is connection
            and not self._shutdown_requested
            and getattr(connection, "_closed", False) is not True
            and self._work_epoch == epoch
            and (owner is None or (not owner.cancelled() and not (isinstance(owner, _WorkTask) and owner.cancel_requested)))
        )

    async def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.diagnostics.lifecycle("tool", "started", name=name)
        try:
            result = await dispatch_tool_call(name, json.dumps(args), self.deps)
        except asyncio.CancelledError:
            self.diagnostics.lifecycle("tool", "cancelled", name=name)
            raise
        except Exception as exc:
            self.diagnostics.lifecycle("tool", "failed", name=name)
            # Provider exceptions can echo credentials. Only the class crosses the voice boundary.
            return {"error": type(exc).__name__, "message": "App command failed; no automatic retry was made."}
        self.diagnostics.lifecycle("tool", "failed" if "error" in result else "completed", name=name)
        return result

    def _stop_work(self) -> asyncio.Task[dict[str, Any]]:
        """Invalidate planner/generation work synchronously before scheduling local stop."""
        self._pending_music.clear()
        self._preview_draft = None
        self._work_epoch += 1
        current = asyncio.current_task()
        obsolete = [task for task in self._work_tasks if task is not current and not task.done()]
        for task in obsolete:
            task.cancel()
        previous = self._stop_barrier

        async def stop() -> dict[str, Any]:
            if previous is not None and not previous.done():
                await asyncio.shield(previous)
            result = await self._dispatch("stop_music", {})
            await asyncio.gather(*obsolete, return_exceptions=True)
            return {
                **result,
                "local_cancellation": True,
                "provider_cancellation": "unconfirmed; a paid provider job may have continued",
            }

        stop_task = asyncio.create_task(stop(), name="voice-stop-music")
        self._stop_barrier = stop_task
        self._control_tasks.add(stop_task)
        stop_task.add_done_callback(self._control_tasks.discard)
        return stop_task

    async def _execute(self, name: str, args: dict[str, Any], connection: Any, epoch: int) -> dict[str, Any]:
        if not self._current(connection, epoch):
            raise asyncio.CancelledError
        if name == "stop_music":
            return await asyncio.shield(self._stop_work())
        if name in SONG_TOOLS:
            if self._stop_barrier is not None:
                await asyncio.shield(self._stop_barrier)
            if not self._current(connection, epoch):
                raise asyncio.CancelledError
            if self._song_task is not None and not self._song_task.done():
                return {"status": "busy", "message": "A song is already generating or playing."}
            owner = asyncio.current_task()
            self._song_task = owner
            self._music_call_active = True
            try:
                return await self._dispatch(name, args)
            finally:
                if self._song_task is owner:
                    self._song_task = None
                    self._music_call_active = False
        return await self._dispatch(name, args)

    async def notify_song_ready(self) -> bool:
        """Announce actual generation completion before playback acquires its mute.

        Only the background song task waits here. The live receiver continues
        accepting conversation, interruption and stop commands throughout.
        Timeout, unavailable drainage, or replaced ownership refuses playback.
        """
        connection, epoch = self.connection, self._work_epoch
        if connection is None or not self._current(connection, epoch):
            return False
        notice = asyncio.Event()
        self._music_ready_notice = notice
        text = json.dumps({
            "tool": "make_song_and_dance", "status": "generated", "playback_status": "not_started",
            "message": "Briefly announce that the song is ready and playback will start next. "
            "Do not ask for feedback yet; the user has not heard the song. "
            "Playback completion will be reported separately.",
        })
        try:
            if voice_config.VOICE_BACKEND == "subscription":
                await connection.send_context(text, "speakable", None)
            else:
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
                await connection.response.create(response={"instructions": text})
            try:
                await asyncio.wait_for(notice.wait(), 8)
            except asyncio.TimeoutError:
                logger.warning("Song-ready speech completion was not confirmed within its bounded wait")
                return False
            if self._current(connection, epoch) and self._drain_queue is not None:
                drained = await self._drain_queue()
                return drained is True and self._current(connection, epoch)
        finally:
            if self._music_ready_notice is notice:
                self._music_ready_notice = None
        return False

    def schedule_tool_call(self, tool_name: str, args_json: str, call_id: str) -> asyncio.Task[None]:
        """Run API Realtime function calls independently and dedupe completed IDs."""
        existing = self._tool_tasks.get(call_id)
        if existing is not None:
            return existing
        duplicate = call_id in self._seen_calls
        self._seen_calls.add(call_id)
        stop = self._stop_work() if tool_name == "stop_music" and not duplicate else None
        task = asyncio.create_task(
            self._run_tool_call(tool_name, args_json, call_id, duplicate, stop), name=f"tool-{call_id}"
        )
        self._tool_tasks[call_id] = task
        if tool_name not in CONTROL_TOOLS:
            self._work_tasks.add(task)
        task.add_done_callback(lambda done: self._tool_tasks.pop(call_id, None))
        task.add_done_callback(self._work_tasks.discard)
        return task

    async def _run_tool_call(
        self,
        tool_name: str,
        args_json: str,
        call_id: str,
        duplicate: bool = False,
        stop: asyncio.Task[dict[str, Any]] | None = None,
    ) -> None:
        connection, epoch = self.connection, self._work_epoch
        if duplicate or connection is None:
            return
        try:
            args = json.loads(args_json)
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            result = {"error": "Tool arguments must be a JSON object"}
        else:
            result = (
                await asyncio.shield(stop)
                if stop is not None
                else await self._execute(tool_name, args, connection, epoch)
            )
        if not self._current(connection, epoch):
            return
        await connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            }
        )
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": json.dumps(result),
                    "metadata": {"title": f"Used tool {tool_name}", "status": "done"},
                }
            )
        )
        if not self.is_idle_tool_call:
            await connection.response.create(
                response={"instructions": "Use the correlated tool result and answer concisely."}
            )
        self.is_idle_tool_call = False
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

    def _control_for_turn(self, turn: int | str, name: str) -> asyncio.Task[dict[str, Any]]:
        existing = self._turn_controls.get(turn)
        if existing is not None:
            return existing
        task = (
            self._stop_work()
            if name == "stop_music"
            else asyncio.create_task(self._dispatch(name, {}), name=f"control-{turn}")
        )
        self._turn_controls[turn] = task
        self._control_tasks.add(task)
        task.add_done_callback(self._control_tasks.discard)
        return task

    def schedule_subscription_delegation(self, event: Any) -> None:
        """Schedule current user speech or an explicitly labeled Live model request."""
        if self.connection is None or self._shutdown_requested:
            return
        decoded = subscription_delegation(getattr(event, "item", None))
        if decoded is None:
            return
        identity, request = decoded
        if identity in self._seen_delegations:
            return
        value = getattr(event, "transcript_context", "")
        transcript = value if isinstance(value, str) else ""
        turn = getattr(event, "turn_id", None)
        context_turn_id = turn if type(turn) is int and turn > 0 else None
        if not transcript.strip() and not request:
            # An empty creation notice may be followed by the populated same-ID item.
            return
        self._seen_delegations.add(identity)
        connection, epoch = self.connection, self._work_epoch
        pending_reply = self._pending_music.pending is not None or turn in self._reply_tasks
        control = fast_control(transcript or request)
        correlation = context_turn_id if transcript and context_turn_id is not None else f"delegation:{identity}"
        control_task = self._control_for_turn(correlation, control) if control else None
        epoch = self._work_epoch
        if pending_reply and control is None and context_turn_id is not None:
            self._reply_identities.setdefault(context_turn_id, identity)
        task_context = request if transcript and control is None else ""
        prior_value = getattr(event, "prior_user_context", ())
        prior_user_context: tuple[str, ...] = ()
        if (
            transcript
            and control is None
            and isinstance(prior_value, tuple)
            and len(prior_value) <= 4
            and all(isinstance(item, str) and item for item in prior_value)
            and sum(len(item.encode("utf-8")) for item in prior_value) <= 8192
        ):
            prior_user_context = prior_value
        awaiting_final = bool(getattr(connection, "_awaiting_final", False)) and context_turn_id is not None
        final_event = (
            self._final_events.setdefault(context_turn_id, asyncio.Event())
            if awaiting_final and context_turn_id is not None
            else None
        )
        allow_music_prepare = bool(transcript.strip()) and context_turn_id not in self._delegated_turns
        if not pending_reply and transcript and context_turn_id is not None:
            self._delegated_turns.add(context_turn_id)

        async def run() -> None:
            actual = transcript
            if final_event is not None and control is None:
                try:
                    await asyncio.wait_for(final_event.wait(), timeout=10)
                except asyncio.TimeoutError:
                    if self._current(connection, epoch):
                        self._pending_music.clear()
                        await connection.send_context(
                            "The complete user request was not received; no music was generated.",
                            "speakable",
                            identity,
                        )
                    return
                actual = self._final_user_turns.get(context_turn_id or 0, "")
            if not self._current(connection, epoch):
                return
            if pending_reply and control is None:
                reply = self._reply_tasks.get(context_turn_id or 0)
                if reply is None and context_turn_id is not None:
                    reply = self._schedule_music_reply(context_turn_id, actual)
                kind = await asyncio.shield(reply) if reply is not None else "stale"
                if not self._current(connection, epoch):
                    return
                if kind != "none" or turn in self._delegated_turns:
                    await connection.send_context(
                        "This reply has already been handled, or requires a fresh user reply to the offered preview. "
                        "Do not generate or repeat a music command.",
                        "commentary",
                        identity,
                    )
                    return
                if context_turn_id is not None:
                    self._delegated_turns.add(context_turn_id)
            if not actual:
                actual = "Live model delegation request (not a verbatim user transcript):\n" + request
            await self._run_subscription_delegation(
                identity,
                actual,
                control_task,
                task_context=task_context,
                prior_user_context=prior_user_context,
                context_turn_id=context_turn_id,
                allow_music_prepare=allow_music_prepare and not pending_reply,
            )

        task = _WorkTask(run(), name=f"delegation-{identity}")
        self._subscription_tasks[identity] = task
        if control is None:
            self._work_tasks.add(task)
        task.add_done_callback(lambda done: self._subscription_tasks.pop(identity, None))
        task.add_done_callback(self._work_tasks.discard)

    async def _run_subscription_delegation(
        self,
        identity: str,
        transcript: str,
        control_task: asyncio.Task[dict[str, Any]] | None = None,
        *,
        task_context: str = "",
        prior_user_context: tuple[str, ...] = (),
        context_turn_id: int | None = None,
        allow_music_prepare: bool = True,
    ) -> None:
        connection, epoch = self.connection, self._work_epoch
        if connection is None:
            return
        self.diagnostics.lifecycle("delegation", "started")
        outcome: Phase = "completed"
        try:
            if not transcript:
                await connection.send_context(
                    "No new user request. Do not repeat any previous command; ask what the user wants.",
                    "commentary",
                    identity,
                )
                return
            if control_task is not None:
                result = await asyncio.shield(control_task)
                if self._current(connection, epoch):
                    await connection.send_context(json.dumps(result), "speakable", identity)
                return
            await connection.send_context(
                "The app is working on the current request. No result yet.", "commentary", identity
            )
            from dj_reachy_remix.voice_planner import DELEGATION_CONTEXT_PREFIX, PENDING_USER_CONTEXT_PREFIX

            history: list[dict[str, Any]] = []
            if prior_user_context:
                history.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": PENDING_USER_CONTEXT_PREFIX + json.dumps(prior_user_context, ensure_ascii=False),
                    }
                )
            if task_context:
                history.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": DELEGATION_CONTEXT_PREFIX + task_context,
                    }
                )
            seen_calls: set[str] = set()
            from dj_reachy_remix.voice_planner import MAX_CALLS, MAX_ROUNDS, validate_app_call

            for _round in range(MAX_ROUNDS):
                self.diagnostics.lifecycle("planner", "started")
                try:
                    step = await connection.plan(transcript, history)
                except asyncio.CancelledError:
                    self.diagnostics.lifecycle("planner", "cancelled")
                    raise
                except Exception:
                    self.diagnostics.lifecycle("planner", "failed")
                    raise
                self.diagnostics.lifecycle("planner", "completed")
                if not self._current(connection, epoch):
                    return
                calls = step.get("calls")
                output = step.get("output")
                if not isinstance(calls, list) or not isinstance(output, list):
                    raise ValueError("Invalid app planner response")
                if not calls:
                    text = step.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("Planner returned no result or clarification")
                    await connection.send_context(text, "speakable", identity)
                    return
                history.extend(output)
                for call in calls:
                    name, args = validate_app_call(call.get("name"), call.get("arguments"))
                    call_id = call.get("call_id")
                    if not isinstance(call_id, str) or call_id in seen_calls or len(seen_calls) >= MAX_CALLS:
                        raise ValueError("Duplicate or excessive app planner calls")
                    seen_calls.add(call_id)
                    clear_context = getattr(connection, "clear_pending_user_context", None)
                    if context_turn_id is not None and callable(clear_context):
                        clear_context(through_turn=context_turn_id)
                    if name == "make_song_and_dance":
                        original = self._final_user_turns.get(context_turn_id or 0, "")
                        if (
                            not allow_music_prepare
                            or not original
                            or context_turn_id is None
                            or context_turn_id != self._latest_user_turn
                        ):
                            raise ValueError(
                                "Music requires a complete current user request and a fresh preview reply"
                            )
                        draft = self._pending_music.prepare(
                            args,
                            runtime_settings_store.read(),
                            original_turn=context_turn_id,
                            original_user_text=original,
                        )
                        await self._offer_music(draft, connection, epoch, identity)
                        return
                    result = await self._execute(name, args, connection, epoch)
                    if name == "stop_music":
                        epoch = self._work_epoch
                        if self._current(connection, epoch):
                            await connection.send_context(json.dumps(result), "speakable", identity)
                        return
                    if not self._current(connection, epoch):
                        return
                    history.append({"type": "function_call_output", "call_id": call_id, "output": json.dumps(result)})
                    # Only the host vision planner receives image bytes. Do not
                    # chunk base64 through the Live model's text context channel.
                    progress = {"status": "image_captured"} if name == "camera" and "b64_im" in result else result
                    await connection.send_context(
                        json.dumps({"tool": name, "result": progress}), "commentary", identity
                    )
            raise RuntimeError("App planner reached its bounded round limit")
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            outcome = "failed"
            if self._current(connection, epoch):
                # Broker/validation errors are safe application messages; other exceptions are class-only.
                detail = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
                await connection.send_context(
                    f"App task failed: {detail}. No automatic retry or API fallback was used.", "speakable", identity
                )
        finally:
            self.diagnostics.lifecycle("delegation", outcome)

    async def _offer_music(self, draft: MusicDraft, connection: Any, epoch: int, identity: str | None) -> None:
        """Activate a frozen request only after complete, correlated preview delivery."""
        presented_turn = self._latest_user_turn
        self._preview_draft = draft
        try:
            if isinstance(connection, SubscriptionRealtimeConnection) and not connection.ready.is_set():
                raise ConnectionError("Music preview requires a ready subscription transport")
            await connection.send_context(draft.preview, "speakable", identity)
            if self._preview_draft is not draft:
                return
            if not self._current(connection, epoch) or self._latest_user_turn != presented_turn:
                self._pending_music.clear()
                return
            self._pending_music.offer(draft, epoch=epoch, presented_turn=presented_turn)
        except BaseException:
            if self._preview_draft is draft:
                self._pending_music.clear()
            raise
        finally:
            if self._preview_draft is draft:
                self._preview_draft = None

    def _schedule_music_reply(self, turn: int, text: str) -> asyncio.Task[str] | None:
        """Assess only one completed actual reply to an offered draft per turn."""
        if turn in self._reply_tasks:
            return self._reply_tasks[turn]
        draft = self._pending_music.pending
        if draft is None:
            return None
        ticket = self._pending_music.begin_reply(turn=turn, text=text, epoch=self._work_epoch)
        if ticket is None:
            return None
        connection, epoch = self.connection, self._work_epoch
        pending = draft.pending_payload(self._assistant_context)

        async def run() -> str:
            execution_attempted = False
            try:
                self.diagnostics.lifecycle("planner", "started")
                try:
                    result = await connection.plan_confirmation(text, pending)
                except asyncio.CancelledError:
                    self.diagnostics.lifecycle("planner", "cancelled")
                    raise
                except Exception:
                    self.diagnostics.lifecycle("planner", "failed")
                    raise
                self.diagnostics.lifecycle("planner", "completed")
                if not self._current(connection, epoch):
                    if (
                        self.connection is connection
                        and self._latest_user_turn == turn
                        and self._pending_music.pending is draft
                    ):
                        self._pending_music.clear()
                    return "stale"
                metadata = result.get("confirmation")
                if not isinstance(metadata, dict):
                    raise ValueError("Invalid confirmation metadata")
                # Do not consume consent ahead of an await which could admit
                # newer speech or settings before the actual generation dispatch.
                if self._stop_barrier is not None:
                    await asyncio.shield(self._stop_barrier)
                if not self._current(connection, epoch):
                    return "stale"
                resolution = self._pending_music.apply(
                    ticket,
                    {**metadata, "text": result.get("text")},
                    epoch=epoch,
                    current_turn=self._latest_user_turn,
                    settings=runtime_settings_store.read(),
                )
                identity = self._reply_identities.get(turn)
                if resolution.kind == "approved" and resolution.arguments is not None:
                    execution_attempted = True
                    outcome = await self._execute("make_song_and_dance", resolution.arguments, connection, epoch)
                    if self._current(connection, epoch):
                        await connection.send_context(json.dumps(outcome), "speakable", identity)
                elif resolution.kind == "revised" and resolution.draft is not None:
                    await self._offer_music(resolution.draft, connection, epoch, identity)
                elif resolution.kind in ("none", "declined") and resolution.text:
                    await connection.send_context(resolution.text, "speakable", identity)
                elif (
                    resolution.kind == "stale"
                    and self._latest_user_turn > turn
                    and self._pending_music.pending is None
                    and self._song_task is None
                    and self._preview_draft is None
                    and self._latest_user_turn not in self._reply_tasks
                    and self._latest_user_turn not in self._delegated_turns
                ):
                    await connection.send_context(
                        "No music was submitted for that earlier reply; newer speech superseded it.",
                        "speakable",
                        None,
                    )
                return resolution.kind
            except asyncio.CancelledError:
                if self._latest_user_turn == turn and self._pending_music.pending is draft:
                    self._pending_music.clear()
                raise
            except Exception:
                if self._latest_user_turn == turn and self._pending_music.pending is draft:
                    self._pending_music.clear()
                if self._current(connection, epoch) and self._latest_user_turn == turn:
                    failure_text = (
                        "Music result delivery failed; submission status is unconfirmed. No automatic retry was made."
                        if execution_attempted
                        else "Music confirmation failed; no music was submitted and no automatic retry was made."
                    )
                    await connection.send_context(
                        failure_text,
                        "speakable",
                        self._reply_identities.get(turn),
                    )
                return "stale"

        task = _WorkTask(run(), name=f"music-reply-{turn}")
        self._reply_tasks[turn] = task
        self._work_tasks.add(task)
        task.add_done_callback(self._work_tasks.discard)
        return task

    def _create_client(self, openai_key: str | None) -> None:
        if voice_config.VOICE_BACKEND == "subscription":
            self.client = SubscriptionRealtimeAdapter(
                model=voice_config.VOICE_MODEL,
                voice=voice_config.VOICE_NAME,
                instructions=get_session_instructions(),
                diagnostics=self.diagnostics,
            )
        elif voice_config.VOICE_BACKEND == "api_key":
            if not openai_key or not openai_key.strip():
                raise RuntimeError(
                    "API-key Realtime is selected, but OPENAI_API_KEY is missing; no subscription fallback is used."
                )
            if not voice_config.VOICE_MODEL.startswith(("gpt-realtime", "gpt-4o-realtime")):
                raise RuntimeError(
                    "API-key Realtime requires an explicit supported Realtime voice.model, such as gpt-realtime."
                )
            self.client = AsyncOpenAI(api_key=openai_key.strip())
        else:
            raise RuntimeError("Unsupported voice backend; no billing fallback is allowed.")

    async def start_up(self) -> None:
        """Supervise sessions. Subscription reconnect is explicit, never an automatic new billed call."""
        from dj_reachy_remix.config import set_api_keys

        openai_key, eleven_key = config.OPENAI_API_KEY, config.ELEVENLABS_API_KEY
        if self.gradio_mode:
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            if len(args) > 3 and isinstance(args[3], str) and args[3].strip():
                openai_key = args[3]
            if len(args) > 4 and isinstance(args[4], str) and args[4].strip():
                eleven_key = args[4]
            set_api_keys(openai_key, eleven_key)
        self.deps.elevenlabs_api_key = eleven_key.strip() if eleven_key else None
        self._shutdown_requested = False
        if self.diagnostics.closed:
            self.diagnostics = AudioDiagnostics(home=self.diagnostics.home)
        self.diagnostics.start()
        self._supervisor = asyncio.current_task()
        attempt = 0
        backend = voice_config.VOICE_BACKEND
        try:
            while not self._shutdown_requested:
                if voice_config.VOICE_BACKEND != backend:
                    raise RuntimeError("Voice backend changed; explicit reconnect required")
                self._restart_requested = False
                self._create_client(openai_key)
                try:
                    await self._run_realtime_session()
                    if not self._restart_requested:
                        return
                    attempt = 0
                except (ConnectionClosedError, ConnectionError):
                    attempt += 1
                    logger.warning("Realtime connection closed unexpectedly: voice_transport (attempt %d)", attempt)
                    if self._shutdown_requested:
                        return
                    if backend == "subscription" or voice_config.VOICE_BACKEND != backend or attempt >= 3:
                        raise
                    await asyncio.sleep(2 ** (attempt - 1))
                finally:
                    await self._end_session_work()
                    self.connection = None
                    self._connected_event.clear()
                    if isinstance(self.client, AsyncOpenAI):
                        await self.client.close()
        finally:
            self._supervisor = None
            self.diagnostics.close()

    async def _run_realtime_session(self) -> None:
        async with self.client.realtime.connect(model=voice_config.VOICE_MODEL) as conn:
            self.connection = conn
            self._seen_calls.clear()
            self._seen_delegations.clear()
            self._delegated_turns.clear()
            self._turn_controls.clear()
            if voice_config.VOICE_BACKEND == "api_key":
                session: dict[str, Any] = {
                    "type": "realtime",
                    "instructions": get_session_instructions(),
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "turn_detection": {"type": "server_vad", "interrupt_response": True},
                            "transcription": {"model": "gpt-4o-transcribe"},
                        },
                        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": get_session_voice()},
                    },
                    "tools": get_tool_specs(),
                    "tool_choice": "auto",
                }
                if voice_config.CONVERSATION_LANGUAGE != "auto":
                    session["audio"]["input"]["transcription"]["language"] = voice_config.CONVERSATION_LANGUAGE
                await conn.session.update(session=session)
            # Subscription configuration was sent with SDP; ordinary session.update is not its dialect.
            self._connected_event.set()
            async for event in conn:
                await self._handle_event(event)

    async def _handle_event(self, event: Any) -> None:
        kind = event.type
        self.diagnostics.provider_event(kind, origin="handled")
        if kind in ("input_transcript.added", "output_transcript.added"):
            item = getattr(event, "item", {})
            text = item.get("text", "") if isinstance(item, dict) else ""
            if kind == "input_transcript.added":
                turn_id = getattr(event, "turn_id", 0)
                if type(turn_id) is int:
                    self._latest_user_turn = max(self._latest_user_turn, turn_id)
                    if turn_id != self._fragment_turn:
                        self._fragment_text = ""
                    self._fragment_turn = turn_id
                    if isinstance(text, str) and self._fragment_text is not None:
                        self._fragment_text += text
                        try:
                            if len(self._fragment_text.encode("utf-8")) > 12000:
                                self._fragment_text = None
                        except UnicodeError:
                            self._fragment_text = None
            if text:
                self.last_activity_time = monotonic()
                await self.output_queue.put(
                    AdditionalOutputs(
                        {"role": "user_partial" if kind.startswith("input") else "assistant_partial", "content": text}
                    )
                )
        elif kind == "turn.done":
            turn = getattr(event, "turn", {})
            if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and turn.get("transcript"):
                text = turn["transcript"]
                if turn["role"] == "assistant" and self._music_ready_notice is not None:
                    self._music_ready_notice.set()
                if turn["role"] == "user":
                    turn_id = getattr(event, "turn_id", 0)
                    provider_id = turn.get("id")
                    if isinstance(provider_id, str) and provider_id:
                        if provider_id in self._seen_user_finals:
                            return
                        self._seen_user_finals.add(provider_id)
                    # Stop is idempotent and safety-critical. An ambiguous ID-less
                    # repeated final must not suppress it while new work runs.
                    if fast_control(text) == "stop_music":
                        self._control_for_turn(getattr(event, "turn_id", -1), "stop_music")
                    # Replayed whole deliveries are not another human turn. A
                    # repeated ID-less final needs a matching new actual fragment,
                    # never approval-word matching or an invented provider ID.
                    final_payload = sha256(
                        json.dumps(turn, ensure_ascii=True, sort_keys=True).encode("ascii")
                    ).digest()
                    if final_payload in self._seen_final_payloads and (
                        turn_id != self._fragment_turn or self._fragment_text != text
                    ):
                        return
                    self._seen_final_payloads.add(final_payload)
                    if type(turn_id) is int and turn_id > 0:
                        self._latest_user_turn = max(self._latest_user_turn, turn_id)
                        self._final_user_turns[turn_id] = text
                        self._final_user_turns = dict(list(self._final_user_turns.items())[-8:])
                        waiting = self._final_events.get(turn_id)
                        if waiting is not None:
                            waiting.set()
                    control = fast_control(text)
                    if control is not None:
                        self._control_for_turn(getattr(event, "turn_id", -1), control)
                    elif type(turn_id) is int and turn_id == self._latest_user_turn:
                        self._schedule_music_reply(turn_id, text)
                elif isinstance(text, str):
                    self._assistant_context.append(text)
                    while (
                        len(self._assistant_context) > 4
                        or sum(len(item.encode("utf-8")) for item in self._assistant_context) > 8192
                    ):
                        self._assistant_context.pop(0)
                await self.output_queue.put(AdditionalOutputs({"role": turn["role"], "content": text}))
        elif kind == "delegation.created":
            self.schedule_subscription_delegation(event)
        elif kind == "response.function_call_arguments.done":
            if voice_config.VOICE_BACKEND != "api_key":
                return
            name, arguments, identity = (getattr(event, field, None) for field in ("name", "arguments", "call_id"))
            if isinstance(name, str) and isinstance(arguments, str) and isinstance(identity, str):
                self.schedule_tool_call(name, arguments, identity)
        elif kind == "input_audio_buffer.speech_started":
            clear = getattr(self, "_clear_queue", None)
            if callable(clear):
                clear()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
        elif kind == "input_audio_buffer.speech_stopped":
            self.deps.movement_manager.set_listening(False)
        elif kind in (
            "conversation.item.input_audio_transcription.completed",
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        ):
            role = "user" if kind.startswith("conversation") else "assistant"
            await self.output_queue.put(AdditionalOutputs({"role": role, "content": event.transcript}))
        elif kind in ("response.audio.delta", "response.output_audio.delta"):
            if not self._mute_output_audio:
                if self.deps.head_wobbler is not None:
                    self.deps.head_wobbler.feed(event.delta)
                self.last_activity_time = monotonic()
                await self.output_queue.put(
                    (24000, np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(1, -1))
                )
        elif kind == "response.done":
            if voice_config.VOICE_BACKEND == "api_key" and self._music_ready_notice is not None:
                self._music_ready_notice.set()
            if self._waiting_for_response:
                self._response_done_event.set()
                self._waiting_for_response = False
        elif kind == "error":
            error = getattr(event, "error", {})
            message = (
                error.get("message", "Voice provider error")
                if isinstance(error, dict)
                else getattr(error, "message", "Voice provider error")
            )
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {message}"}))

    async def _end_session_work(self) -> None:
        active = bool(self._work_tasks or self._song_task)
        self._work_epoch += 1
        self._pending_music.clear()
        tasks = list(set(self._tool_tasks.values()) | set(self._subscription_tasks.values()) | self._work_tasks)
        for task in tasks:
            task.cancel()
        if active:
            await self._dispatch("stop_music", {})
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*list(self._control_tasks), return_exceptions=True)
        self._tool_tasks.clear()
        self._subscription_tasks.clear()
        self._work_tasks.clear()
        self._control_tasks.clear()
        self._turn_controls.clear()
        self._song_task = None
        self._music_call_active = False
        self._reply_tasks.clear()
        self._reply_identities.clear()
        self._final_events.clear()
        self._final_user_turns.clear()
        self._assistant_context.clear()
        self._latest_user_turn = 0
        self._fragment_turn = 0
        self._fragment_text = ""
        self._seen_final_payloads.clear()
        self._seen_user_finals.clear()
        self._preview_draft = None

    async def _restart_session(self) -> None:
        """Request one supervised reconnect without launching a second receiver."""
        if self._supervisor is None or self._supervisor.done():
            raise RuntimeError("Voice supervisor is not running; start the handler before reconnecting")
        owner_loop = self._supervisor.get_loop()
        if owner_loop is not asyncio.get_running_loop():
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._restart_session(), owner_loop))
            return
        self._restart_requested = True
        self._pending_music.clear()
        self._work_epoch += 1
        self._connected_event.clear()
        if self.connection is not None:
            await self.connection.close()
        await asyncio.wait_for(self._connected_event.wait(), timeout=45)

    async def reload_instructions(self) -> str:
        """Reload user-owned instructions with one explicit reconnect for either backend."""
        await self._restart_session()
        return "Instructions reloaded and voice reconnected."

    async def apply_personality(self, profile: str | None) -> str:
        """Retain the UI contract; the selected app profile remains config-owned."""
        from dj_reachy_remix.config import set_custom_profile

        set_custom_profile(profile)
        if self.connection is None:
            return "Instructions will load on the next voice connection."
        return await self.reload_instructions()

    async def receive(self, frame: tuple[int, NDArray[np.int16 | np.float32]]) -> None:
        """Send supplied PCM to the selected adapter, never open a microphone device."""
        connection = self.connection
        if connection is None:
            self.diagnostics.drop("no_connection")
            return
        rate, audio = frame
        try:
            if audio.ndim == 2:
                if audio.shape[1] > audio.shape[0]:
                    audio = audio.T
                if audio.shape[1] > 1:
                    audio = audio[:, 0]
            self.diagnostics.audio("selected", audio, rate)
            if rate != 24000:
                audio = resample(audio, int(len(audio) * 24000 / rate))
            self.diagnostics.audio("resampled", audio, 24000)
            pcm = audio_to_int16(audio)
            self.diagnostics.audio("encoded", pcm, 24000)
        except Exception:
            self.diagnostics.drop("conversion_failed")
            raise
        try:
            await connection.input_audio_buffer.append(audio=base64.b64encode(pcm.tobytes()).decode("ascii"))
        except Exception:
            self.diagnostics.drop("append_failed", size_bytes=pcm.nbytes)
            # Propagate to the owner: silently dropping would keep capture alive
            # after a dead transport. Never expose provider/auth response bodies.
            raise VoiceTransportError("Voice input transport failed") from None

    async def emit(self) -> tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit already received media; never infer consumer response completion from PCM."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Cancel all owned tasks, stop local song work, and close transport."""
        self._shutdown_requested = True
        supervisor = self._supervisor
        if supervisor is not None and supervisor is not asyncio.current_task() and not supervisor.done():
            supervisor.cancel()
            await asyncio.gather(supervisor, return_exceptions=True)
        await self._end_session_work()
        if self.connection is not None:
            await self.connection.close()
            self.connection = None
        self._connected_event.clear()
        while not self.output_queue.empty():
            self.output_queue.get_nowait()
        self.diagnostics.close()

    def format_timestamp(self) -> str:
        """Format wall-clock time and monotonic elapsed time."""
        elapsed = monotonic() - self.start_time
        return f"[{datetime.now():%Y-%m-%d %H:%M:%S} | +{elapsed:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Return documented backend choices without a separate authenticated discovery call."""
        if voice_config.VOICE_BACKEND == "subscription":
            return ["cove"]
        return ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"]

    async def trigger_response_and_wait(
        self, system_message: str, response_instructions: str, timeout: float = 15.0
    ) -> bool:
        """Return false for consumer context delivery: its protocol has no response.done guarantee."""
        if self.connection is None:
            return False
        if hasattr(self.connection, "send_context"):
            await self.connection.send_context(system_message, "commentary")
            await self.connection.send_context(response_instructions, "speakable")
            return False
        self._response_done_event.clear()
        self._waiting_for_response = True
        try:
            await self.connection.conversation.item.create(
                item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": system_message}]}
            )
            await self.connection.response.create(response={"instructions": response_instructions})
            await asyncio.wait_for(self._response_done_event.wait(), timeout)
            return True
        except (ConnectionError, asyncio.TimeoutError):
            return False
        finally:
            self._waiting_for_response = False

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Keep compatibility without authorizing unsolicited music generation or motion."""
        if self.connection is not None and hasattr(self.connection, "send_context"):
            await self.connection.send_context(
                f"App idle for {idle_duration:.1f} seconds. No new user command.", "commentary"
            )
