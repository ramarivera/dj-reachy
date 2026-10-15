import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from dj_reachy.config import config
from dj_reachy.prompts import get_session_voice, get_session_instructions
from dj_reachy.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

OPEN_AI_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OPEN_AI_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """An OpenAI realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=OPEN_AI_INPUT_SAMPLE_RATE,
        )

        # Override typing of the sample rates to match OpenAI's requirements
        self.output_sample_rate: Literal[24000] = self.output_sample_rate
        self.input_sample_rate: Literal[24000] = self.input_sample_rate

        self.deps = deps

        # Override type annotations for OpenAI strict typing (only for values used in API)
        self.output_sample_rate = OPEN_AI_OUTPUT_SAMPLE_RATE
        self.input_sample_rate = OPEN_AI_INPUT_SAMPLE_RATE

        self.connection: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0  # sequence counter to prevent stale emissions
        self.partial_debounce_delay = 0.5  # seconds

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # Muting flag for external audio playback (e.g., songs)
        self._mute_output_audio: bool = False

        # Event for waiting on triggered responses to complete
        self._response_done_event: asyncio.Event = asyncio.Event()
        self._waiting_for_response: bool = False

        self._tool_tasks: dict[str, asyncio.Task[None]] = {}
        self._music_call_active = False
        self._audio_mute_owner: str | None = None
        self._subscription_tasks: dict[str, asyncio.Task[None]] = {}

    def schedule_tool_call(self, tool_name: str, args_json: str, call_id: str) -> asyncio.Task[None]:
        """Run a tool independently from event reception and retain its result correlation."""
        existing = self._tool_tasks.get(call_id)
        if existing is not None:
            return existing
        task = asyncio.create_task(self._run_tool_call(tool_name, args_json, call_id), name=f"tool-{call_id}")
        self._tool_tasks[call_id] = task
        task.add_done_callback(lambda _task: self._tool_tasks.pop(call_id, None))
        return task

    async def _run_tool_call(self, tool_name: str, args_json: str, call_id: str) -> None:
        connection = self.connection
        if tool_name == "make_song_and_dance" and self._music_call_active:
            result: dict[str, Any] = {"status": "busy", "message": "A song is already generating or playing."}
        else:
            if tool_name == "make_song_and_dance":
                self._music_call_active = True
            try:
                result = await dispatch_tool_call(tool_name, args_json, self.deps)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                if tool_name == "make_song_and_dance":
                    self._music_call_active = False
        if connection is None or self.connection is not connection or self._shutdown_requested:
            return
        await connection.conversation.item.create(
            item={"type": "function_call_output", "call_id": call_id, "output": json.dumps(result)}
        )
        await self.output_queue.put(
            AdditionalOutputs(
                {"role": "assistant", "content": json.dumps(result),
                 "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"}}
            )
        )
        if not self.is_idle_tool_call:
            await connection.response.create(
                response={"instructions": "Use the tool result just returned and answer concisely."}
            )
        else:
            self.is_idle_tool_call = False
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()
    def schedule_subscription_delegation(self, event: Any) -> None:
        """Run one correlated subscription delegation without blocking event reception."""
        item = getattr(event, "item", {})
        if isinstance(item, dict):
            delegation_id = item.get("id") or item.get("delegation_item_id")
            content = item.get("content", [])
            event_text = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            event_text = event_text or str(item.get("text") or "")
        else:
            delegation_id = getattr(item, "id", None)
            event_text = str(getattr(item, "text", "") or "")
        if not isinstance(delegation_id, str) or not delegation_id:
            delegation_id = getattr(event, "delegation_item_id", None)
        if not isinstance(delegation_id, str) or not delegation_id or delegation_id in self._subscription_tasks:
            return
        transcript = str(getattr(event, "transcript_context", "") or "")
        task = asyncio.create_task(
            self._run_subscription_delegation(delegation_id, event_text, transcript),
            name=f"delegation-{delegation_id}",
        )
        self._subscription_tasks[delegation_id] = task
        task.add_done_callback(lambda _task: self._subscription_tasks.pop(delegation_id, None))

    async def _run_subscription_delegation(self, delegation_id: str, event_text: str, transcript: str) -> None:
        connection = self.connection
        if connection is None or not hasattr(connection, "send_context"):
            return
        from dj_reachy.subscription_voice import plan_delegation
        action = plan_delegation(event_text, transcript)
        if action is None:
            spoken = "I can control music and dance, but I couldn't map that request to an app command."
        else:
            tool_name, arguments = action
            if tool_name == "make_song_and_dance" and self._music_call_active:
                result: dict[str, Any] = {"status": "busy"}
            else:
                if tool_name == "make_song_and_dance":
                    self._music_call_active = True
                await connection.send_context("Working on that now.", "commentary", delegation_id)
                try:
                    result = await dispatch_tool_call(tool_name, json.dumps(arguments), self.deps)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    result = {"error": type(exc).__name__}
                finally:
                    if tool_name == "make_song_and_dance":
                        self._music_call_active = False
            status = str(result.get("status") or "")
            if status in ("completed", "saved"):
                spoken = "Done. The track finished."
            elif status == "paused":
                spoken = "The music is paused."
            elif status == "resumed":
                spoken = "The music resumed."
            elif status == "busy":
                spoken = "A track is already generating or playing."
            elif status == "cancelled":
                spoken = str(result.get("message") or "Local work stopped. The provider job may have continued.")
            elif status == "stopped":
                spoken = "Music stopped."
            elif status in ("not_playing", "not_paused"):
                spoken = "There isn't active music to control."
            elif "error" in result:
                spoken = "That app command failed."
            else:
                spoken = f"Done: {status or 'command completed'}."
        if self.connection is connection and not self._shutdown_requested:
            await connection.send_context(spoken, "speakable", delegation_id)

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    def mute_output_audio(self, owner: str = "external") -> None:
        """Mute assistant audio for one playback owner."""
        self._audio_mute_owner = owner
        self._mute_output_audio = True

    def unmute_output_audio(self, owner: str = "external") -> None:
        """Release only the matching playback owner's mute."""
        if self._audio_mute_owner == owner:
            self._audio_mute_owner = None
            self._mute_output_audio = False

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        - Updates the global config's selected profile for subsequent calls.
        - If a realtime connection is active, sends a session.update with the
          freshly resolved instructions so the change takes effect immediately.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from dj_reachy.config import config as _config
            from dj_reachy.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = get_session_instructions()
                voice = get_session_voice()
            except BaseException as e:  # catch SystemExit from prompt loader without crashing
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Attempt a live update first, then force a full restart to ensure it sticks
            if self.connection is not None:
                try:
                    await self.connection.session.update(
                        session={
                            "type": "realtime",
                            "instructions": instructions,
                            "audio": {"output": {"voice": voice}},
                        },
                    )
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

                # Force a real restart to guarantee the new instructions/voice
                try:
                    await self._restart_session()
                    return "Applied personality and restarted realtime session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                logger.info(
                    "Applied personality recorded: %s (no live connection; will apply on next session)",
                    profile or "built-in default",
                )
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _emit_debounced_partial(self, transcript: str, sequence: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)
            # Only emit if this is still the latest partial (by sequence number)
            if self.partial_transcript_sequence == sequence:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    async def start_up(self) -> None:
        from dj_reachy.config import set_api_keys
        openai_api_key = config.OPENAI_API_KEY
        elevenlabs_api_key = config.ELEVENLABS_API_KEY

        if self.gradio_mode:
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_openai_key = args[3] if len(args) > 3 and isinstance(args[3], str) and args[3].strip() else None
            textbox_elevenlabs_key = args[4] if len(args) > 4 and isinstance(args[4], str) and args[4].strip() else None
            if textbox_openai_key:
                openai_api_key = textbox_openai_key
            if textbox_elevenlabs_key:
                elevenlabs_api_key = textbox_elevenlabs_key
            set_api_keys(openai_api_key, elevenlabs_api_key)

        self.deps.elevenlabs_api_key = elevenlabs_api_key.strip() if elevenlabs_api_key else None
        if config.VOICE_BACKEND == "subscription":
            from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter
            self.client = SubscriptionRealtimeAdapter(
                model=config.VOICE_MODEL,
                voice=config.VOICE_NAME,
                instructions=get_session_instructions(),
            )
        elif config.VOICE_BACKEND == "api_key":
            if not openai_api_key or not openai_api_key.strip():
                raise RuntimeError("API-key Realtime is selected, but OPENAI_API_KEY is missing. Choose subscription Live or provide a key.")
            self.client = AsyncOpenAI(api_key=openai_api_key.strip())
        else:
            raise RuntimeError(f"Unsupported voice backend {config.VOICE_BACKEND!r}; no billing fallback is allowed.")

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                return
            except (ConnectionClosedError, ConnectionError) as e:
                logger.warning("Realtime connection closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, type(e).__name__)
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** (attempt - 1) + random.uniform(0, 0.5))
                    continue
                raise
            finally:
                self.connection = None
                self._connected_event.clear()


    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: OpenAI client not initialized yet.")
                return
            if config.VOICE_BACKEND == "subscription":
                from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter
                self.client = SubscriptionRealtimeAdapter(
                    model=config.VOICE_MODEL, voice=config.VOICE_NAME, instructions=get_session_instructions()
                )
            elif config.VOICE_BACKEND == "api_key":
                if not config.OPENAI_API_KEY:
                    raise RuntimeError("API-key Realtime selected without OPENAI_API_KEY; subscription was not used.")
                self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            else:
                raise RuntimeError("Unsupported voice backend; no billing fallback is allowed.")

            # Fire-and-forget new session and wait briefly for connection
            try:
                self._connected_event.clear()
            except Exception:
                pass
            asyncio.create_task(self._run_realtime_session(), name="openai-realtime-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        model = config.VOICE_MODEL if config.VOICE_BACKEND == "subscription" else config.MODEL_NAME
        async with self.client.realtime.connect(model=model) as conn:
            try:
                session_config: dict[str, Any] = {
                    "type": "realtime",
                    "instructions": get_session_instructions(),
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
                            "turn_detection": {"type": "server_vad", "interrupt_response": True},
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": self.output_sample_rate},
                            "voice": get_session_voice(),
                        },
                    },
                }
                if config.VOICE_BACKEND == "api_key":
                    session_config["audio"]["input"]["transcription"] = {
                        "model": "gpt-4o-transcribe",
                        **({"language": config.CONVERSATION_LANGUAGE} if config.CONVERSATION_LANGUAGE != "auto" else {}),
                    }
                    session_config["tools"] = get_tool_specs()
                    session_config["tool_choice"] = "auto"
                await conn.session.update(session=session_config)
                logger.info("Realtime session initialized for backend=%s", config.VOICE_BACKEND)
            except Exception:
                logger.exception("Realtime session.update failed; aborting startup")
                return

            logger.info("Realtime session updated successfully")

            # Manage event received from the openai server
            self.connection = conn
            
            # Trigger automatic startup greeting
            await self._send_startup_greeting()
            self._connected_event.set()
            async for event in self.connection:
                logger.debug("Realtime event: %s", event.type)
                if event.type == "input_transcript.added":
                    text = str(getattr(event, "text", "") or "")
                    item = getattr(event, "item", None)
                    if not text and isinstance(item, dict):
                        text = str(item.get("text") or "")
                    if text:
                        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))
                if event.type == "output_transcript.added":
                    text = str(getattr(event, "text", "") or "")
                    if text:
                        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))
                if event.type == "turn.done":
                    turn = getattr(event, "turn", {})
                    role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", None)
                    transcript = turn.get("transcript") if isinstance(turn, dict) else getattr(turn, "transcript", "")
                    if role in ("user", "assistant") and transcript:
                        await self.output_queue.put(AdditionalOutputs({"role": role, "content": transcript}))
                if event.type == "delegation.created":
                    self.schedule_subscription_delegation(event)
                if event.type == "input_audio_buffer.speech_started":
                    if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                        self._clear_queue()
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()
                    self.deps.movement_manager.set_listening(True)
                    logger.debug("User speech started")

                if event.type == "input_audio_buffer.speech_stopped":
                    self.deps.movement_manager.set_listening(False)
                    logger.debug("User speech stopped - server will auto-commit with VAD")

                if event.type in (
                    "response.audio.done",  # GA
                    "response.output_audio.done",  # GA alias
                    "response.audio.completed",  # legacy (for safety)
                    "response.completed",  # text-only completion
                ):
                    logger.debug("response completed")

                if event.type == "response.created":
                    logger.debug("Response created")

                if event.type == "response.done":
                    # Doesn't mean the audio is done playing
                    logger.debug("Response done")
                    # Signal any waiting trigger that the response is complete
                    if self._waiting_for_response:
                        self._response_done_event.set()
                        self._waiting_for_response = False

                # Handle partial transcription (user speaking in real-time)
                if event.type == "conversation.item.input_audio_transcription.partial":
                    logger.debug(f"User partial transcript: {event.transcript}")

                    # Increment sequence
                    self.partial_transcript_sequence += 1
                    current_sequence = self.partial_transcript_sequence

                    # Cancel previous debounce task if it exists
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass

                    # Start new debounce timer with sequence number
                    self.partial_transcript_task = asyncio.create_task(
                        self._emit_debounced_partial(event.transcript, current_sequence)
                    )

                # Handle completed transcription (user finished speaking)
                if event.type == "conversation.item.input_audio_transcription.completed":
                    logger.debug(f"User transcript: {event.transcript}")

                    # Cancel any pending partial emission
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass

                    await self.output_queue.put(AdditionalOutputs({"role": "user", "content": event.transcript}))

                # Handle assistant transcription
                if event.type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                    logger.debug(f"Assistant transcript: {event.transcript}")
                    await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": event.transcript}))

                # Handle audio delta
                if event.type in ("response.audio.delta", "response.output_audio.delta"):
                    if self._mute_output_audio:
                        # Drop assistant audio while an external track is playing
                        continue
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.feed(event.delta)
                    self.last_activity_time = asyncio.get_event_loop().time()
                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(1, -1),
                        ),
                    )

                # ---- tool-calling plumbing ----
                if event.type == "response.function_call_arguments.done":
                    tool_name = getattr(event, "name", None)
                    args_json_str = getattr(event, "arguments", None)
                    call_id = getattr(event, "call_id", None)
                    if not all(isinstance(value, str) for value in (tool_name, args_json_str, call_id)):
                        logger.error("Ignoring uncorrelated or malformed tool call event")
                        continue
                    self.schedule_tool_call(tool_name, args_json_str, call_id)

                # server error
                if event.type == "error":
                    err = getattr(event, "error", None)
                    msg = getattr(err, "message", str(err) if err else "unknown error")
                    code = getattr(err, "code", "")

                    logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

                    # Only show user-facing errors, not internal state errors
                    if code not in ("input_audio_buffer_commit_empty", "conversation_already_has_active_response"):
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                        )

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the OpenAI server.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for OpenAI's API. Resamples if the input sample rate differs
        from the expected rate.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        if not self.connection:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Send to OpenAI (guard against races during reconnect)
        try:
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream

        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped (connection closed?): %s", e)
                return None

            self.last_activity_time = asyncio.get_event_loop().time()  # avoid repeated resets

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True
        # Cancel any pending debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass
        owned_tasks = [*self._tool_tasks.values(), *self._subscription_tasks.values()]
        for task in owned_tasks:
            task.cancel()
        if owned_tasks:
            await asyncio.gather(*owned_tasks, return_exceptions=True)
        self._tool_tasks.clear()
        self._subscription_tasks.clear()

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Try to discover available voices for the configured realtime model.

        Attempts to retrieve model metadata from the OpenAI Models API and look
        for any keys that might contain voice names. Falls back to a curated
        list known to work with realtime if discovery fails.
        """
        # Conservative fallback list with default first
        fallback = [
            "cedar",
            "alloy",
            "aria",
            "ballad",
            "verse",
            "sage",
            "coral",
        ]
        try:
            # Best effort discovery; safe-guarded for unexpected shapes
            model = await self.client.models.retrieve(config.MODEL_NAME)
            # Try common serialization paths
            raw = None
            for attr in ("model_dump", "to_dict"):
                fn = getattr(model, attr, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        pass
            if raw is None:
                try:
                    raw = dict(model)
                except Exception:
                    raw = None
            # Scan for voice candidates
            candidates: set[str] = set()

            def _collect(obj: object) -> None:
                try:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            kl = str(k).lower()
                            if "voice" in kl and isinstance(v, (list, tuple)):
                                for item in v:
                                    if isinstance(item, str):
                                        candidates.add(item)
                                    elif isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                                        candidates.add(item["name"])
                            else:
                                _collect(v)
                    elif isinstance(obj, (list, tuple)):
                        for it in obj:
                            _collect(it)
                except Exception:
                    pass

            if isinstance(raw, dict):
                _collect(raw)
            # Ensure default present and stable order
            voices = sorted(candidates) if candidates else fallback
            if "cedar" not in voices:
                voices = ["cedar", *[v for v in voices if v != "cedar"]]
            return voices
        except Exception:
            return fallback

    async def trigger_response_and_wait(
        self,
        system_message: str,
        response_instructions: str,
        timeout: float = 15.0,
    ) -> bool:
        """Trigger a response and wait for it to complete speaking.
        
        Args:
            system_message: The [SYSTEM: ...] message to inject
            response_instructions: Instructions for this specific response
            timeout: Maximum time to wait for response completion
            
        Returns:
            True if response completed, False if timed out or failed
        """
        if not self.connection:
            logger.debug("No connection, cannot trigger response")
            return False
        
        try:
            # Reset the event and set waiting flag
            self._response_done_event.clear()
            self._waiting_for_response = True
            
            # Create the system message
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": system_message}],
                },
            )
            
            # Trigger the response with specific instructions
            await self.connection.response.create(
                response={
                    "instructions": response_instructions + " Always respond in ENGLISH.",
                },
            )
            
            # Wait for the response to complete (response.done event)
            try:
                await asyncio.wait_for(self._response_done_event.wait(), timeout=timeout)
                logger.info("Triggered response completed")
                # Give a small buffer for audio to finish playing
                await asyncio.sleep(0.5)
                return True
            except asyncio.TimeoutError:
                logger.warning("Triggered response timed out after %.1fs", timeout)
                self._waiting_for_response = False
                return False
                
        except Exception as e:
            logger.warning("Failed to trigger response: %s", e)
            self._waiting_for_response = False
            return False

    async def _send_startup_greeting(self) -> None:
        """Send a startup signal to trigger the automatic greeting."""
        if not self.connection:
            logger.debug("No connection, cannot send startup greeting")
            return
        
        logger.info("Triggering startup greeting...")
        try:
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "[SYSTEM: Application just started. Greet the user warmly in ENGLISH and ask what kind of song they'd like!]"}],
                },
            )
            await self.connection.response.create(
                response={
                    "instructions": "Give a warm, enthusiastic greeting IN ENGLISH and ask the user what kind of song they'd like. Do NOT generate any songs - just greet and ask. SPEAK ENGLISH ONLY.",
                },
            )
            logger.info("Startup greeting triggered")
        except Exception as e:
            logger.warning("Failed to send startup greeting: %s", e)

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to the openai server."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        timestamp_msg = f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] You've been idle for a while. Feel free to get creative - dance, show an emotion, look around, do nothing, or just be yourself!"
        if not self.connection:
            logger.debug("No connection, cannot send idle signal")
            return
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": timestamp_msg}],
            },
        )
        await self.connection.response.create(
            response={
                "instructions": "You MUST respond with function calls only - no speech or text. Choose appropriate actions for idle behavior.",
                "tool_choice": "required",
            },
        )
