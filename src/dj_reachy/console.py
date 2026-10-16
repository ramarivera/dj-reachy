"""Local audio loop and user-owned app-manager settings page."""

import os
import time
import asyncio
import logging
import threading
from queue import Queue
from typing import Any, TypeVar, NoReturn, cast
from pathlib import Path
from collections.abc import Callable
from concurrent.futures import Future

import numpy as np
from fastapi import FastAPI
from fastrtc import AdditionalOutputs, audio_to_float32
from numpy.typing import NDArray
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

from reachy_mini import ReachyMini
from dj_reachy.config import config
from dj_reachy.openai_realtime import OpenaiRealtimeHandler
from dj_reachy.runtime_settings import register_runtime_settings_routes
from dj_reachy.audio.streaming_pcm import StreamingPCMConverter
from dj_reachy.audio.native_playback import configure_native_playback


logger = logging.getLogger(__name__)
_AudioResult = TypeVar("_AudioResult")


class _PlaybackFence(AdditionalOutputs):
    """An internal FIFO marker, never a user-visible chat event."""

    def __init__(self, epoch: int, completed: asyncio.Future[bool]) -> None:
        super().__init__()
        self.epoch = epoch
        self.completed = completed


class _AudioIO:
    """One lazy daemon owner, one mailbox slot, and at most one native call.

    Cancellation releases the async waiter, NOT the synchronous operation. The
    retained future fences subsequent calls (including stop) until it completes.
    No use of asyncio's executor: asyncio.run would join stalled executor threads.
    """

    def __init__(self, name: str) -> None:
        self._mailbox: Queue[tuple[Callable[[], object], Future[object]] | None] = Queue(maxsize=1)
        self._lock = asyncio.Lock()
        self._pending: asyncio.Future[object] | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False
        self._finished = False

    async def _drain(self) -> None:
        if self._pending is not None:
            try:
                await asyncio.shield(self._pending)
            except Exception:
                # The caller observes operation failures; stop must still run.
                pass

    def _run(self) -> None:
        while (item := self._mailbox.get()) is not None:
            operation, result = item
            try:
                result.set_result(operation())
            except BaseException as exc:
                result.set_exception(exc)

    async def call(self, operation: Callable[[], _AudioResult]) -> _AudioResult:
        """Apply backpressure and retain ownership across waiter cancellation."""
        async with self._lock:
            await self._drain()
            if self._finished:
                raise RuntimeError("Native audio owner has already stopped")
            if not self._started:
                self._thread.start()
                self._started = True
            result: Future[object] = Future()
            self._pending = asyncio.wrap_future(result)
            # Retrieve exceptions even if cancellation removes the only waiter.
            self._pending.add_done_callback(lambda done: done.exception())
            self._mailbox.put_nowait((operation, result))
            return cast(_AudioResult, await asyncio.shield(self._pending))

    async def finish(self) -> None:
        """Drain the owned operation before asking the daemon owner to exit."""
        async with self._lock:
            await self._drain()
            if self._finished:
                return
            self._finished = True
            if self._started:
                self._mailbox.put_nowait(None)
                while self._thread.is_alive():
                    await asyncio.sleep(0.001)


class LocalStream:
    """Own the audio tasks; HTTP controls execute on their owning loop."""

    def __init__(
        self,
        handler: OpenaiRealtimeHandler,
        robot: ReachyMini,
        *,
        settings_app: FastAPI | None = None,
        instance_path: str | None = None,
    ) -> None:
        """Attach settings immediately, regardless of credential availability."""
        self.handler = handler
        self._robot = robot
        self._output_tail_delay = configure_native_playback(getattr(robot.media, "audio", None))
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._closing = False
        self._launched = False
        self._media_started = False
        self._play_epoch = 0
        self._playback_clock = 0.0
        self._playback_interrupt = threading.Event()
        self._quiet_seconds = 0.0
        self._native_idle = False
        self._pcm_converter: StreamingPCMConverter | None = None
        self._pcm_rates: tuple[int, int] | None = None
        self._output_changed = asyncio.Event()
        self._flushed_epoch = 0
        self._flush_task: asyncio.Task[None] | None = None
        self._emit_task: asyncio.Task[tuple[int, NDArray[np.int16]] | AdditionalOutputs | None] | None = None
        self._capture_io = _AudioIO("native-audio-capture")
        self._playback_io = _AudioIO("native-audio-playback")
        self.handler._clear_queue = self.clear_audio_queue
        self.handler._drain_queue = self.drain_audio if self._output_tail_delay else None
        self._settings_app = settings_app
        # instance_path is retained in the public constructor, never a storage root.
        self._init_settings_ui_if_needed()

    def _init_settings_ui_if_needed(self) -> None:
        """Serve package assets, which are also present in an installed wheel."""
        app = self._settings_app
        if app is None:
            return
        register_runtime_settings_routes(app, config.RUNTIME_SETTINGS_PATH)
        static_dir = Path(__file__).parent / "static"
        if not any(getattr(route, "path", None) == "/static" for route in app.routes):
            app.mount("/static", StaticFiles(directory=static_dir), name="static")
        if not any(getattr(route, "path", None) == "/" for route in app.routes):

            @app.get("/")
            def settings_page() -> FileResponse:
                return FileResponse(static_dir / "index.html")

        app.state.dj_invoke = self.invoke
        app.state.dj_dependencies = self.handler.deps

    async def invoke(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Keep decoder/task ownership off the HTTP server's separate event loop."""
        import json

        from dj_reachy.tools.core_tools import dispatch_tool_call

        loop = self._asyncio_loop
        if loop is None or not loop.is_running():
            raise RuntimeError("Audio session is not running. Connect voice before playback.")
        future = asyncio.run_coroutine_threadsafe(
            dispatch_tool_call(tool, json.dumps(arguments), self.handler.deps), loop
        )
        return await asyncio.wrap_future(future)

    def launch(self) -> None:
        """Run the explicitly selected voice backend, with no key-download fallback."""
        if self._launched:
            raise RuntimeError("LocalStream owns a single native audio session")
        self._launched = True

        async def runner() -> None:
            self._asyncio_loop = asyncio.get_running_loop()
            self._runner_task = asyncio.current_task()
            try:
                while config.VOICE_BACKEND == "api_key" and not config.OPENAI_API_KEY:
                    if self._stop_event.is_set():
                        return
                    await asyncio.sleep(0.2)
                if self._stop_event.is_set():
                    return
                self.handler.diagnostics.start()
                self._media_started = True
                await self._capture_io.call(self._robot.media.start_recording)
                await self._playback_io.call(self._robot.media.start_playing)
                self._tasks = [
                    asyncio.create_task(self.handler.start_up(), name="voice-handler"),
                    asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                    asyncio.create_task(self.play_loop(), name="stream-play-loop"),
                ]
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Local audio session stopped")
            finally:
                self._closing = True
                self._stop_event.set()
                for task in self._tasks:
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
                await self._shutdown_audio()
                await self.handler.shutdown()
                self._asyncio_loop = None

        asyncio.run(runner())

    def close(self) -> None:
        """Request teardown; never race stop/close against an owned SDK call."""
        self._playback_interrupt.set()

        def cancel() -> None:
            self._stop_event.set()
            self._output_changed.set()
            if not self._closing and self._runner_task is not None:
                self._runner_task.cancel()

        loop = self._asyncio_loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(cancel)
        else:
            self._stop_event.set()
            self._output_changed.set()

    async def drain_audio(self) -> bool:
        """Fence preceding native pushes and wait a cancellable scheduled tail."""
        loop = asyncio.get_running_loop()
        owner = self._asyncio_loop
        if owner is not None and owner is not loop:
            if owner.is_closed():
                return False
            return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self.drain_audio(), owner))
        if self._closing or self._stop_event.is_set():
            return False
        epoch = self._play_epoch
        changed = self._output_changed
        completed: asyncio.Future[bool] = loop.create_future()
        self.handler.output_queue.put_nowait(_PlaybackFence(epoch, completed))
        invalidated: asyncio.Task[bool] = asyncio.create_task(changed.wait())
        try:
            watched: set[asyncio.Future[bool]] = {completed, invalidated}
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
            if invalidated in done or epoch != self._play_epoch or self._stop_event.is_set():
                return False
            scheduled = max(0.0, self._playback_clock - time.monotonic())
            tail_done, _ = await asyncio.wait({invalidated}, timeout=self._output_tail_delay + scheduled)
            return not tail_done and epoch == self._play_epoch and not self._stop_event.is_set() and not self._closing
        finally:
            if not completed.done():
                completed.cancel()
            invalidated.cancel()
            await asyncio.gather(invalidated, return_exceptions=True)

    def clear_audio_queue(self) -> None:
        """Invalidate PCM now; serialize the native flush after any in-flight push.

        The handler/music contract is synchronous. Return means invalidated, not
        that a blocked native push was interrupted: the one playback owner must
        finish it, flush, and only then accept replacement-generation frames.
        """
        try:
            caller_loop = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        owner_loop = getattr(self, "_asyncio_loop", None)
        if owner_loop is not None and caller_loop is not owner_loop:
            if not owner_loop.is_closed():
                # Wake a paced owner even if the event loop cannot dispatch yet.
                self._playback_interrupt.set()
                owner_loop.call_soon_threadsafe(self.clear_audio_queue)
            return
        if getattr(self, "_closing", False):
            return
        audio = self._robot.media.audio
        clear = None
        if audio is not None:
            clear = getattr(audio, "clear_player", None) or getattr(audio, "clear_output_buffer", None)
            if not callable(clear):
                raise RuntimeError("The selected audio backend cannot flush its playback buffer")
        interrupt = getattr(self, "_playback_interrupt", None)
        if interrupt is not None:
            interrupt.set()
        self._playback_interrupt = threading.Event()
        self.handler.output_queue = asyncio.Queue()
        self._playback_clock = 0.0
        self._quiet_seconds = 0.0
        self._native_idle = False
        self._pcm_converter = None
        self._pcm_rates = None
        if caller_loop is None:
            # Pre-launch callers have no native owner or in-flight work.
            if callable(clear):
                clear()
            return
        self._play_epoch += 1
        self._output_changed.set()
        self._output_changed = asyncio.Event()
        if self._emit_task is not None:
            self._emit_task.cancel()
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_output(clear), name="native-audio-flush")

    async def _flush_output(self, clear: Callable[[], object] | None) -> None:
        while self._flushed_epoch != self._play_epoch:
            epoch = self._play_epoch
            if clear is not None:
                await self._playback_io.call(clear)
            self._flushed_epoch = epoch

    _IO_SHUTDOWN_TIMEOUT = 2.0

    async def _shutdown_audio(self) -> None:
        """Bound native shutdown without pretending cancellation kills threads."""
        self._playback_interrupt.set()
        self._output_changed.set()
        cleanup = asyncio.create_task(self._drain_and_stop_audio(), name="native-audio-shutdown")
        done, _ = await asyncio.wait({cleanup}, timeout=self._IO_SHUTDOWN_TIMEOUT)
        if not done:
            self._native_io_stalled()
        await cleanup

    def _native_io_stalled(self) -> NoReturn:
        """Terminate only at an unrecoverable native-stall boundary.

        main.run's finally unconditionally closes media after launch returns, so
        neither returning nor raising can safely leave a native SDK call alive.
        Python cannot terminate a thread. A process supervisor must restart us;
        do not replace this with to_thread cancellation or a fake clean shutdown.
        """
        logger.critical("Unrecoverable native audio I/O stall; terminating process before unsafe media close")
        os._exit(70)

    async def _drain_and_stop_audio(self) -> None:
        """Drain native operations, then stop media on those same owners."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            await asyncio.gather(self._flush_task, return_exceptions=True)

        async def stop(io: _AudioIO, operation: Callable[[], object]) -> None:
            try:
                if self._media_started:
                    await io.call(operation)
            finally:
                await io.finish()

        # Media managers may share a pipeline: quiesce BOTH directions before
        # either stop can tear down shared native resources.
        await asyncio.gather(self._capture_io._drain(), self._playback_io._drain())
        results = await asyncio.gather(
            stop(self._capture_io, self._robot.media.stop_recording),
            stop(self._playback_io, self._robot.media.stop_playing),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.debug("Media shutdown: %s", type(result).__name__)
        self._media_started = False

    async def record_loop(self) -> None:
        """Forward microphone frames to the active voice adapter."""
        sample_rate = self._robot.media.get_input_audio_samplerate()
        try:
            self.handler.diagnostics.capture_configuration(
                backend=self._robot.media.audio,
                rate=sample_rate,
                channels=self._robot.media.get_input_channels(),
            )
        except Exception:
            self.handler.diagnostics.capture_configuration_unavailable()
        while not self._stop_event.is_set():
            audio_frame = await self._capture_io.call(self._robot.media.get_audio_sample)
            if audio_frame is not None:
                self.handler.diagnostics.audio("capture", audio_frame, sample_rate)
                await self.handler.receive((sample_rate, audio_frame))
            else:
                self.handler.diagnostics.drop("capture_empty")
            await asyncio.sleep(0.005)

    async def _finish_native_conversion(self, epoch: int) -> None:
        """Submit the ordered filter tail; interruptions discard it instead."""
        converter, rates = self._pcm_converter, self._pcm_rates
        self._pcm_converter = None
        self._pcm_rates = None
        if converter is not None and rates is not None:
            for frame in converter.finish():
                await self._submit_native_audio(frame, rates[1], epoch)

    async def _submit_native_audio(self, frame: NDArray[np.float32], sample_rate: int, epoch: int) -> None:
        """Pace <=100ms owner batches, with <=20ms SDK writes and <=100ms lead.

        Only the existing serialized owner writes or sleeps. Completion resumes
        on the main loop once per batch, not once per native chunk. The epoch's
        thread-safe signal interrupts pacing before a serialized flush/stop.
        """
        interrupted = self._playback_interrupt
        mono = frame.reshape(-1)
        chunk_samples = max(1, sample_rate // 50)
        batch_samples = max(1, sample_rate // 10)
        audio = self._robot.media.audio
        clear = getattr(audio, "clear_player", None) or getattr(audio, "clear_output_buffer", None)

        def push_batch(batch: NDArray[np.float32]) -> None:
            for offset in range(0, batch.size, chunk_samples):
                if interrupted.is_set() or epoch != self._play_epoch:
                    return
                chunk = batch[offset : offset + chunk_samples]
                duration = chunk.size / sample_rate
                now = time.monotonic()
                if now > self._playback_clock + 0.2:
                    self._playback_clock = now
                delay = max(0.0, self._playback_clock + duration - 0.1 - now)
                if interrupted.wait(delay):
                    return
                suppress_idle = False
                if self._output_tail_delay > 0 and callable(clear):
                    quiet = float(np.mean(chunk * chunk)) <= 1e-8
                    self._quiet_seconds = self._quiet_seconds + duration if quiet else 0.0
                    if quiet and self._quiet_seconds >= max(0.6, self._output_tail_delay + 0.2):
                        # Preserve carrier timing without aging the native cue.
                        self._native_idle = True
                        suppress_idle = True
                    elif not quiet and self._native_idle:
                        # Drain preceding speech and a potentially wider tail.
                        if interrupted.wait(max(0.0, self._playback_clock - time.monotonic())):
                            return
                        clear()
                        if interrupted.is_set() or epoch != self._play_epoch:
                            return
                        self._native_idle = False
                        self._playback_clock = time.monotonic()
                if interrupted.is_set() or epoch != self._play_epoch:
                    return
                if not suppress_idle:
                    self._robot.media.push_audio_sample(chunk)
                if interrupted.is_set() or epoch != self._play_epoch:
                    return
                self._playback_clock += duration

        for offset in range(0, mono.size, batch_samples):
            if interrupted.is_set() or epoch != self._play_epoch or self._stop_event.is_set():
                return
            if self._flush_task is not None:
                await asyncio.shield(self._flush_task)
            batch = mono[offset : offset + batch_samples]
            await self._playback_io.call(lambda: push_batch(batch))

    async def play_loop(self) -> None:
        """Deliver ordered PCM through the single playback owner."""
        pending: tuple[int, NDArray[np.int16]] | AdditionalOutputs | None = None
        pending_epoch = self._play_epoch
        output: tuple[int, NDArray[np.int16]] | AdditionalOutputs | None
        try:
            while not self._stop_event.is_set():
                epoch = self._play_epoch
                if pending is not None and pending_epoch == epoch:
                    output, pending = pending, None
                else:
                    pending = None
                    self._emit_task = asyncio.create_task(self.handler.emit())
                    try:
                        output = await self._emit_task
                    except asyncio.CancelledError:
                        current = asyncio.current_task()
                        if current is not None and current.cancelling():
                            raise
                        # A flush replaced the queue while emit was waiting.
                        continue
                    finally:
                        self._emit_task = None
                if isinstance(output, _PlaybackFence):
                    if epoch == self._play_epoch == output.epoch and not output.completed.done():
                        await self._finish_native_conversion(epoch)
                        if epoch == self._play_epoch and not output.completed.done():
                            output.completed.set_result(True)
                    continue
                if epoch != self._play_epoch or isinstance(output, AdditionalOutputs):
                    continue
                if isinstance(output, tuple):
                    sample_rate, audio_data = output
                    output_rate = self._robot.media.get_output_audio_samplerate()
                    if audio_data.ndim == 2:
                        if audio_data.shape[1] > audio_data.shape[0]:
                            audio_data = audio_data.T
                        if audio_data.shape[1] > 1:
                            audio_data = audio_data[:, 0]
                    frame: NDArray[np.float32] = audio_to_float32(audio_data).reshape(-1)
                    # Never wait for more audio. Stop at the first incompatible
                    # format, fence or control, retaining that exact FIFO item.
                    frames = [frame]
                    samples = frame.size
                    while samples < max(1, sample_rate // 10):
                        try:
                            following = self.handler.output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if (
                            not isinstance(following, tuple)
                            or following[0] != sample_rate
                            or following[1].ndim != 1
                            or samples + following[1].size > max(1, sample_rate // 10)
                        ):
                            pending, pending_epoch = following, epoch
                            break
                        frames.append(audio_to_float32(following[1]).reshape(-1))
                        samples += following[1].size
                    if len(frames) > 1:
                        frame = np.concatenate(frames)
                    rates = (sample_rate, output_rate)
                    if rates != self._pcm_rates:
                        await self._finish_native_conversion(epoch)
                        if epoch != self._play_epoch:
                            continue
                        self._pcm_converter = StreamingPCMConverter(*rates)
                        self._pcm_rates = rates
                    converter = self._pcm_converter
                    if converter is not None:
                        for converted in converter.push(frame):
                            await self._submit_native_audio(converted, output_rate, epoch)
                await asyncio.sleep(0)
        finally:
            if self._emit_task is not None:
                self._emit_task.cancel()
                await asyncio.gather(self._emit_task, return_exceptions=True)
                self._emit_task = None
