"""Local audio loop and user-owned app-manager settings page."""

import asyncio
import logging
from typing import Any
from pathlib import Path

from fastapi import FastAPI
from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

from reachy_mini import ReachyMini
from dj_reachy.config import config
from dj_reachy.openai_realtime import OpenaiRealtimeHandler
from dj_reachy.runtime_settings import register_runtime_settings_routes


logger = logging.getLogger(__name__)


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
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None
        self.handler._clear_queue = self.clear_audio_queue
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
        self._stop_event.clear()

        async def runner() -> None:
            self._asyncio_loop = asyncio.get_running_loop()
            try:
                while config.VOICE_BACKEND == "api_key" and not config.OPENAI_API_KEY:
                    if self._stop_event.is_set():
                        return
                    await asyncio.sleep(0.2)
                if self._stop_event.is_set():
                    return
                self.handler.diagnostics.start()
                self._robot.media.start_recording()
                self._robot.media.start_playing()
                self._tasks = [
                    asyncio.create_task(self.handler.start_up(), name="voice-handler"),
                    asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                    asyncio.create_task(self.play_loop(), name="stream-play-loop"),
                ]
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Local audio session stopped")
            finally:
                for task in self._tasks:
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
                await self.handler.shutdown()
                self._asyncio_loop = None

        asyncio.run(runner())

    def close(self) -> None:
        """Stop media and cancel tasks on the session's owning event loop."""
        for stop in (self._robot.media.stop_recording, self._robot.media.stop_playing):
            try:
                stop()
            except Exception as exc:
                logger.debug("Media shutdown: %s", type(exc).__name__)

        def cancel() -> None:
            self._stop_event.set()
            for task in self._tasks:
                task.cancel()

        if self._asyncio_loop is not None:
            self._asyncio_loop.call_soon_threadsafe(cancel)
        else:
            self._stop_event.set()

    def clear_audio_queue(self) -> None:
        """Drop queued hardware audio when playback is interrupted."""
        audio = self._robot.media.audio
        if audio is not None:
            clear = getattr(audio, "clear_player", None) or getattr(audio, "clear_output_buffer", None)
            if not callable(clear):
                raise RuntimeError("The selected audio backend cannot flush its playback buffer")
            clear()
        self.handler.output_queue = asyncio.Queue()

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
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                self.handler.diagnostics.audio("capture", audio_frame, sample_rate)
                await self.handler.receive((sample_rate, audio_frame))
            else:
                self.handler.diagnostics.drop("capture_empty")
            await asyncio.sleep(0.005)

    async def play_loop(self) -> None:
        """Deliver assistant PCM to the existing local audio backend."""
        while not self._stop_event.is_set():
            output = await self.handler.emit()
            if isinstance(output, AdditionalOutputs):
                # Transcripts may contain secrets supplied by a user. Don't log them.
                continue
            if isinstance(output, tuple):
                sample_rate, audio_data = output
                output_rate = self._robot.media.get_output_audio_samplerate()
                if audio_data.ndim == 2:
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]
                frame = audio_to_float32(audio_data)
                if sample_rate != output_rate:
                    frame = resample(frame, int(len(frame) * output_rate / sample_rate))
                self._robot.media.push_audio_sample(frame)
            await asyncio.sleep(0)
