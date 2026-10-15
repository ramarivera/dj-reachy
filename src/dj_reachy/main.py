"""CLI and Reachy Mini app-manager entry points."""

from __future__ import annotations
import os
import json
import time
import asyncio
import argparse
import threading
from typing import TYPE_CHECKING, Any
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream

from reachy_mini import ReachyMini, ReachyMiniApp
from dj_reachy.utils import parse_args as parse_runtime_args
from dj_reachy.utils import setup_logger, handle_vision_stuff


if TYPE_CHECKING:
    from dj_reachy.openai_realtime import OpenaiRealtimeHandler

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Extend the upstream parser without changing its hardware/media options."""
    args, remaining = parse_runtime_args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--settings-only", action="store_true")
    parser.add_argument("--voice-backend", choices=("subscription", "api_key"), default=None)
    ui_args, remaining = parser.parse_known_args(remaining)
    args.settings_only = ui_args.settings_only
    args.voice_backend = ui_args.voice_backend
    return args, remaining



def update_chatbot(chatbot: list[dict[str, Any]], response: dict[str, Any]) -> list[dict[str, Any]]:
    """Append one assistant or user transcript to the browser conversation."""
    return [*chatbot, response]


def mount_voice_ui(app: FastAPI, handler: OpenaiRealtimeHandler) -> Stream:
    """Mount browser voice at /voice alongside the shared control panel at /."""
    from dj_reachy.config import config
    from dj_reachy.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy.runtime_settings import register_settings_page
    from dj_reachy.tools.core_tools import dispatch_tool_call

    register_settings_page(app, config.RUNTIME_SETTINGS_PATH)
    app.state.dj_dependencies = handler.deps
    chatbot = gr.Chatbot(type="messages", allow_tags=False, resizable=True)
    # Preserve the handler's additional-input positions, but never route credentials
    # through FastRTC's argument channel, which can log values at DEBUG.
    openai_key = gr.State(value="")
    elevenlabs_key = gr.State(value="")
    stream = Stream(
        handler=handler, mode="send-receive", modality="audio", concurrency_limit=1,
        additional_inputs=[chatbot, openai_key, elevenlabs_key], additional_outputs=[chatbot],
        additional_outputs_handler=update_chatbot,
        ui_args={"title": "DJ Reachy voice"},
    )
    with stream.ui:
        gr.HTML('<a href="/" target="_blank" rel="noopener">Open music, conversation and motion controls</a>')
        gr.HTML('<iframe title="DJ Reachy controls" src="/" style="width:100%;height:760px;border:0"></iframe>')

    async def invoke(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        live = [item for item in stream.webrtc_component.handlers.values()
                if isinstance(item, OpenaiRealtimeHandler)]
        if len(live) != 1:
            raise RuntimeError("Connect exactly one browser voice session before using playback.")
        if not hasattr(live[0], "_loop"):
            raise RuntimeError("The browser voice session is still connecting.")
        active = live[0]
        active.deps.openai_realtime_handler = active
        loop = active.loop
        if loop is None or not loop.is_running():
            raise RuntimeError("The browser voice session is not running.")
        work = dispatch_tool_call(tool, json.dumps(arguments), active.deps)
        if loop is asyncio.get_running_loop():
            return await work
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(work, loop))

    app.state.dj_invoke = invoke
    app.state.dj_voice_stream = stream
    gr.mount_gradio_app(app, stream.ui, path="/voice")
    return stream


class SettingsServer:
    """Own the standalone CLI HTTP server, not the SDK-managed server."""

    def __init__(self, app: FastAPI) -> None:
        """Bind to loopback unless the operator explicitly selects another host."""
        import uvicorn
        self.server = uvicorn.Server(uvicorn.Config(
            app, host=os.getenv("DJ_REACHY_UI_HOST", "127.0.0.1"),
            port=int(os.getenv("DJ_REACHY_UI_PORT", "7860")), log_level="warning",
        ))
        self.thread = threading.Thread(target=self.server.run, name="dj-settings", daemon=True)

    def start(self) -> None:
        """Start serving the configured FastAPI application."""
        self.thread.start()

    def close(self) -> None:
        """Stop and join the owned server thread."""
        self.server.should_exit = True
        self.thread.join(timeout=10)


def main() -> None:
    """Run the installed dj-reachy CLI with the existing argument parser."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini | None = None,
    app_stop_event: threading.Event | None = None,
    settings_app: FastAPI | None = None,
    instance_path: str | None = None,
) -> None:
    """Build the app without interpreting the SDK module path as writable data."""
    from dotenv import load_dotenv

    from dj_reachy.moves import MovementManager
    from dj_reachy.config import config, apply_runtime_settings
    from dj_reachy.console import LocalStream
    from dj_reachy.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy.runtime_settings import RuntimeSettingsStore, register_settings_page
    from dj_reachy.tools.core_tools import ToolDependencies
    from dj_reachy.audio.head_wobbler import HeadWobbler

    # The installed module's legacy .env is read-only migration input. Explicit env wins.
    for env_path in (
        config.APP_HOME / ".env",
        Path(instance_path) / ".env" if instance_path else None,
    ):
        if env_path is not None and env_path.is_file():
            load_dotenv(env_path, override=False, interpolate=False)
    config.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
    config.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or config.ELEVENLABS_API_KEY
    apply_runtime_settings(RuntimeSettingsStore(config.RUNTIME_SETTINGS_PATH).read())
    if getattr(args, "voice_backend", None) is not None:
        from dj_reachy.config import update_runtime_settings
        update_runtime_settings({"voice": {"backend": args.voice_backend}})
    if getattr(args, "settings_only", False):
        app = settings_app if settings_app is not None else FastAPI()
        register_settings_page(app, config.RUNTIME_SETTINGS_PATH)
        if settings_app is not None:
            if app_stop_event is not None:
                app_stop_event.wait()
            return
        settings_server = SettingsServer(app)
        settings_server.start()
        try:
            settings_server.thread.join()
        except KeyboardInterrupt:
            pass
        finally:
            settings_server.close()
        return
    managed_server = settings_app is not None
    app = settings_app if settings_app is not None else FastAPI()
    register_settings_page(app, config.RUNTIME_SETTINGS_PATH)
    logger = setup_logger(args.debug)
    if args.no_camera and args.head_tracker is not None:
        logger.warning("Head tracking is disabled by --no-camera.")
    if robot is None:
        backend = "no_media" if args.no_camera else (
            "gstreamer" if args.wireless_version and args.on_device else
            "webrtc" if args.wireless_version else "default"
        )
        robot = ReachyMini(media_backend=backend, localhost_only=not (args.wireless_version and not args.on_device))
    status = robot.client.get_status()
    simulation = status.get("simulation_enabled", False) if isinstance(status, dict) else status.simulation_enabled
    if simulation and not args.gradio:
        robot.client.disconnect()
        raise SystemExit("Simulation requires --gradio.")

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)
    movement = MovementManager(current_robot=robot, camera_worker=camera_worker)
    wobbler = HeadWobbler(set_speech_offsets=movement.set_speech_offsets)
    deps = ToolDependencies(reachy_mini=robot, movement_manager=movement, camera_worker=camera_worker,
                            vision_manager=vision_manager, head_wobbler=wobbler)
    handler = OpenaiRealtimeHandler(deps, gradio_mode=args.gradio, instance_path=str(config.APP_HOME))
    deps.openai_realtime_handler = handler
    local = None if args.gradio else LocalStream(handler, robot, settings_app=app, instance_path=str(config.APP_HOME))
    voice_stream = mount_voice_ui(app, handler) if args.gradio else None
    server = None if managed_server else SettingsServer(app)
    stop = app_stop_event if app_stop_event is not None else threading.Event()
    services = [movement, wobbler, *([camera_worker] if camera_worker else []),
                *([vision_manager] if vision_manager else [])]
    for service in services:
        service.start()
    if server is not None:
        server.start()
    if app_stop_event is not None and local is not None:
        def stop_local() -> None:
            stop.wait()
            local.close()
        threading.Thread(target=stop_local, name="dj-stop", daemon=True).start()
    try:
        if local is not None:
            local.launch()
        else:
            stop.wait()
    except KeyboardInterrupt:
        logger.info("Stopping DJ Reachy")
    finally:
        if local is not None:
            local.close()
        if voice_stream is not None:
            voice_stream.ui.close()
        if server is not None:
            server.close()
        for service in reversed(services):
            service.stop()
        robot.media.close()
        robot.client.disconnect()
        time.sleep(0.1)


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Preserve the upstream Reachy Mini app ID and entry-point class."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Use SDK-owned serving and user-owned configuration in managed mode."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            args, _ = parse_args()
            run(args, robot=reachy_mini, app_stop_event=stop_event,
                settings_app=self.settings_app, instance_path=str(self._get_instance_path().parent))
        finally:
            loop.close()
            asyncio.set_event_loop(None)


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
