"""Entrypoint for the DJ Reachy conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
import tempfile
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from dj_reachy.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    from pathlib import Path
    
    # Load instance .env BEFORE importing modules that use config
    # This ensures REACHY_MINI_CUSTOM_PROFILE is set before tools are loaded
    if instance_path:
        try:
            from dotenv import load_dotenv
            from dj_reachy.config import config, set_custom_profile
            
            env_path = Path(instance_path) / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=True)
                # Update config with profile from instance .env
                new_profile = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
                if new_profile is not None:
                    set_custom_profile(new_profile.strip() or None)
                # Update API keys
                new_key = os.getenv("OPENAI_API_KEY", "").strip()
                if new_key:
                    config.OPENAI_API_KEY = new_key
                new_el_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
                if new_el_key:
                    config.ELEVENLABS_API_KEY = new_el_key
        except Exception as e:
            print(f"Warning: Failed to load instance .env: {e}")
    from dj_reachy.config import apply_runtime_settings, config
    from dj_reachy.runtime_settings import RuntimeSettingsStore, register_runtime_settings_routes
    runtime_settings_path = Path(instance_path) / "settings.json" if instance_path else config.RUNTIME_SETTINGS_PATH
    runtime_store = RuntimeSettingsStore(runtime_settings_path)
    config.RUNTIME_SETTINGS_PATH = runtime_settings_path
    from dj_reachy import config as config_module
    config_module.runtime_settings_store = runtime_store
    apply_runtime_settings(runtime_store.read())
    settings_app = settings_app or FastAPI()
    register_runtime_settings_routes(settings_app, runtime_settings_path)
    if instance_path:
        config.APP_HOME = Path(instance_path)
        config.SONGS_DIR = config.APP_HOME / "songs"
    
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from dj_reachy.moves import MovementManager
    from dj_reachy.console import LocalStream
    from dj_reachy.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy.tools.core_tools import ToolDependencies
    from dj_reachy.audio.head_wobbler import HeadWobbler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    if args.no_camera and args.head_tracker is not None:
        logger.warning("Head tracking is not activated due to --no-camera.")

    if robot is None:
        # Initialize robot with appropriate backend
        # TODO: Implement dynamic robot connection detection
        # Automatically detect and connect to available Reachy Mini robot(s!)
        # Priority checks (in order):
        #   1. Reachy Lite connected directly to the host
        #   2. Reachy Mini daemon running on localhost (same device)
        #   3. Reachy Mini daemon on local network (same subnet)

        # Determine media backend based on args
        if args.no_camera:
            media_backend = "no_media"
            logger.info("Using no_media backend (camera disabled)")
        elif args.wireless_version and not args.on_device:
            media_backend = "webrtc"
            logger.info("Using WebRTC backend for fully remote wireless version")
        elif args.wireless_version and args.on_device:
            media_backend = "gstreamer"
            logger.info("Using GStreamer backend for on-device wireless version")
        else:
            media_backend = "default"
            logger.info("Using default backend for lite version")

        robot = ReachyMini(
            media_backend=media_backend,
            localhost_only=not (args.wireless_version and not args.on_device),
        )

    # Check if running in simulation mode without --gradio
    if robot.client.get_status()["simulation_enabled"] and not args.gradio:
        logger.error(
            "Simulation mode requires Gradio interface. Please use --gradio flag when running in simulation mode.",
        )
        robot.client.disconnect()
        sys.exit(1)

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = OpenaiRealtimeHandler(deps, gradio_mode=args.gradio, instance_path=instance_path)

    # Wire handler to deps so tools can access the output queue for audio streaming
    deps.openai_realtime_handler = handler

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        from dj_reachy.gradio_theme import create_dj_reachy_theme, DJ_REACHY_CSS
        from dj_reachy.gradio_dj_ui import DJReachyUI

        # Create custom theme
        dj_theme = create_dj_reachy_theme()

        openai_key_textbox = gr.Textbox(
            label="🔑 OpenAI API Key",
            type="password",
            placeholder="sk-...",
            value=os.getenv("OPENAI_API_KEY") if not get_space() else "",
        )
        elevenlabs_key_textbox = gr.Textbox(
            label="🎵 ElevenLabs API Key (for music generation)",
            type="password",
            placeholder="Required for make_song_and_dance",
            value=os.getenv("ELEVENLABS_API_KEY") if not get_space() else "",
        )

        # Create the Stream first (this creates the base UI)
        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[
                chatbot,
                openai_key_textbox,
                elevenlabs_key_textbox,
            ],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={
                "title": "🎧 DJ Reachy",
                "theme": dj_theme,
                "css": DJ_REACHY_CSS,
            },
        )
        stream_manager = stream.ui

        # Now add DJ UI components inside the Stream's Blocks context
        dj_ui = DJReachyUI()
        with stream_manager:
            with gr.Accordion("🎵 Song library and playback", open=True):
                with gr.Row():
                    with gr.Column(scale=2):
                        dj_ui.songs_list = gr.Dataframe(
                            headers=["Filename", "Title", "Size"],
                            datatype=["str", "str", "str"],
                            value=dj_ui._list_songs(),
                            label="Saved tracks",
                            interactive=False,
                            wrap=True,
                        )
                        with gr.Row():
                            dj_ui.refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")
                            dj_ui.play_btn = gr.Button("Play selected", variant="primary", size="sm")
                            dj_ui.delete_btn = gr.Button("Delete selected", variant="stop", size="sm")
                    with gr.Column(scale=1):
                        with gr.Row():
                            dj_ui.pause_btn = gr.Button("Pause")
                            dj_ui.resume_btn = gr.Button("Resume")
                            dj_ui.stop_btn = gr.Button("Stop", variant="stop")
                        dj_ui.dance_intensity = gr.Slider(
                            label="Dance intensity", minimum=0, maximum=100,
                            value=dj_ui._get_dance_intensity(), step=1,
                        )
                        dj_ui.status_md = gr.Markdown(value="", visible=True)
                        dj_ui.song_audio = gr.Audio(label="Track preview", type="filepath",
                                                    interactive=False, visible=True, autoplay=False)

            with gr.Accordion("Music generation", open=False):
                music_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the song to generate.")
                composition_plan = gr.Textbox(label="Versioned composition plan JSON (optional)", lines=5)
                music_model = gr.Dropdown(
                    choices=["music_v1", "music_v2", "music_v2_5"], value=config.MUSIC_MODEL_ID,
                    label="Eleven Music model",
                )
                music_duration = gr.Slider(
                    minimum=3, maximum=600, step=1, value=config.ELEVEN_MUSIC_LENGTH_MS // 1000,
                    label="Duration (seconds)",
                )
                positive_styles = gr.Textbox(value=", ".join(config.MUSIC_POSITIVE_STYLES), label="Styles")
                negative_styles = gr.Textbox(value=", ".join(config.MUSIC_NEGATIVE_STYLES), label="Excluded styles")
                music_lyrics = gr.Textbox(value=config.MUSIC_LYRICS, label="Original lyrics", lines=4)
                music_language = gr.Textbox(value=config.MUSIC_LANGUAGE, label="Lyric language (auto or language name)")
                force_instrumental = gr.Checkbox(value=config.MUSIC_FORCE_INSTRUMENTAL, label="Force instrumental (prompt mode)")
                generate_music_btn = gr.Button("Generate and play", variant="primary")

            with gr.Accordion("Voice, billing, and motion settings", open=False):
                voice_backend = gr.Dropdown(
                    choices=[("ChatGPT subscription Live", "subscription"), ("OpenAI API-key Realtime", "api_key")],
                    value=config.VOICE_BACKEND, label="Voice backend and billing",
                )
                voice_model = gr.Textbox(value=config.VOICE_MODEL, label="Realtime model")
                voice_name = gr.Textbox(value=config.VOICE_NAME, label="Conversational voice (not the Eleven singer)")
                conversation_language = gr.Textbox(value=config.CONVERSATION_LANGUAGE, label="Conversation language (auto or language name)")
                music_output_format = gr.Dropdown(
                    choices=sorted(__import__("dj_reachy.runtime_settings", fromlist=["_ALLOWED_FORMATS"])._ALLOWED_FORMATS),
                    value=config.MUSIC_OUTPUT_FORMAT, label="Music output format",
                )
                motion_style = gr.Dropdown(
                    choices=["groove", "bounce", "smooth", "classic"], value=config.MOTION_STYLE, label="Motion style",
                )
                instructions_text = gr.Textbox(
                    value=(config.APP_HOME / "instructions.md").read_text(encoding="utf-8")
                    if (config.APP_HOME / "instructions.md").is_file() else "",
                    label="User-owned voice instructions", lines=6,
                )
                save_settings_btn = gr.Button("Save settings and reload voice")
                settings_status = gr.Markdown(value="")

        async def _generate_music(prompt: str, plan_json: str, model: str, duration_s: float,
                                  styles: str, exclusions: str, lyrics: str, language: str,
                                  instrumental: bool) -> str:
            if bool(prompt.strip()) == bool(plan_json.strip()):
                return "Enter exactly one of a prompt or a versioned composition plan."
            args: dict[str, Any] = {
                "music_length_ms": int(duration_s * 1000), "model_id": model,
                "positive_styles": [item.strip() for item in styles.split(",") if item.strip()],
                "negative_styles": [item.strip() for item in exclusions.split(",") if item.strip()],
                "lyrics": lyrics, "language": language, "force_instrumental": instrumental,
            }
            if plan_json.strip():
                try:
                    import json
                    args["composition_plan"] = json.loads(plan_json)
                except json.JSONDecodeError:
                    return "Composition plan must be valid JSON."
            else:
                args["prompt"] = prompt.strip()
            from dj_reachy.tools.core_tools import dispatch_tool_call
            import json
            result = await dispatch_tool_call("make_song_and_dance", json.dumps(args), deps)
            return str(result.get("status") or result.get("error") or "Generation finished.")

        async def _save_settings(backend: str, model: str, voice: str, language: str, music_model_id: str,
                                 duration_s: float, fmt: str, style: str, intensity_value: float,
                                 styles: str, exclusions: str, lyrics: str, lyric_language: str,
                                 instrumental: bool, instructions: str) -> str:
            updates = {
                "voice": {"backend": backend, "model": model, "voice": voice, "language": language},
                "music": {"model_id": music_model_id, "duration_ms": int(duration_s * 1000),
                          "output_format": fmt,
                          "positive_styles": [item.strip() for item in styles.split(",") if item.strip()],
                          "negative_styles": [item.strip() for item in exclusions.split(",") if item.strip()],
                          "lyrics": lyrics, "language": lyric_language, "force_instrumental": instrumental},
                "motion": {"style": style, "intensity": max(0.0, min(100.0, intensity_value)) / 400.0},
            }
            try:
                saved = runtime_store.update(updates)
                apply_runtime_settings(saved)
                from dj_reachy.profiles.dj_reachy.make_song_and_dance import set_dance_intensity, set_dance_style
                set_dance_intensity(saved["motion"]["intensity"])
                set_dance_style(saved["motion"]["style"])
                instructions_path = config.APP_HOME / "instructions.md"
                instructions_path.parent.mkdir(parents=True, exist_ok=True)
                fd, temp_name = tempfile.mkstemp(prefix=".instructions.", suffix=".tmp", dir=instructions_path.parent)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as stream_file:
                        stream_file.write(instructions[:12000])
                        stream_file.flush()
                        os.fsync(stream_file.fileno())
                    os.replace(temp_name, instructions_path)
                finally:
                    Path(temp_name).unlink(missing_ok=True)
                await handler._restart_session()
                return "Settings saved. Voice reconnect requested; subscription access is still unverified."
            except (ValueError, OSError) as exc:
                return f"Settings not saved: {exc}"

        generate_music_btn.click(
            fn=_generate_music,
            inputs=[music_prompt, composition_plan, music_model, music_duration, positive_styles,
                    negative_styles, music_lyrics, music_language, force_instrumental],
            outputs=[dj_ui.status_md],
        )
        save_settings_btn.click(
            fn=_save_settings,
            inputs=[voice_backend, voice_model, voice_name, conversation_language, music_model,
                    music_duration, music_output_format, motion_style, dj_ui.dance_intensity,
                    positive_styles, negative_styles, music_lyrics, music_language, force_instrumental,
                    instructions_text],
            outputs=[settings_status],
        )
        dj_ui.wire_events(handler, stream_manager)
        app = gr.mount_gradio_app(settings_app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
        )

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        # is_wireless = reachy_mini.client.get_status()["wireless_version"]
        # args.head_tracker = None if is_wireless else "mediapipe"

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
