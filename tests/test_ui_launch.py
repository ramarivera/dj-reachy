"""Headless app construction with real SDK status and intercepted hardware."""
from __future__ import annotations
import time
import asyncio
import threading
from types import SimpleNamespace
from pathlib import Path
from argparse import Namespace
from unittest.mock import Mock
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.daemon.daemon import DaemonState, DaemonStatus


@pytest.fixture(autouse=True)
def construction_loop() -> Iterator[None]:
    """Supply the real handler's synchronous construction loop without connecting it."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        loop.close()
        asyncio.set_event_loop(None)

def test_managed_launch_uses_user_home_not_sdk_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK module paths never own runtime settings, songs or persona."""
    from dj_reachy import main, moves, console
    from dj_reachy.audio import head_wobbler
    from dj_reachy.config import config

    status = DaemonStatus(robot_name="fixture", state=DaemonState.RUNNING,
                          wireless_version=False, desktop_app_daemon=False,
                          stream_enabled=False, simulation_enabled=False, backend_status=None)
    robot = SimpleNamespace(client=SimpleNamespace(get_status=lambda: status, disconnect=Mock()), media=Mock())
    user_home = tmp_path / "owned"
    package_home = tmp_path / "site-packages"
    package_home.mkdir()
    monkeypatch.setattr(config, "APP_HOME", user_home)
    monkeypatch.setattr(config, "SONGS_DIR", user_home / "songs")
    monkeypatch.setattr(config, "RUNTIME_SETTINGS_PATH", user_home / "settings.json")
    monkeypatch.setattr(main, "handle_vision_stuff", lambda *args: (None, None, None))
    monkeypatch.setattr(moves, "MovementManager", Mock())
    monkeypatch.setattr(head_wobbler, "HeadWobbler", Mock())
    monkeypatch.setattr(time, "sleep", lambda _: None)
    launched = []
    monkeypatch.setattr(console.LocalStream, "launch", lambda self: launched.append(self))
    app = FastAPI()
    args = Namespace(debug=False, no_camera=True, head_tracker=None, gradio=False)
    main.run(args, robot=robot, settings_app=app, instance_path=str(package_home))
    assert config.APP_HOME == user_home
    assert config.SONGS_DIR == user_home / "songs"
    assert config.RUNTIME_SETTINGS_PATH == user_home / "settings.json"
    assert list(package_home.iterdir()) == []
    assert len(launched) == 1
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        response = client.put("/api/dj-reachy/settings", json={"music": {"duration_ms": 10000}})
        assert response.status_code == 200, response.text
        assert config.ELEVEN_MUSIC_LENGTH_MS == 10000
        assert client.get("/static/main.js").status_code == 200
    robot.media.start_recording.assert_not_called()
    robot.media.start_playing.assert_not_called()


def test_cli_parser_and_sdk_app_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK entry point retains the real CLI media flags."""
    from dj_reachy import main
    monkeypatch.setattr("sys.argv", ["dj-reachy", "--no-camera", "--gradio"])
    args, unknown = main.parse_args()
    assert args.gradio and args.no_camera
    assert not unknown
    call = Mock()
    monkeypatch.setattr(main, "run", call)
    app = main.ReachyMiniConversationApp()
    app.run(Mock(), threading.Event())
    assert call.call_args.kwargs["settings_app"] is app.settings_app


def test_gradio_mounts_shared_settings_and_does_not_serialize_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The constructed browser app exposes settings even with credentials."""
    from dj_reachy import main
    from dj_reachy.config import config
    from dj_reachy.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy.tools.core_tools import ToolDependencies

    monkeypatch.setattr(config, "APP_HOME", tmp_path)
    monkeypatch.setattr(config, "RUNTIME_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "fixture-not-for-browser")
    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())
    handler = OpenaiRealtimeHandler(deps, gradio_mode=True)
    app = FastAPI()
    main.mount_voice_ui(app, handler)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/dj-reachy/settings").status_code == 200
        response = client.get("/voice/config")
        assert response.status_code == 200, response.text
        assert "fixture-not-for-browser" not in response.text
        assert client.post("/api/dj-reachy/music/stop", json={}).status_code == 409


def test_cli_explicit_backend_and_no_hardware_settings_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select API billing explicitly without constructing a robot."""
    from dj_reachy.main import parse_args
    monkeypatch.setattr("sys.argv", ["dj-reachy", "--settings-only", "--voice-backend", "api_key"])
    args, unknown = parse_args()
    assert not unknown
    assert args.settings_only is True
    assert args.voice_backend == "api_key"


@pytest.mark.asyncio
async def test_gradio_controls_target_live_component_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The active FastRTC copy, not the disconnected template, owns playback."""
    import httpx
    import numpy as np

    from dj_reachy import main
    from dj_reachy.config import config
    from dj_reachy.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy.tools.core_tools import ToolDependencies

    monkeypatch.setattr(config, "RUNTIME_SETTINGS_PATH", tmp_path / "settings.json")
    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())
    template = OpenaiRealtimeHandler(deps, gradio_mode=True)
    active = template.copy()
    active._loop = asyncio.get_running_loop()
    active.output_queue.put_nowait((24000, np.zeros(240, dtype=np.int16)))
    app = FastAPI()
    stream = main.mount_voice_ui(app, template)
    stream.webrtc_component.handlers["fixture-connection"] = active
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
        response = await client.post("/api/dj-reachy/music/stop", json={})
    assert response.status_code == 200, response.text
    assert active.output_queue.empty()
    assert deps.openai_realtime_handler is active
