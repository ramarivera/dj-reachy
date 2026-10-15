"""Exercise settings and playback through actual ASGI requests."""
from __future__ import annotations
import json
import asyncio
import threading
from pathlib import Path
from unittest.mock import Mock
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dj_reachy.openai_realtime import OpenaiRealtimeHandler


Dashboard = tuple[TestClient, OpenaiRealtimeHandler, Path]


@pytest.fixture
def dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Dashboard]:
    """Run HTTP and real music controls on separate loops without hardware."""
    from dj_reachy.config import config
    from dj_reachy.console import LocalStream
    from dj_reachy.tools.core_tools import ToolDependencies
    for key in tuple(vars(config)):
        monkeypatch.setattr(config, key, getattr(config, key))
    monkeypatch.setattr(config, "APP_HOME", tmp_path)
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path / "songs")
    monkeypatch.setattr(config, "RUNTIME_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "fixture-present")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fixture-existing")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-present")
    robot = Mock()
    deps = ToolDependencies(reachy_mini=robot, movement_manager=Mock())
    handler = OpenaiRealtimeHandler(deps)
    deps.openai_realtime_handler = handler
    app = FastAPI()
    stream = LocalStream(handler, robot, settings_app=app, instance_path="/must-not-write")
    loop = asyncio.new_event_loop()
    started = threading.Event()
    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()
    thread = threading.Thread(target=run_loop)
    thread.start()
    started.wait(5)
    stream._asyncio_loop = loop
    try:
        with TestClient(app) as client:
            yield client, handler, tmp_path
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def test_controls_stop_actual_tool_queue_without_voice_or_provider(dashboard: Dashboard) -> None:
    """Stop clears actual PCM queued by the active handler."""
    client, handler, _ = dashboard
    handler.output_queue.put_nowait((24000, np.zeros(240, dtype=np.int16)))
    response = client.post("/api/dj-reachy/music/stop", json={})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "stopped"
    assert handler.output_queue.empty()
    assert client.post("/api/dj-reachy/music/unknown", json={}).status_code == 404


def test_saved_files_reject_symlinks_traversal_and_preserve_external_files(dashboard: Dashboard) -> None:
    """Symlink and traversal requests never read or delete external files."""
    client, _, home = dashboard
    songs = home / "songs"
    songs.mkdir()
    outside = home / "outside.mp3"
    outside.write_bytes(b"external")
    (songs / "escape.mp3").symlink_to(outside)
    (songs / "safe.mp3").write_bytes(b"fixture-audio")
    (songs / "safe.json").write_text(json.dumps({"title": "<script>alert(1)</script>"}))
    assert client.get("/songs/escape.mp3").status_code == 404
    assert client.delete("/songs/escape.mp3").status_code == 404
    assert client.delete("/songs/..%2Foutside.mp3").status_code in (400, 404)
    assert outside.read_bytes() == b"external"
    assert [song["filename"] for song in client.get("/songs").json()["songs"]] == ["safe.mp3"]
    assert client.get("/songs/safe.mp3").content == b"fixture-audio"
    assert client.delete("/songs/safe.mp3").status_code == 200
    assert not (songs / "safe.json").exists()


def test_credentials_are_private_and_never_returned_in_settings(dashboard: Dashboard) -> None:
    """Credential persistence uses 0600 and never returns its value."""
    client, _, home = dashboard
    response = client.post("/api/dj-reachy/credentials", json={"provider": "elevenlabs", "key": "fixture-secret"})
    assert response.status_code == 200, response.text
    assert "fixture-secret" not in response.text
    assert (home / ".env").stat().st_mode & 0o777 == 0o600
    assert "fixture-secret" in (home / ".env").read_text()
    assert "fixture-secret" not in client.get("/api/dj-reachy/settings").text
    assert "fixture-secret" not in client.get("/api/dj-reachy/status").text
    assert client.post("/api/dj-reachy/credentials", json={"provider": "openai", "key": "bad\nINJECTED=1"}).status_code == 400


def test_generation_rejects_malformed_plans_before_any_provider_work(dashboard: Dashboard) -> None:
    """Malformed nested plans never reach the music executor."""
    client, _, _ = dashboard
    body: object
    for body in ({"prompt": "test", "music_length_ms": True},
                 {"prompt": "test", "composition_plan": {"chunks": []}},
                 {"composition_plan": []},
                 {"composition_plan": {"chunks": [None]}},
                 {"prompt": {"nested": "invalid"}},
                 {"prompt": "test", "api_key": "fixture-secret"}):
        response = client.post("/api/dj-reachy/music/generate", json=body)
        assert response.status_code == 400, response.text
        assert "fixture-secret" not in response.text
