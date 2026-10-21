"""Offline LocalStream/real-handler regressions with no SDK or provider construction."""
from __future__ import annotations
import time
import asyncio
import threading
from types import SimpleNamespace
from typing import NoReturn, cast
from pathlib import Path
from contextlib import asynccontextmanager
from unittest.mock import Mock
from collections.abc import Callable, Iterator, AsyncIterator

import numpy as np
import pytest
from fastapi import FastAPI
from numpy.typing import NDArray
from fastapi.testclient import TestClient

from reachy_mini import ReachyMini
from dj_reachy_remix.config import config
from dj_reachy_remix.console import LocalStream
from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler
from dj_reachy_remix.tools.core_tools import ToolDependencies


SECRET = "fixture-private-provider-body"


class OfflineMedia:
    """Only synchronous in-memory device boundaries; never construct an SDK."""

    def __init__(self) -> None:
        """Create only an in-memory lifecycle recorder."""
        self.calls: list[str] = []
        self.audio = SimpleNamespace(clear_player=lambda: self.calls.append("flush"))

    def start_recording(self) -> None:
        """Observe capture activation without opening a microphone."""
        self.calls.append("start_recording")

    def start_playing(self) -> None:
        """Observe output activation without opening a speaker."""
        self.calls.append("start_playing")

    def stop_recording(self) -> None:
        """Observe capture teardown."""
        self.calls.append("stop_recording")

    def stop_playing(self) -> None:
        """Observe output teardown."""
        self.calls.append("stop_playing")

    def get_audio_sample(self) -> None:
        """Count capture polls without recording anything."""
        self.calls.append("capture")
        return None

    def get_input_audio_samplerate(self) -> int:
        """Return the fixture's input rate."""
        return 24000

    def get_input_channels(self) -> int:
        """Return the fixture's mono layout."""
        return 1


def wait_until(predicate: Callable[[], object]) -> None:
    """Bound all cross-thread observations without assuming scheduler timing."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "offline lifecycle did not settle"


@pytest.fixture
def offline_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Guard prior to handler/base construction and inject at provider connect."""
    monkeypatch.setattr(config, "RUNTIME_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "VOICE_BACKEND", "subscription")
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", None)
    construction_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(construction_loop)
    media = OfflineMedia()
    robot = cast(ReachyMini, SimpleNamespace(media=media))
    deps = ToolDependencies(reachy_mini=robot, movement_manager=Mock())
    handler = OpenaiRealtimeHandler(deps)
    deps.openai_realtime_handler = handler
    attempts: list[str] = []

    @asynccontextmanager
    async def connect(**kwargs: object) -> AsyncIterator[object]:
        attempts.append(config.VOICE_BACKEND)
        raise RuntimeError(SECRET)
        yield  # pragma: no cover - required async context-manager shape

    monkeypatch.setattr(handler, "_create_client", lambda key: setattr(
        handler, "client", SimpleNamespace(realtime=SimpleNamespace(connect=connect))))
    app = FastAPI()
    stream = LocalStream(handler, robot, settings_app=app)
    errors: list[BaseException] = []

    def launch() -> None:
        try:
            stream.launch()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=launch, name="offline-stream-fixture")
    try:
        yield SimpleNamespace(stream=stream, handler=handler, media=media, app=app,
                              worker=worker, errors=errors, attempts=attempts)
    finally:
        stream.close()
        if worker.ident is not None:
            worker.join(4)
            assert not worker.is_alive(), "offline launch leaked its owner thread"
        construction_loop.close()
        asyncio.set_event_loop(None)


def test_settings_survive_exact_handler_startup_failure(offline_stream: SimpleNamespace, caplog: pytest.LogCaptureFixture) -> None:
    """Keep the UI owner and settings writable after the reported startup error."""
    fixture = offline_stream
    fixture.worker.start()
    wait_until(lambda: "stop_recording" in fixture.media.calls)
    assert fixture.worker.is_alive(), "handler.start_up failure terminated the UI owner"
    assert fixture.errors == []
    with TestClient(fixture.app) as client:
        assert client.get("/").status_code == 200
        response = client.put("/api/dj-reachy-remix/settings", json={"motion": {"intensity": 0.1}})
        assert response.status_code == 200, response.text
    captures = fixture.media.calls.count("capture")
    time.sleep(0.05)
    assert fixture.media.calls.count("capture") == captures
    assert fixture.attempts == ["subscription"]
    assert all(task.done() for task in fixture.stream._tasks)
    assert SECRET not in caplog.text


def test_transport_status_rejects_stale_handshake_after_close(offline_stream: SimpleNamespace) -> None:
    """A retained handshake flag must not report an already closed transport ready."""
    fixture = offline_stream
    fixture.stream._voice_state = "connecting"
    fixture.handler._connected_event.set()
    fixture.handler.connection = SimpleNamespace(_closed=True)
    assert fixture.stream.voice_status()["voice_state"] != "connected"
    assert fixture.stream.voice_status()["voice_reconnect_available"] is False


def test_offline_status_is_sanitized_and_does_not_claim_ready(offline_stream: SimpleNamespace) -> None:
    """Expose transport failure without exposing its body or claiming readiness."""
    fixture = offline_stream
    fixture.worker.start()
    wait_until(lambda: "stop_recording" in fixture.media.calls)
    with TestClient(fixture.app) as client:
        state = client.get("/api/dj-reachy-remix/status").json()
        assert state["voice_state"] == "offline"
        assert state["voice_error"] == "voice_transport"
        assert state["provider_access"] == "unverified"
        assert SECRET not in str(state)
        assert client.post("/api/dj-reachy-remix/music/stop", json={}).status_code == 409


def test_only_explicit_reconnect_opens_one_new_session(offline_stream: SimpleNamespace) -> None:
    """Require a fresh user POST for each attempt and finish every native owner."""
    fixture = offline_stream
    fixture.worker.start()
    wait_until(lambda: "stop_recording" in fixture.media.calls)
    first_capture = fixture.stream._capture_io
    with TestClient(fixture.app) as client:
        for attempt in (2, 3):
            state = client.get("/api/dj-reachy-remix/status").json()
            assert state["voice_reconnect_available"] is True
            response = client.post("/api/dj-reachy-remix/voice/reconnect", json={})
            assert response.status_code == 202, response.text
            wait_until(lambda: fixture.media.calls.count("stop_recording") == attempt)
            wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "offline")
            time.sleep(0.03)
            assert fixture.attempts == ["subscription"] * attempt
            assert fixture.stream._capture_io is not first_capture
            assert not fixture.stream._capture_io._thread.is_alive()
            assert all(task.done() for task in fixture.stream._tasks)
    fixture.stream.close()
    fixture.stream.close()
    fixture.worker.join(3)
    assert not fixture.worker.is_alive()
    assert fixture.errors == []
    assert fixture.stream.voice_status()["voice_state"] == "stopped"


@pytest.mark.parametrize("failure,category", [
    ("end", "voice_transport"), ("receive", "voice_transport"),
    ("append", "voice_transport"), ("capture", "media"), ("start_media", "media"),
])
def test_runtime_failures_quiesce_capture_and_remain_distinguishable(
    offline_stream: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str, category: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Contain provider EOF/send/receive and media failures at real task boundaries."""
    fixture = offline_stream
    fail = threading.Event()
    closed = threading.Event()

    class Connection:
        input_audio_buffer: Connection

        async def append(self, **kwargs: object) -> None:
            if fail.is_set() and failure == "append":
                raise RuntimeError(SECRET)

        def __aiter__(self) -> Connection:
            return self

        async def __anext__(self) -> NoReturn:
            while not fail.is_set() or failure in ("append", "capture"):
                await asyncio.sleep(0.005)
            if failure == "end":
                raise StopAsyncIteration
            raise ConnectionError(SECRET)

        async def close(self) -> None:
            closed.set()

    connection = Connection()
    connection.input_audio_buffer = connection

    @asynccontextmanager
    async def connect(**kwargs: object) -> AsyncIterator[object]:
        fixture.attempts.append(config.VOICE_BACKEND)
        try:
            yield connection
        finally:
            await connection.close()

    def capture() -> NDArray[np.float32]:
        fixture.media.calls.append("capture")
        if fail.is_set() and failure == "capture":
            raise RuntimeError(SECRET)
        import numpy as np
        return np.ones(240, dtype=np.float32)

    monkeypatch.setattr(fixture.media, "get_audio_sample", capture)
    monkeypatch.setattr(fixture.handler, "_create_client", lambda key: setattr(
        fixture.handler, "client", SimpleNamespace(realtime=SimpleNamespace(connect=connect))))
    if failure == "start_media":
        def start() -> NoReturn:
            raise RuntimeError(SECRET)
        monkeypatch.setattr(fixture.media, "start_playing", start)
    fixture.worker.start()
    if failure != "start_media":
        wait_until(lambda: fixture.handler._connected_event.is_set())
        assert fixture.stream.voice_status()["voice_state"] == "connected"
        fail.set()
    wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "offline")
    assert fixture.stream.voice_status()["voice_error"] == category
    captured = fixture.media.calls.count("capture")
    time.sleep(0.03)
    assert captured == fixture.media.calls.count("capture")
    assert fixture.attempts == ([] if failure == "start_media" else ["subscription"])
    assert fixture.worker.is_alive() and fixture.errors == []
    assert all(task.done() for task in fixture.stream._tasks)
    assert failure == "start_media" or closed.is_set()
    assert SECRET not in caplog.text


@pytest.mark.parametrize("boundary,category", [("handler", "voice_shutdown"), ("media", "media_shutdown")])
def test_cleanup_failure_keeps_settings_but_refuses_reconnect(offline_stream: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, boundary: str, category: str, caplog: pytest.LogCaptureFixture) -> None:
    """Refuse to replace an owner whose cleanup did not complete safely."""
    fixture = offline_stream
    if boundary == "handler":
        original = fixture.handler.shutdown

        async def failed_shutdown() -> None:
            await original()
            raise RuntimeError(SECRET)
        monkeypatch.setattr(fixture.handler, "shutdown", failed_shutdown)
    else:
        def failed_stop() -> NoReturn:
            raise RuntimeError(SECRET)
        monkeypatch.setattr(fixture.media, "stop_playing", failed_stop)
    fixture.worker.start()
    wait_until(lambda: fixture.stream.voice_status()["voice_state"] in ("offline", "stopped"))
    assert fixture.worker.is_alive() and fixture.errors == []
    with TestClient(fixture.app) as client:
        state = client.get("/api/dj-reachy-remix/status").json()
        assert state["voice_error"] == category
        assert state["voice_reconnect_available"] is False
        assert client.get("/api/dj-reachy-remix/settings").status_code == 200
        assert client.post("/api/dj-reachy-remix/voice/reconnect", json={}).status_code == 409
    assert SECRET not in caplog.text


def test_settings_change_cannot_turn_subscription_failure_into_api_retry(offline_stream: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Fence billing selection even when a settings save races transport failure."""
    fixture = offline_stream

    class PrivateTransportFailure(ConnectionError):
        pass

    @asynccontextmanager
    async def connect(**kwargs: object) -> AsyncIterator[object]:
        fixture.attempts.append(config.VOICE_BACKEND)
        # A UI settings save races a subscription transport failure.
        config.VOICE_BACKEND = "api_key"
        raise PrivateTransportFailure(SECRET)
        yield

    monkeypatch.setattr(fixture.handler, "_create_client", lambda key: setattr(
        fixture.handler, "client", SimpleNamespace(realtime=SimpleNamespace(connect=connect))))
    fixture.worker.start()
    wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "offline")
    assert fixture.attempts == ["subscription"]
    assert "PrivateTransportFailure" not in caplog.text
    assert SECRET not in caplog.text


def test_reconnect_recovers_with_one_receiver_and_close_is_safe(offline_stream: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recover once through the public endpoint and reject duplicate live owners."""
    fixture = offline_stream
    fixture.worker.start()
    wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "offline")
    closed = threading.Event()

    class Connection:
        def __aiter__(self) -> Connection:
            return self

        async def __anext__(self) -> object:
            return await asyncio.Future[object]()

        async def close(self) -> None:
            closed.set()

    @asynccontextmanager
    async def connect(**kwargs: object) -> AsyncIterator[object]:
        fixture.attempts.append(config.VOICE_BACKEND)
        try:
            yield Connection()
        finally:
            closed.set()

    monkeypatch.setattr(fixture.handler, "_create_client", lambda key: setattr(
        fixture.handler, "client", SimpleNamespace(realtime=SimpleNamespace(connect=connect))))
    with TestClient(fixture.app) as client:
        assert client.post("/api/dj-reachy-remix/voice/reconnect", json={}).status_code == 202
        wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "connected")
        assert client.post("/api/dj-reachy-remix/voice/reconnect", json={}).status_code == 409
        state = client.get("/api/dj-reachy-remix/status").json()
        assert state["voice_state"] == "connected" and state["voice_error"] is None
        assert state["voice_reconnect_available"] is False
    assert fixture.attempts == ["subscription", "subscription"]
    assert len([task for task in fixture.stream._tasks if task.get_name() == "voice-handler" and not task.done()]) == 1
    fixture.stream.close()
    fixture.stream.close()
    fixture.worker.join(3)
    assert not fixture.worker.is_alive() and fixture.errors == []
    assert closed.is_set() and not fixture.handler._connected_event.is_set()


def test_failure_discards_native_filter_state_before_explicit_reconnect(offline_stream: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Discard actual resampler history rather than replaying it after reconnect."""
    import numpy as np

    fixture = offline_stream
    pushed = threading.Event()
    fail = threading.Event()

    class Connection:
        def __aiter__(self) -> Connection:
            return self

        async def __anext__(self) -> NoReturn:
            while not fail.is_set():
                await asyncio.sleep(0.005)
            raise ConnectionError(SECRET)

    @asynccontextmanager
    async def connect(**kwargs: object) -> AsyncIterator[object]:
        yield Connection()

    monkeypatch.setattr(fixture.handler, "_create_client", lambda key: setattr(
        fixture.handler, "client", SimpleNamespace(realtime=SimpleNamespace(connect=connect))))
    monkeypatch.setattr(fixture.media, "get_output_audio_samplerate", lambda: 24000, raising=False)
    monkeypatch.setattr(fixture.media, "push_audio_sample", lambda frame: pushed.set(), raising=False)
    fixture.handler.output_queue.put_nowait((12000, np.ones(1200, dtype=np.int16)))
    fixture.worker.start()
    wait_until(pushed.is_set)
    fail.set()
    wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "offline")
    assert fixture.stream._pcm_converter is None, "stale filter tail can cross reconnect"
    assert fixture.stream._pcm_rates is None
    assert fixture.stream._playback_clock == 0.0
    assert fixture.stream._play_epoch > 0
    assert fixture.handler.output_queue.empty()


def test_status_schema_is_stable_without_a_voice_stream(tmp_path: Path) -> None:
    """Settings-only mode must still report reconnect availability (PR #8 review)."""
    from dj_reachy_remix.runtime_settings import register_runtime_settings_routes

    app = FastAPI()
    register_runtime_settings_routes(app, tmp_path / "settings.json")
    with TestClient(app) as client:
        state = client.get("/api/dj-reachy-remix/status").json()
    assert state["voice_state"] == "not_connected"
    assert state["voice_reconnect_available"] is False


def test_cancelled_flush_does_not_cross_an_explicit_reconnect(
    offline_stream: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flush cancelled by session cleanup must not cancel the next session's audio."""
    fixture = offline_stream
    fail = threading.Event()

    class Connection:
        def __aiter__(self) -> Connection:
            return self

        async def __anext__(self) -> NoReturn:
            while not fail.is_set():
                await asyncio.sleep(0.005)
            raise ConnectionError(SECRET)

    @asynccontextmanager
    async def connect(**kwargs: object) -> AsyncIterator[object]:
        yield Connection()

    monkeypatch.setattr(fixture.handler, "_create_client", lambda key: setattr(
        fixture.handler, "client", SimpleNamespace(realtime=SimpleNamespace(connect=connect))))
    fixture.worker.start()
    wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "connected")

    async def start_flush() -> None:
        fixture.stream._flush_task = asyncio.create_task(asyncio.sleep(30), name="native-audio-flush")

    loop = fixture.stream._asyncio_loop
    asyncio.run_coroutine_threadsafe(start_flush(), loop).result(2)
    fail.set()
    wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "offline")
    assert fixture.stream._flush_task is None, "cancelled flush leaked into the next session"


def test_session_failure_stops_http_owned_playback(
    offline_stream: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Music started from the control panel must stop before the offline wait (PR #8 review)."""
    import dj_reachy_remix.tools.core_tools as core_tools

    fixture = offline_stream
    fail = threading.Event()
    playing = threading.Event()
    cancelled = threading.Event()
    stopped: list[str] = []

    class Connection:
        def __aiter__(self) -> Connection:
            return self

        async def __anext__(self) -> NoReturn:
            while not fail.is_set():
                await asyncio.sleep(0.005)
            raise ConnectionError(SECRET)

    @asynccontextmanager
    async def connect(**kwargs: object) -> AsyncIterator[object]:
        yield Connection()

    async def dispatch(tool: str, args_json: str, deps: object) -> dict[str, object]:
        if tool == "stop_music":
            stopped.append(tool)
            return {"status": "stopped"}
        playing.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {}

    monkeypatch.setattr(fixture.handler, "_create_client", lambda key: setattr(
        fixture.handler, "client", SimpleNamespace(realtime=SimpleNamespace(connect=connect))))
    monkeypatch.setattr(core_tools, "dispatch_tool_call", dispatch)
    fixture.worker.start()
    wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "connected")
    http_loop = asyncio.new_event_loop()
    try:
        invocation = http_loop.create_task(fixture.stream.invoke("play_saved_song", {"filename": "fixture.mp3"}))
        http_loop.run_until_complete(asyncio.sleep(0))
        wait_until(playing.is_set)
        fail.set()
        wait_until(lambda: fixture.stream.voice_status()["voice_state"] == "offline")
        assert cancelled.is_set(), "HTTP-owned playback survived the session failure"
        assert stopped == ["stop_music"]
        with pytest.raises(asyncio.CancelledError):
            http_loop.run_until_complete(asyncio.wait_for(invocation, 2))
    finally:
        http_loop.close()
