"""Exercise managed SIGINT teardown in a real process, without devices or providers."""
from __future__ import annotations
import os
import sys
import time
import signal
import textwrap
import subprocess
from pathlib import Path

import pytest


CHILD = textwrap.dedent("""
    import asyncio, faulthandler, gc, runpy, sys, threading, time
    from argparse import Namespace
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    from aiortc import RTCPeerConnection, RTCConfiguration
    from reachy_mini.apps import app as sdk_app
    from reachy_mini.io import protocol
    from dj_reachy_remix import main, moves
    from dj_reachy_remix.audio import head_wobbler
    from dj_reachy_remix.config import config
    from dj_reachy_remix.subscription_voice import SubscriptionRealtimeConnection
    from dj_reachy_remix.voice_media import MicrophoneTrack

    source, ready, state, action = sys.argv[1:]
    assert Path(main.__file__).resolve() == Path(source) / 'src/dj_reachy_remix/main.py'
    fixtures = runpy.run_path(str(Path(source) / 'tests/test_startup_resilience.py'))
    media = fixtures['OfflineMedia']()
    native_stopped = threading.Event()
    original_stop = media.stop_recording
    def stop_recording():
        if state == 'stopping':
            deadline = time.monotonic() + 5
            while not Path(ready).with_name('release').exists():
                assert time.monotonic() < deadline, 'fixture native stop was not released'
                time.sleep(.005)
        original_stop()
        native_stopped.set()
    media.stop_recording = stop_recording
    def close_media():
        assert native_stopped.is_set(), 'unsafe media.close before native stop completed'
        media.calls.append('close')
    media.close = close_media
    if hasattr(protocol, 'DaemonStatus'):
        # Target SDK1.12 returns this real protocol object, never a dict stand-in.
        status = protocol.DaemonStatus(robot_name='offline', state=protocol.DaemonState.RUNNING,
            wireless_version=False, desktop_app_daemon=False, simulation_enabled=False,
            mockup_sim_enabled=False, backend_status=None)
        assert isinstance(status, protocol.DaemonStatus)
    else:
        # Locked SDK1.2 ZenohClient.get_status returns Dict[str, Any].
        status = {'simulation_enabled': False}
    robot = SimpleNamespace(media=media, client=SimpleNamespace(
        get_status=lambda: status, disconnect=lambda: media.calls.append('disconnect')))
    class RobotContext:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return robot
        def __exit__(self, *args):
            media.calls.append('sdk_exit')
    # Before AppBase construction: never discover a daemon or construct an SDK.
    sdk_app.ReachyMini = RobotContext
    main.ReachyMiniConversationApp._check_daemon_on_localhost = staticmethod(lambda: False)
    main.ReachyMiniConversationApp._check_systemd_service_exists = staticmethod(lambda: False)
    main.ReachyMiniConversationApp.custom_app_url = 'http://127.0.0.1:0/'
    args = Namespace(no_camera=True, settings_only=False, gradio=False, debug=False,
        head_tracker=None, voice_backend='subscription')
    main.parse_args = lambda: (args, [])
    main._initialize_audio_only = lambda *args, **kwargs: None
    main._prepare_robot_motion = lambda *args: None
    main.handle_vision_stuff = lambda *args: (None, None, None)
    services = []
    def service(*args, **kwargs):
        result = Mock()
        services.append(result)
        return result
    moves.MovementManager = service
    head_wobbler.HeadWobbler = service
    config.OPENAI_API_KEY = config.ELEVENLABS_API_KEY = None
    connections = []
    async def enter(self):
        connections.append(self)
        self._pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._microphone = MicrophoneTrack()
        self._pc.addTrack(self._microphone)
        self._stats_task = asyncio.create_task(self._monitor_stats())
        self._audio_readers.append(asyncio.create_task(asyncio.Event().wait()))
        if state != 'connected':
            await self.close()
            raise ConnectionError('synthetic offline failure')
        self.ready.set()
        return self
    # Replace only connection establishment; close/__aexit__/handler stay real.
    SubscriptionRealtimeConnection.__aenter__ = enter
    import uvicorn
    servers = []
    class ObservedServer(uvicorn.Server):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            servers.append(self)
    uvicorn.Server = ObservedServer
    app = main.ReachyMiniConversationApp()
    baseline = set(threading.enumerate())
    observer_errors = []
    def observe():
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                stream_status = getattr(app.settings_app.state, 'dj_voice_status', None)
                if (stream_status and stream_status()['voice_state'] == state
                        and servers and all(server.started for server in servers)):
                    Path(ready).write_text('ready')
                    if action == 'stop':
                        app.stop()
                        app.stop()
                        time.sleep(.05)
                        Path(ready).with_name('release').write_text('release')
                    return
                time.sleep(.005)
            raise AssertionError('voice did not reach expected state')
        except BaseException as exc:
            observer_errors.append(exc)
    observer = threading.Thread(target=observe, name='offline-readiness', daemon=True)
    observer.start()
    faulthandler.dump_traceback_later(8)
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
    observer.join(1)
    assert not observer_errors, observer_errors
    assert Path(ready).exists()
    stream = app.settings_app.state.dj_voice_status.__self__
    assert stream.voice_status()['voice_state'] == 'stopped'
    assert all(task.done() for task in stream._tasks)
    assert stream._capture_io._finished and not stream._capture_io._thread.is_alive()
    assert stream._playback_io._finished and not stream._playback_io._thread.is_alive()
    assert stream.handler.connection is None
    assert not stream.handler._connected_event.is_set()
    assert stream.handler.diagnostics.closed
    assert len(connections) == 1  # subscription only; no retry/fallback
    for connection in connections:
        assert connection._closed
        assert connection._pc.connectionState == 'closed'
        assert connection._microphone.readyState == 'ended'
        assert connection._stats_task.done()
        assert all(task.done() for task in connection._audio_readers)
    assert media.calls.count('stop_recording') == 1, media.calls
    assert media.calls.count('stop_playing') == 1, media.calls
    assert media.calls.index('stop_recording') < media.calls.index('close')
    assert media.calls.index('stop_playing') < media.calls.index('close')
    assert media.calls[-1] == 'sdk_exit', media.calls
    for item in services:
        item.start.assert_called_once()
        item.stop.assert_called_once()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and set(threading.enumerate()) - baseline:
        time.sleep(.005)
    assert not (set(threading.enumerate()) - baseline), threading.enumerate()
    gc.collect()
    faulthandler.cancel_dump_traceback_later()
    print('CLEAN_MANAGED_SHUTDOWN', flush=True)
""")


@pytest.mark.parametrize("state", ["connected", "offline", "stopping"])
@pytest.mark.parametrize("action", ["sigint", "stop"])
def test_managed_shutdown_exits_and_releases_owned_workers(tmp_path: Path, state: str, action: str) -> None:
    """Reach real SDK wrapper/main/LocalStream/handler teardown, not a fake launch."""
    ready = tmp_path / "ready"
    source = Path(__file__).resolve().parents[1]
    with subprocess.Popen(
        [sys.executable, "-c", CHILD, str(source), str(ready), state, action],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": str(source / "src"), "PYTHONWARNINGS": "error::ResourceWarning"},
    ) as child:
        try:
            deadline = time.monotonic() + 12
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            if not ready.exists():
                child.kill()
                stdout, stderr = child.communicate()
                pytest.fail(f"offline managed launch never reached shutdown precondition\n{stdout}\n{stderr}")
            requested = time.monotonic()
            if action == "sigint":
                child.send_signal(signal.SIGINT)
                time.sleep(.05)
                ready.with_name("release").write_text("release")
            try:
                stdout, stderr = child.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                child.kill()
                stdout, stderr = child.communicate()
                pytest.fail(f"managed {state} {action} hung after shutdown request\n{stdout}\n{stderr}")
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
    print(f"managed_shutdown state={state} action={action} elapsed_s={time.monotonic() - requested:.3f}")
    assert child.returncode == 0, stdout + stderr
    assert "CLEAN_MANAGED_SHUTDOWN" in stdout
    for warning in ("ResourceWarning", "unraisable", "Exception ignored", "Task was destroyed", "was never awaited", "Exception in thread"):
        assert warning not in stderr, stderr
