"""Privacy-safe aggregate diagnostics exercised without devices or providers."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest


def records(path: Path) -> list[dict[str, object]]:
    """Decode the private JSONL diagnostic file."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_audio_metrics_rate_limit_changes_and_run_summary(tmp_path: Path) -> None:
    """Keep totals even while suppressing repeated frames, not silent transitions."""
    from dj_reachy_remix.audio_diagnostics import AudioDiagnostics

    now = [10.0]
    trace = AudioDiagnostics(home=tmp_path, clock=lambda: now[0])
    trace.start()
    raw = np.column_stack((np.zeros(480, dtype=np.float32), np.full(480, 0.25, dtype=np.float32)))
    for _ in range(20):
        trace.audio("capture", raw, 48000)
    trace.audio("capture", np.zeros_like(raw), 48000)
    trace.audio("capture", raw.astype(np.float64), 48000)
    now[0] += 5
    trace.audio("capture", raw, 48000)
    trace.close()
    events = records(tmp_path / "logs" / "audio-trace.jsonl")
    audio = [event for event in events if event["event"] == "audio"]
    assert len(audio) == 4
    assert audio[0]["shape"] == [480, 2]
    assert audio[0]["dtype"] == "float32"
    assert audio[0]["rate"] == 48000
    assert audio[0]["channels"] == [
        {"peak": 0.0, "rms": 0.0, "nonzero": 0},
        {"peak": 0.25, "rms": 0.25, "nonzero": 480},
    ]
    assert audio[1]["warning"] == "silence_change"
    assert audio[2]["warning"] == "format_change"
    assert audio[-1]["frames_total"] == 23
    assert len({event["run_id"] for event in events}) == 1
    assert events[-1]["event"] == "run_summary"
    assert events[-1]["last_audio_boundary"] == "capture"
    assert events[-1]["counts"] == {"audio.capture": 23}
    assert trace.closed
    assert not trace.owned_handlers


def test_private_rotating_file_accepts_only_diagnostics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bound persistence, restrict permissions, and exclude arbitrary provider fields."""
    import logging

    from dj_reachy_remix.audio_diagnostics import AudioDiagnostics

    trace = AudioDiagnostics(home=tmp_path, max_bytes=2048, backups=2)
    trace.start()
    secret = "SECRET prompt transcript bearer sdp provider-error-body"
    logging.getLogger("dj_reachy_remix.subscription_voice").warning(secret)
    for _ in range(100):
        trace.provider_event(secret)
        trace.lifecycle("tool", "started", name=secret)
    trace.close()
    paths = list((tmp_path / "logs").iterdir())
    assert len(paths) <= 3
    assert (tmp_path / "logs").stat().st_mode & 0o777 == 0o700
    for path in paths:
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.stat().st_size <= 2048
        assert secret not in path.read_text()
    output = capsys.readouterr().out
    assert '"run_id"' in output
    assert secret not in output
    assert secret not in str(trace.counts)


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", [16000, 48000])
@pytest.mark.parametrize("first_channel", [0.0, 0.25])
async def test_synthetic_console_handler_track_path_keeps_channel_selection(
    tmp_path: Path,
    sample_rate: int,
    first_channel: float,
) -> None:
    """Locate the first-channel silence without changing existing selection semantics."""
    from typing import cast
    from unittest.mock import MagicMock

    from numpy.typing import NDArray

    from reachy_mini import ReachyMini
    from dj_reachy_remix.console import LocalStream
    from dj_reachy_remix.voice_media import MicrophoneTrack
    from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy_remix.tools.core_tools import ToolDependencies
    from dj_reachy_remix.subscription_voice import SubscriptionRealtimeAdapter

    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    app.diagnostics.home = tmp_path
    app.diagnostics.start()
    raw = np.column_stack((np.full(960, first_channel, dtype=np.float32), np.full(960, 0.25, dtype=np.float32)))

    class GStreamerAudio:
        def __repr__(self) -> str:
            raise AssertionError("backend configuration must not be formatted")

    class Media:
        audio = GStreamerAudio()

        def get_input_channels(self) -> int:
            return 2

        def get_input_audio_samplerate(self) -> int:
            return sample_rate

        def get_audio_sample(self) -> NDArray[np.float32] | None:
            stream._stop_event.set()
            return raw

        def stop_recording(self) -> None:
            pass

        def stop_playing(self) -> None:
            pass

    robot = MagicMock(spec=ReachyMini)
    robot.media = Media()
    stream = LocalStream(app, cast(ReachyMini, robot))
    connection = SubscriptionRealtimeAdapter(diagnostics=app.diagnostics).connect()
    connection._microphone = MicrophoneTrack(diagnostics=app.diagnostics)
    app.connection = connection
    try:
        await stream.record_loop()
        frame = await connection._microphone.recv()
        assert bool(np.count_nonzero(frame.to_ndarray())) == (first_channel != 0)
    finally:
        stream._closing = True
        await stream._shutdown_audio()
        await app.shutdown()
    assert not stream._capture_io._thread.is_alive()
    assert not stream._playback_io._thread.is_alive()
    events = records(tmp_path / "logs" / "audio-trace.jsonl")
    configuration = [event for event in events if event["event"] == "capture_configuration"]
    assert len(configuration) == 1
    assert configuration[0]["rate"] == sample_rate
    assert configuration[0]["channels"] == 2
    assert configuration[0]["backend"] == "GStreamerAudio"
    audio = {event["boundary"]: event for event in events if event["event"] == "audio"}
    assert set(audio) == {"capture", "selected", "resampled", "encoded", "append", "feed", "track"}
    assert audio["capture"]["dtype"] == "float32"
    assert audio["capture"]["channels"] == [
        {"peak": first_channel, "rms": first_channel, "nonzero": 960 if first_channel else 0},
        {"peak": 0.25, "rms": 0.25, "nonzero": 960},
    ]
    assert audio["selected"]["rate"] == sample_rate
    assert audio["resampled"]["rate"] == 24000
    converted_samples = int(raw.shape[0] * 24000 / sample_rate)
    assert audio["resampled"]["shape"] == [converted_samples]
    assert audio["encoded"]["dtype"] == "int16"
    assert audio["track"]["rate"] == 48000
    summary = events[-1]
    assert summary["event"] == "run_summary"
    assert summary["last_audio_boundary"] == "track"
    counts = summary["counts"]
    assert isinstance(counts, dict)
    assert counts["flow.append.input_bytes"] == converted_samples * 2
    assert counts["flow.feed.input_bytes"] == converted_samples * 2
    assert (
        counts["flow.track.underrun_bytes"] > 0 if sample_rate == 48000 else counts["flow.track.underrun_bytes"] == 0
    )
    assert summary["furthest_nonzero_boundary"] == ("track" if first_channel else "capture")
    assert app.diagnostics.closed
    assert not app.diagnostics.owned_handlers


@pytest.mark.asyncio
async def test_drop_reasons_and_track_backlog_are_counted(tmp_path: Path) -> None:
    """Expose input losses and bounded track queues without altering PCM handling."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from dj_reachy_remix.voice_media import MicrophoneTrack
    from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy_remix.tools.core_tools import ToolDependencies
    from dj_reachy_remix.subscription_voice import SubscriptionRealtimeAdapter

    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    app.diagnostics.home = tmp_path
    app.diagnostics.start()
    pcm = np.full(480, 10000, dtype=np.int16)
    await app.receive((24000, pcm))

    async def append(*, audio: str) -> None:
        raise ConnectionError("SENSITIVE exception body token")

    app.connection = SimpleNamespace(input_audio_buffer=SimpleNamespace(append=append))
    await app.receive((24000, pcm))
    connection = SubscriptionRealtimeAdapter(diagnostics=app.diagnostics).connect()
    app.connection = connection
    await connection._append_audio(audio="not decoded when no microphone exists")
    track = MicrophoneTrack(diagnostics=app.diagnostics)
    track.feed(np.full(100000, 10000, dtype=np.int16).tobytes())
    track.feed(np.full(48000, 10000, dtype=np.int16).tobytes())
    assert len(track.backlog) <= 192000
    frame = await track.recv()
    assert np.abs(frame.to_ndarray()).max() > 9000
    track.stop()
    track.feed(pcm.tobytes())
    await app.shutdown()
    events = records(tmp_path / "logs" / "audio-trace.jsonl")
    counts = app.diagnostics.counts
    assert counts["drop.no_connection"] == 1
    assert counts["drop.append_failed"] == 1
    assert counts["drop.no_microphone"] == 1
    assert counts["drop.track_ended"] == 1
    assert counts["drop.feed_input_trimmed"] == 1
    assert counts["drop.feed_backlog_trimmed"] == 1
    assert counts["flow.feed.dropped_bytes"] > 0
    assert "SENSITIVE" not in json.dumps(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
async def test_real_handler_planner_tool_lifecycle_never_logs_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    """Exercise actual delegation/planning/dispatch with secret-bearing injected fixtures."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import dj_reachy_remix.openai_realtime as realtime
    from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy_remix.tools.core_tools import ToolDependencies

    secret = "PRIVATE transcript prompt payload credentials sdp header failure"
    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    app.diagnostics.home = tmp_path
    app.diagnostics.start()
    entered = asyncio.Event()

    async def plan(transcript: str, history: list[dict[str, object]]) -> dict[str, object]:
        assert transcript == secret
        if history:
            return {"calls": [], "output": [], "text": secret}
        return {
            "calls": [{"name": "play_saved_song", "arguments": {"filename": secret + ".mp3"}, "call_id": secret}],
            "output": [],
        }

    async def context(text: str, channel: str, identity: str | None = None) -> None:
        pass

    async def dispatch(name: str, arguments: str, deps: ToolDependencies) -> dict[str, str]:
        if name == "stop_music":
            return {"status": "stopped"}
        entered.set()
        if outcome == "failed":
            raise RuntimeError(secret)
        if outcome == "cancelled":
            await asyncio.Event().wait()
        return {"status": "completed", "payload": secret}

    async def close() -> None:
        pass

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    app.connection = SimpleNamespace(plan=plan, send_context=context, close=close)
    await app._handle_event(SimpleNamespace(type="input_transcript.added", item={"text": secret}))
    await app._handle_event(SimpleNamespace(type="error", error={"message": secret}))
    await app._handle_event(
        SimpleNamespace(
            type="delegation.created",
            item={"type": "delegation", "target": "client", "id": secret},
            transcript_context=secret,
            turn_id=1,
        )
    )
    tasks = list(app._subscription_tasks.values())
    await asyncio.wait_for(entered.wait(), 1)
    if outcome == "cancelled":
        tasks[0].cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await app.shutdown()
    counts = app.diagnostics.counts
    assert counts["provider.handled.input_transcript"] == 1
    assert counts["provider.handled.error"] == 1
    assert counts["planner.started"] >= 1
    assert counts["planner.completed"] >= 1
    assert counts["tool.started.play_saved_song"] == 1
    assert counts[f"tool.{outcome}.play_saved_song"] == 1
    assert counts["delegation.started"] == 1
    assert counts["delegation.cancelled" if outcome == "cancelled" else "delegation.completed"] == 1
    text = (tmp_path / "logs" / "audio-trace.jsonl").read_text()
    assert secret not in text
    assert not app._subscription_tasks
    assert app.diagnostics.closed


@pytest.mark.asyncio
async def test_close_failure_still_releases_owned_diagnostic_task_and_handlers(tmp_path: Path) -> None:
    """A close-channel error must not leak the monitor or private file descriptor."""
    import asyncio

    from aiortc import RTCConfiguration, RTCPeerConnection

    from dj_reachy_remix.voice_media import MicrophoneTrack
    from dj_reachy_remix.subscription_voice import SubscriptionRealtimeAdapter

    class BrokenChannel:
        readyState = "open"

        def send(self, message: str) -> None:
            raise ConnectionError("PRIVATE provider close body")

    connection = SubscriptionRealtimeAdapter().connect()
    connection.diagnostics.home = tmp_path
    connection.diagnostics.start()
    connection._pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    connection._channel = BrokenChannel()
    connection._microphone = MicrophoneTrack(diagnostics=connection.diagnostics)
    connection._stats_task = asyncio.create_task(connection._monitor_stats())
    await asyncio.sleep(0)
    try:
        await connection.close()
    except ConnectionError:
        pass
    assert connection._stats_task.done()
    assert connection._pc.connectionState == "closed"
    assert connection._microphone.readyState == "ended"
    assert connection.diagnostics.closed
    assert not connection.diagnostics.owned_handlers
    assert "PRIVATE" not in (tmp_path / "logs" / "audio-trace.jsonl").read_text()


def test_summary_separates_nonzero_progress_from_paced_silence(tmp_path: Path) -> None:
    """Track-generated silence must not hide the boundary where real input energy vanished."""
    from dj_reachy_remix.audio_diagnostics import AudioDiagnostics

    trace = AudioDiagnostics(home=tmp_path)
    trace.start()
    trace.audio("capture", np.full((480, 2), 0.25, dtype=np.float32), 16000)
    trace.audio("selected", np.zeros(480, dtype=np.float32), 16000)
    for _ in range(100):
        trace.audio("track", np.zeros((1, 960), dtype=np.int16), 48000)
    trace.close()
    summary = records(tmp_path / "logs" / "audio-trace.jsonl")[-1]
    assert summary["furthest_audio_boundary"] == "track"
    assert summary["furthest_nonzero_boundary"] == "capture"
    assert summary["nonzero_frames"] == {"capture": 1, "selected": 0, "track": 0}
    assert summary["samples_total"] == {"capture": 480 * 2, "selected": 480, "track": 100 * 960}


def test_default_diagnostic_root_matches_app_xdg_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unset DJ_REACHY_REMIX_HOME diagnostics alongside the rest of the app's state."""
    from dj_reachy_remix.audio_diagnostics import AudioDiagnostics

    monkeypatch.delenv("DJ_REACHY_REMIX_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    trace = AudioDiagnostics()
    trace.start()
    trace.close()
    assert (tmp_path / "dj-reachy-remix" / "logs" / "audio-trace.jsonl").exists()


def test_symlinked_diagnostic_targets_are_refused(tmp_path: Path) -> None:
    """Never follow log-file or log-directory symlinks into another private state file."""
    from dj_reachy_remix.audio_diagnostics import AudioDiagnostics

    victim = tmp_path / "private-state"
    victim.write_text("DO NOT TOUCH")
    victim.chmod(0o640)
    directory = tmp_path / "app" / "logs"
    directory.mkdir(parents=True)
    (directory / "audio-trace.jsonl").symlink_to(victim)
    trace = AudioDiagnostics(home=tmp_path / "app")
    trace.start()
    trace.close()
    assert trace.counts["diagnostic_errors"] == 1
    assert victim.read_text() == "DO NOT TOUCH"
    assert victim.stat().st_mode & 0o777 == 0o640
    assert not trace.owned_handlers
    second = tmp_path / "second"
    second.mkdir()
    (second / "logs").symlink_to(directory, target_is_directory=True)
    trace = AudioDiagnostics(home=second)
    trace.start()
    trace.close()
    assert trace.counts["diagnostic_errors"] == 1


def test_stats_and_alternating_audio_are_bounded_without_losing_totals(tmp_path: Path) -> None:
    """Rate limiting must not let rapid silence/format changes flood either sink."""
    from dj_reachy_remix.audio_diagnostics import AudioDiagnostics

    now = [0.0]
    trace = AudioDiagnostics(home=tmp_path, clock=lambda: now[0])
    trace.start()
    for index in range(100):
        trace.audio("capture", np.full(480, index % 2, dtype=np.int16 if index % 2 else np.float32), 24000)
        trace.webrtc(
            packets_sent=index,
            bytes_sent=index * 10,
            connection_state="connected",
            ice_state="completed",
            signaling_state="stable",
            channel_state="open",
        )
    now[0] = 5.0
    trace.webrtc(
        packets_sent=100,
        bytes_sent=1000,
        connection_state="connected",
        ice_state="completed",
        signaling_state="stable",
        channel_state="open",
    )
    trace.close()
    events = records(tmp_path / "logs" / "audio-trace.jsonl")
    assert len([event for event in events if event["event"] == "audio"]) <= 3
    assert len([event for event in events if event["event"] == "webrtc_stats"]) == 2
    assert trace.counts["audio.capture"] == 100
    assert trace.counts["webrtc.stats_samples"] == 101
    assert events[-1]["nonzero_frames"] == {"capture": 50}


def test_diagnostics_io_failure_does_not_escape(tmp_path: Path) -> None:
    """An unwritable diagnostic root must not break the audio path."""
    from dj_reachy_remix.audio_diagnostics import AudioDiagnostics

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    trace = AudioDiagnostics(home=blocked)
    trace.start()
    trace.audio("capture", np.zeros((4, 2), dtype=np.int16), 48000)
    trace.close()
    trace.close()
    assert trace.closed
    assert not trace.owned_handlers


@pytest.mark.asyncio
async def test_conversion_failure_records_the_failed_boundary_without_error_payload(tmp_path: Path) -> None:
    """Keep a known conversion exception visible without changing the audio contract."""
    from unittest.mock import MagicMock

    from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy_remix.tools.core_tools import ToolDependencies
    from dj_reachy_remix.subscription_voice import SubscriptionRealtimeAdapter

    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    app.diagnostics.home = tmp_path
    app.diagnostics.start()
    app.connection = SubscriptionRealtimeAdapter(diagnostics=app.diagnostics).connect()
    try:
        with pytest.raises(TypeError):
            await app.receive((48000, np.full(960, 8192, dtype=np.int16)))
    finally:
        await app.shutdown()
    assert app.diagnostics.counts.get("drop.conversion_failed") == 1
    summary = records(tmp_path / "logs" / "audio-trace.jsonl")[-1]
    assert summary["furthest_audio_boundary"] == "resampled"
    assert "audio.encoded" not in app.diagnostics.counts
