"""Parent-owned behavioral acceptance tests. No providers or robot I/O."""

from __future__ import annotations
import os
import shutil
import asyncio
import subprocess
from pathlib import Path
from dataclasses import field, dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

import dj_reachy.profiles.dj_reachy.make_song_and_dance as music
from dj_reachy.tools.core_tools import ToolDependencies


@dataclass
class Dancer:
    """Record local motion calls without hardware."""

    running: bool = False
    feed_count: int = 0

    def start(self) -> None:
        """Record playback starting."""
        self.running = True

    def stop(self) -> None:
        """Record playback stopping."""
        self.running = False

    def feed(self, _data: NDArray[np.int16]) -> None:
        """Count consumed PCM frames."""
        self.feed_count += 1


@dataclass
class PlaybackHandler:
    """Keep observable queue and mute ownership without media devices."""

    owners: set[str]
    cleared: list[bool]
    output_queue: asyncio.Queue[object] = field(default_factory=asyncio.Queue)
    last_activity_time: float = 0

    def mute_output_audio(self, owner: str) -> None:
        """Acquire the named output mute."""
        self.owners.add(owner)

    def unmute_output_audio(self, owner: str) -> None:
        """Release only the named output mute."""
        self.owners.discard(owner)

    def _clear_queue(self) -> None:
        self.cleared.append(True)


@dataclass
class PlaybackActor:
    """Expose typed playback dependencies and their observations."""

    deps: ToolDependencies
    handler: PlaybackHandler
    dancer: Dancer
    owners: set[str]
    cleared: list[bool]


@pytest.fixture
def actor(monkeypatch: pytest.MonkeyPatch) -> PlaybackActor:
    """Intercept hardware only, leaving the asyncio decoder lifecycle real."""
    dancer = Dancer()
    owners: set[str] = set()
    cleared: list[bool] = []
    handler = PlaybackHandler(owners, cleared)
    deps = ToolDependencies(
        reachy_mini=MagicMock(), movement_manager=MagicMock(), openai_realtime_handler=handler
    )
    monkeypatch.setattr(music, "_get_music_dancer", lambda _deps: dancer)
    monkeypatch.setattr(music, "_generation_cancel", None)
    monkeypatch.setattr(music, "_active_decoder", None)
    monkeypatch.setattr(music, "_playback_pause", None)
    monkeypatch.setattr(music, "_playback_stop", None)
    return PlaybackActor(deps, handler, dancer, owners, cleared)


@pytest.fixture(scope="session")
def real_mp3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Encode a real three-second fixture without accessing microphones or speakers."""
    override = os.environ.get("FFMPEG_PATH")
    ffmpeg = shutil.which(override or "ffmpeg")
    if not ffmpeg:
        pytest.fail(
            f"Real decoder tests require an executable FFMPEG_PATH, got {override!r}"
            if override else "Real decoder tests require ffmpeg on PATH or an explicit FFMPEG_PATH"
        )
    root = tmp_path_factory.mktemp("decode-fixture")
    path = root / "silent-test.mp3"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3",
        "-codec:a",
        "libmp3lame",
        str(path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert path.stat().st_size > 1000
    return path


@pytest.mark.asyncio
async def test_stop_while_paused_finishes_without_external_task_cancellation(
    actor: PlaybackActor, real_mp3: Path
) -> None:
    """Stop paused playback, reap mpg123 and drain its pipe transports."""
    task = asyncio.create_task(music._play_mp3(str(real_mp3), actor.deps))
    try:
        await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        decoder = music._active_decoder
        assert decoder is not None and decoder.stdout is not None and decoder.stderr is not None
        await music.PauseMusic()(actor.deps)
        await asyncio.sleep(0.15)
        await music.StopMusic()(actor.deps)
        # Wait without injecting cancellation, which would conceal a stuck pause wait.
        done, _pending = await asyncio.wait({task}, timeout=1.2)
        assert task in done, "Stop stranded the paused playback task instead of waking and cleaning it"
        assert await task is True
        assert not actor.owners, "Stop left assistant-output muting owned by the dead song"
        assert not actor.dancer.running
        assert music._active_decoder is None
        assert decoder.stdout.at_eof(), "Stopped decoder left unread PCM and an open pipe transport"
        assert decoder.stderr.at_eof(), "Stopped decoder left stderr undrained"
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_pause_allows_conversation_and_resume_does_not_burst(actor: PlaybackActor, real_mp3: Path) -> None:
    """Allow conversation while paused and resume at bounded real-time cadence."""
    task = asyncio.create_task(music._play_mp3(str(real_mp3), actor.deps))
    try:
        await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        await music.PauseMusic()(actor.deps)
        await asyncio.sleep(0.25)
        assert not actor.owners, "Paused music still suppressed conversational speech"
        assert actor.handler.output_queue.empty(), "Pause left queued audio audible"
        assert actor.cleared, "Pause never flushed the physical backend"
        assert not actor.dancer.running, "Pause left the dancer running"
        await music.ResumeMusic()(actor.deps)
        await asyncio.sleep(0.11)
        assert actor.owners, "Resumed music didn't restore its mute ownership"
        assert actor.dancer.running
        assert 1 <= actor.handler.output_queue.qsize() <= 5, "Resume dumped a decoder backlog"
        await music.StopMusic()(actor.deps)
        await asyncio.wait_for(task, 1)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_generation_is_busy_during_replay_before_key_or_http(
    actor: PlaybackActor, real_mp3: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent a second paid generation while replay owns music playback."""

    def no_credentials(*_args: object) -> str:
        raise AssertionError("Busy generation must not load a key or contact the provider")

    monkeypatch.setattr(music, "_get_elevenlabs_key", no_credentials)
    task = asyncio.create_task(music._play_mp3(str(real_mp3), actor.deps))
    try:
        await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        result = await music.MakeSongAndDance()(actor.deps, prompt="fixture", music_length_ms=10000)
        assert result["status"] == "busy"
    finally:
        await music.StopMusic()(actor.deps)
        await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_stop_acknowledges_reaped_process_and_finished_motion_fade(
    actor: PlaybackActor, real_mp3: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return stop only after process cleanup and the bounded real motion fade."""
    from dj_reachy.audio.music_dancer import MusicDancer

    offsets = []
    dancer = MusicDancer(lambda *values: offsets.append(values), intensity=0.2)
    monkeypatch.setattr(music, "_get_music_dancer", lambda _deps: dancer)
    task = asyncio.create_task(music._play_mp3(str(real_mp3), actor.deps))
    try:
        await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        decoder = music._active_decoder
        assert decoder is not None
        start = asyncio.get_running_loop().time()
        await music.StopMusic()(actor.deps)
        elapsed = asyncio.get_running_loop().time() - start
        assert decoder.returncode is not None, "Stop acknowledged before reaping mpg123"
        assert not dancer.is_active(), "Stop acknowledged before motion reset"
        assert offsets[-1] == ((0.0,) * 6, (0.0, 0.0), 0.0)
        assert not actor.owners and task.done()
        assert elapsed < 1.5
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_decoder_stderr_cannot_block_pcm_or_cleanup(
    actor: PlaybackActor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuously drain stderr so decoder diagnostics cannot block PCM."""
    import sys

    spawn = asyncio.create_subprocess_exec

    async def noisy_child(
        *_args: str, stdout: int, stderr: int, pass_fds: tuple[int, ...] = ()
    ) -> asyncio.subprocess.Process:
        return await spawn(
            sys.executable,
            "-c",
            "import os; os.write(2, b'e' * 1048576); os.write(1, b'\\0' * 1920)",
            stdout=stdout, stderr=stderr, pass_fds=pass_fds,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", noisy_child)
    task = asyncio.create_task(music._play_mp3("fixture.mp3", actor.deps))
    try:
        done, _ = await asyncio.wait({task}, timeout=0.7)
        assert task in done, "Unread stderr blocked the decoder before PCM could reach playback"
        assert await task is False
        assert actor.dancer.feed_count == 1
        assert not actor.owners and music._active_decoder is None
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_second_playback_cannot_replace_decoder_and_old_cleanup_keeps_other_owner(
    actor: PlaybackActor, real_mp3: Path
) -> None:
    """Reject competing decoders and release only the terminating mute owner."""
    task = asyncio.create_task(music._play_mp3(str(real_mp3), actor.deps))
    try:
        await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        decoder = music._active_decoder
        assert decoder is not None and decoder.stdout is not None and decoder.stderr is not None
        actor.owners.add("another-audio-owner")
        with pytest.raises(RuntimeError, match="active"):
            await music._play_mp3(str(real_mp3), actor.deps)
        assert music._active_decoder is decoder
        await music.StopMusic()(actor.deps)
        assert actor.owners == {"another-audio-owner"}
        assert decoder.returncode is not None and decoder.stdout.at_eof() and decoder.stderr.at_eof()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_decoder_start_failure_propagates_without_mute_leak(
    actor: PlaybackActor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Propagate decoder startup failures without stranding music ownership."""

    async def fail_start(*_args: object, **_kwargs: object) -> asyncio.subprocess.Process:
        raise OSError("fixture-decoder-start-error")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_start)
    with pytest.raises(OSError, match="fixture-decoder-start-error"):
        await music._play_mp3("unused.mp3", actor.deps)
    assert not actor.owners and music._playback_pause is None


@pytest.mark.asyncio
async def test_pause_freezes_consumed_samples_and_resume_continues_in_order(
    actor: PlaybackActor, real_mp3: Path
) -> None:
    """Hold consumed samples during pause, then continue one frame at a time."""
    task = asyncio.create_task(music._play_mp3(str(real_mp3), actor.deps))
    try:
        await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        await music.PauseMusic()(actor.deps)
        feeds = actor.dancer.feed_count
        await asyncio.sleep(0.3)
        assert actor.dancer.feed_count == feeds
        await music.ResumeMusic()(actor.deps)
        await asyncio.wait_for(actor.handler.output_queue.get(), 1)
        assert actor.dancer.feed_count == feeds + 1
        await music.StopMusic()(actor.deps)
        assert task.done()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_kills_uncooperative_decoder_and_drains_both_pipes(
    actor: PlaybackActor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escalate SIGTERM to SIGKILL for a genuinely uncooperative child."""
    import sys

    spawn = asyncio.create_subprocess_exec

    async def stubborn_child(
        *_args: str, stdout: int, stderr: int, pass_fds: tuple[int, ...] = ()
    ) -> asyncio.subprocess.Process:
        return await spawn(
            sys.executable,
            "-c",
            "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "os.write(1, b'\\0' * 1920); time.sleep(60)",
            stdout=stdout, stderr=stderr, pass_fds=pass_fds,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", stubborn_child)
    task = asyncio.create_task(music._play_mp3("fixture.mp3", actor.deps))
    try:
        await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        proc = music._active_decoder
        assert proc is not None and proc.stdout is not None and proc.stderr is not None
        await asyncio.wait_for(music.StopMusic()(actor.deps), 1.6)
        assert await task is True
        assert proc.returncode == -9
        assert proc.stdout.at_eof() and proc.stderr.at_eof()
        assert not actor.owners
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_playback_task_reaps_process_and_preserves_chat(
    actor: PlaybackActor, real_mp3: Path
) -> None:
    """Reap canceled playback while retaining conversational queue entries."""
    task = asyncio.create_task(music._play_mp3(str(real_mp3), actor.deps))
    await asyncio.wait_for(actor.handler.output_queue.get(), 2)
    decoder = music._active_decoder
    assert decoder is not None and decoder.stdout is not None and decoder.stderr is not None
    chat = {"transcript": "keep this"}
    actor.handler.output_queue.put_nowait(chat)
    await music.PauseMusic()(actor.deps)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert actor.handler.output_queue.get_nowait() is chat
    assert actor.handler.output_queue.empty()
    assert decoder.returncode is not None and decoder.stdout.at_eof() and decoder.stderr.at_eof()
    assert not actor.owners


@pytest.mark.asyncio
async def test_repeated_shutdown_cancellation_cannot_release_owner_before_fade(
    actor: PlaybackActor, real_mp3: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep admission reserved until a canceled owner's fade thread exits."""
    from dj_reachy.audio.music_dancer import MusicDancer

    offsets = []
    dancer = MusicDancer(lambda *values: offsets.append(values), intensity=0.2)
    monkeypatch.setattr(music, "_get_music_dancer", lambda _deps: dancer)
    task = asyncio.create_task(music._play_mp3(str(real_mp3), actor.deps))
    try:
        await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        task.cancel()
        await asyncio.sleep(0.08)
        task.cancel()
        await asyncio.sleep(0.08)
        assert music._playback_pause is not None, (
            "Repeated cancellation admitted a replacement while old fade still ran"
        )
        assert not task.done()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1.0)
        assert music._playback_pause is None and not actor.owners
        assert offsets[-1] == ((0.0,) * 6, (0.0, 0.0), 0.0)
    finally:
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0.6)


@pytest.mark.asyncio
async def test_duplicate_controls_preserve_music_and_paused_conversation_audio(
    actor: PlaybackActor, real_mp3: Path
) -> None:
    """Treat repeated pause/resume as no-ops instead of flushing unrelated audio."""
    task = asyncio.create_task(music._play_mp3(str(real_mp3), actor.deps))
    try:
        frame = await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        actor.handler.output_queue.put_nowait(frame)
        await music.ResumeMusic()(actor.deps)
        assert actor.handler.output_queue.get_nowait() is frame, "Redundant resume discarded pending music"
        await music.PauseMusic()(actor.deps)
        actor.handler.output_queue.put_nowait(frame)
        await music.PauseMusic()(actor.deps)
        assert actor.handler.output_queue.get_nowait() is frame, "Redundant pause discarded conversation PCM"
    finally:
        await music.StopMusic()(actor.deps)
        await asyncio.gather(task, return_exceptions=True)
