"""Independent songs must not inherit speech queues, native writes or FIR state."""

import base64
import asyncio
import threading
from types import SimpleNamespace
from typing import cast
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from collections.abc import Callable, AsyncIterator

import numpy as np
import pytest
from fastrtc import audio_to_float32
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from dj_reachy_remix.console import LocalStream
from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler
from dj_reachy_remix.tools.core_tools import ToolDependencies
from dj_reachy_remix.audio.streaming_pcm import StreamingPCMConverter


class HandoverMedia:
    """Intercept only synchronous SDK writes/flushes; never open hardware."""

    def __init__(self, blocked: bool = False) -> None:
        """Retain an ordered trace and model the native pending playback buffer."""
        self.entered = threading.Event()
        self.release = threading.Event()
        if not blocked:
            self.release.set()
        self.calls: list[str] = []
        self.pending: list[NDArray[np.float32]] = []
        self.pushed: list[NDArray[np.float32]] = []
        self.owner_threads: set[int] = set()
        self.audio = SimpleNamespace(clear_player=self.clear_player)

    def push_audio_sample(self, frame: NDArray[np.float32]) -> None:
        """Hold the first real native-owner call until the test releases it."""
        self.entered.set()
        if not self.release.wait(2):
            raise TimeoutError("Offline SDK write was not released")
        self.owner_threads.add(threading.get_ident())
        self.pending.append(frame.copy())
        self.pushed.append(frame.copy())
        self.calls.append("push")

    def clear_player(self) -> None:
        """Flush only through the same synchronous owner as the writes."""
        self.owner_threads.add(threading.get_ident())
        self.pending.clear()
        self.calls.append("flush")

    def get_output_audio_samplerate(self) -> int:
        """Use the native 16k rate, with speech and song both entering at 24k."""
        return 16000

    def stop_recording(self) -> None:
        """No device was started."""

    stop_playing = stop_recording


def make_handler() -> OpenaiRealtimeHandler:
    """Create the real queue owner without connecting a voice provider."""
    return OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


def reference(pcm: NDArray[np.int16]) -> NDArray[np.float32]:
    """Convert the COMPLETE song independently, from sample zero through EOF."""
    converter = StreamingPCMConverter(24000, 16000)
    return np.concatenate(converter.push(audio_to_float32(pcm).reshape(-1)) + converter.finish())


async def until(predicate: Callable[[], bool]) -> None:
    """Bound scheduler waits and propagate unexpected playback failures later."""
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


@asynccontextmanager
async def pipeline(media: HandoverMedia) -> AsyncIterator[tuple[OpenaiRealtimeHandler, LocalStream]]:
    """Release native I/O, join owners and observe player exceptions on every exit."""
    app = make_handler()
    stream = LocalStream(app, cast(ReachyMini, SimpleNamespace(media=media)))
    player = asyncio.create_task(stream.play_loop())
    try:
        yield app, stream
    finally:
        media.release.set()
        stream._stop_event.set()
        player.cancel()
        try:
            await player
        except asyncio.CancelledError:
            pass
        finally:
            await stream._shutdown_audio()
            app.diagnostics.close()
            assert not stream._playback_io._thread.is_alive()
            assert not stream._capture_io._thread.is_alive()
            assert stream._emit_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_state", ["fir_history", "queued", "inflight"])
async def test_new_music_owner_starts_complete_independent_song(prior_state: str) -> None:
    """Real mute handover discards old PCM without losing the song's first samples."""
    media = HandoverMedia(blocked=prior_state == "inflight")
    speech = np.full(480, 24000, dtype=np.int16)
    song = np.random.default_rng(715).integers(-12000, 12000, 2400, dtype=np.int16)
    expected = reference(song)
    assert expected.size == 1600
    async with pipeline(media) as (app, stream):
        app.output_queue.put_nowait((24000, speech.reshape(1, -1)))
        if prior_state != "queued":
            await until(media.entered.is_set)
            if prior_state == "fir_history":
                # An empty/awaiting emit proves speech was submitted, not that
                # the converter's delayed FIR tail was drained or reset.
                await until(lambda: stream._emit_task is not None and not stream._emit_task.done())
                assert stream._pcm_converter is not None
            else:
                app.output_queue.put_nowait((24000, speech.reshape(1, -1)))
        app.mute_output_audio("song-new")
        app.output_queue.put_nowait((24000, song[:777].reshape(1, -1)))
        app.output_queue.put_nowait((24000, song[777:].reshape(1, -1)))
        if prior_state == "inflight":
            for _ in range(8):
                await asyncio.sleep(0)
            assert "flush" not in media.calls, "Flush raced the in-flight native write"
        media.release.set()
        assert await asyncio.wait_for(stream.drain_audio(), 2) is True
        actual = np.concatenate(media.pending)
        # The final EXACT song duration must include its beginning, not just a
        # convenient matching suffix after dropping contaminated samples.
        assert actual.size >= expected.size
        np.testing.assert_array_equal(actual[-expected.size:], expected)
        assert actual.size == expected.size, "Prior speech survived the new-owner flush"
        assert media.calls.count("flush") == 1
        if prior_state == "inflight":
            assert media.calls[:2] == ["push", "flush"]
        assert len(media.owner_threads) == 1
        assert threading.get_ident() not in media.owner_threads


@pytest.mark.asyncio
async def test_same_owner_retains_queued_song_and_fir_continuity() -> None:
    """Reasserting an active mute cannot discard or restart an ongoing song."""
    media = HandoverMedia()
    song = np.random.default_rng(89).integers(-10000, 10000, 2400, dtype=np.int16)
    async with pipeline(media) as (app, stream):
        app.mute_output_audio("song")
        app.output_queue.put_nowait((24000, song[:480].reshape(1, -1)))
        await until(lambda: stream._emit_task is not None and not stream._emit_task.done())
        epoch, converter, queue = stream._play_epoch, stream._pcm_converter, app.output_queue
        assert converter is not None
        app.output_queue.put_nowait((24000, song[480:].reshape(1, -1)))
        app.mute_output_audio("song")
        assert stream._play_epoch == epoch
        assert stream._pcm_converter is converter
        assert app.output_queue is queue and not queue.empty()
        assert await asyncio.wait_for(stream.drain_audio(), 2) is True
        np.testing.assert_array_equal(np.concatenate(media.pending), reference(song))
        assert media.calls.count("flush") == 1


def test_replacement_mute_rejects_stale_unmute_and_reacquires_after_release() -> None:
    """Ownership replacement clears once; only the current owner may release it."""
    app = make_handler()
    clears: list[str | None] = []
    app._clear_queue = lambda: clears.append(app._audio_mute_owner)
    try:
        app.mute_output_audio("old")
        app.mute_output_audio("replacement")
        app.mute_output_audio("replacement")
        assert len(clears) == 2
        app.unmute_output_audio("old")
        assert app._mute_output_audio and app._audio_mute_owner == "replacement"
        app.unmute_output_audio("replacement")
        assert not app._mute_output_audio and app._audio_mute_owner is None
        app.mute_output_audio("replacement")
        assert len(clears) == 3
    finally:
        app.diagnostics.close()


@pytest.mark.parametrize("callback", [None, object()])
def test_no_installed_clear_callback_preserves_transport_queue(
    callback: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-native transports must not acquire an invented queue-reset policy."""
    app = make_handler()
    try:
        queue = app.output_queue
        pcm = np.ones((1, 480), dtype=np.int16)
        queue.put_nowait((24000, pcm))
        # Cover both an absent callback and a present non-callable callback.
        app.mute_output_audio("no-callback")
        monkeypatch.setattr(app, "_clear_queue", callback)
        app.mute_output_audio("another-owner")
        assert app.output_queue is queue
        queued = queue.get_nowait()
        assert isinstance(queued, tuple)
        assert queued[1] is pcm
        app.unmute_output_audio("no-callback")
        assert app._mute_output_audio
        app.unmute_output_audio("another-owner")
        assert not app._mute_output_audio
    finally:
        app.diagnostics.close()


@pytest.mark.parametrize("previous_owner", [None, "old"])
def test_failed_clear_propagates_without_claiming_new_owner(previous_owner: str | None) -> None:
    """A failed synchronous fence must remain retryable, not fake a handover."""
    app = make_handler()
    try:
        if previous_owner is not None:
            app.mute_output_audio(previous_owner)

        def failed_clear() -> None:
            raise RuntimeError("offline flush rejected")

        app._clear_queue = failed_clear
        with pytest.raises(RuntimeError, match="offline flush rejected"):
            app.mute_output_audio("new")
        assert app._audio_mute_owner == previous_owner
        assert app._mute_output_audio is (previous_owner is not None)
        cleared: list[bool] = []
        app._clear_queue = lambda: cleared.append(True)
        app.mute_output_audio("new")
        assert cleared == [True]
        assert app._audio_mute_owner == "new" and app._mute_output_audio
    finally:
        app.diagnostics.close()


@pytest.mark.asyncio
async def test_already_drained_ready_notice_keeps_its_complete_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ready-notice fence finishes speech before the new-owner clear."""
    from dj_reachy_remix.openai_realtime import voice_config

    monkeypatch.setattr(voice_config, "VOICE_BACKEND", "subscription")
    media = HandoverMedia()
    speech = np.random.default_rng(91).integers(-12000, 12000, 2400, dtype=np.int16)
    song = np.random.default_rng(92).integers(-12000, 12000, 2400, dtype=np.int16)
    announced = asyncio.Event()

    async def send_context(*args: object) -> None:
        announced.set()

    async with pipeline(media) as (app, stream):
        stream._output_tail_delay = 0.01
        app._drain_queue = stream.drain_audio
        app.connection = SimpleNamespace(send_context=send_context)
        notice = asyncio.create_task(app.notify_song_ready())
        try:
            await asyncio.wait_for(announced.wait(), 1)
            await app._handle_event(SimpleNamespace(
                type="response.audio.delta", delta=base64.b64encode(speech.tobytes()).decode("ascii")
            ))
            await app._handle_event(SimpleNamespace(
                type="turn.done", turn={"role": "assistant", "transcript": "Your song is ready."}
            ))
            await asyncio.wait_for(notice, 2)
            assert stream._pcm_converter is None
            assert "flush" not in media.calls
            np.testing.assert_array_equal(np.concatenate(media.pushed), reference(speech))
            speech_packets = len(media.pushed)
            app.mute_output_audio("song-after-ready")
            app.output_queue.put_nowait((24000, song.reshape(1, -1)))
            assert await asyncio.wait_for(stream.drain_audio(), 2) is True
            # Capture is append-only even though pending native PCM was flushed:
            # both complete independent streams retain their own final samples.
            np.testing.assert_array_equal(np.concatenate(media.pushed[:speech_packets]), reference(speech))
            np.testing.assert_array_equal(np.concatenate(media.pushed[speech_packets:]), reference(song))
            assert media.calls.count("flush") == 1
        finally:
            notice.cancel()
            await asyncio.gather(notice, return_exceptions=True)


@pytest.mark.asyncio
async def test_fixture_failure_releases_inflight_sdk_write_and_joins_workers() -> None:
    """A failing assertion/body cannot leak the deterministic blocked native owner."""
    media = HandoverMedia(blocked=True)
    with pytest.raises(RuntimeError, match="offline body failed"):
        async with pipeline(media) as (app, stream):
            app.output_queue.put_nowait((24000, np.full((1, 480), 1000, dtype=np.int16)))
            await until(media.entered.is_set)
            raise RuntimeError("offline body failed")
    assert media.release.is_set()
    assert not stream._playback_io._thread.is_alive()
