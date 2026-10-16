"""Exercise real native loops with controlled, synchronous SDK boundaries only."""

import asyncio
import threading
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from collections.abc import Callable, Coroutine

import numpy as np
import pytest
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from dj_reachy.console import LocalStream
from dj_reachy.openai_realtime import OpenaiRealtimeHandler


class BlockingMedia:
    """A deterministic SDK boundary: release only after loop progress is observed."""

    def __init__(self, lane: str) -> None:
        """Do not touch a device or start an SDK backend."""
        self.lane = lane
        self.entered = threading.Event()
        self.release = threading.Event()
        self.expired = False
        self.calls: list[str] = []
        self.pushed: list[NDArray[np.float32]] = []
        self.push_threads: set[int] = set()
        self.capture_threads: set[int] = set()
        self.audio = SimpleNamespace(clear_player=lambda: self.calls.append("flush"))

    def _block(self, lane: str) -> None:
        if lane == self.lane:
            self.entered.set()
            if not self.release.wait(0.5):
                self.expired = True

    def get_audio_sample(self) -> NDArray[np.float32]:
        """Model synchronous appsink try_pull_sample, with no synthetic async sleep."""
        self.capture_threads.add(threading.get_ident())
        self._block("capture")
        return np.ones((8, 2), dtype=np.float32)

    def push_audio_sample(self, frame: NDArray[np.float32]) -> None:
        """Model synchronous appsrc push_buffer."""
        self.push_threads.add(threading.get_ident())
        self._block("playback")
        self.pushed.append(frame.copy())
        self.calls.append("push")

    def get_input_audio_samplerate(self) -> int:
        """Return the fixture format."""
        return 24000

    get_output_audio_samplerate = get_input_audio_samplerate

    def get_input_channels(self) -> int:
        """Return stereo input."""
        return 2

    def start_recording(self) -> None:
        """Record lifecycle ordering."""
        self.calls.append("start_recording")

    def start_playing(self) -> None:
        """Record lifecycle ordering."""
        self.calls.append("start_playing")

    def stop_recording(self) -> None:
        """Detect stopping a live SDK operation."""
        assert not self.entered.is_set() or self.release.is_set()
        self.calls.append("stop_recording")

    def stop_playing(self) -> None:
        """Detect stopping a live SDK operation."""
        assert not self.entered.is_set() or self.release.is_set()
        self.calls.append("stop_playing")


def make_stream(media: BlockingMedia) -> LocalStream:
    """Use the actual LocalStream with an offline handler boundary."""
    queue: asyncio.Queue[tuple[int, NDArray[np.int16]]] = asyncio.Queue()
    queue.put_nowait((24000, np.arange(8, dtype=np.int16)))
    handler = SimpleNamespace(output_queue=queue, diagnostics=Mock(), deps=Mock())

    async def emit() -> tuple[int, NDArray[np.int16]]:
        current: asyncio.Queue[tuple[int, NDArray[np.int16]]] = handler.output_queue
        return await current.get()

    async def receive(frame: tuple[int, NDArray[np.int16 | np.float32]]) -> None:
        return None

    async def start_up() -> None:
        await asyncio.Event().wait()

    async def shutdown() -> None:
        handler.shutdown_complete = True

    handler.emit = emit
    handler.receive = receive
    handler.start_up = start_up
    handler.shutdown = shutdown
    handler.shutdown_complete = False
    return LocalStream(cast(OpenaiRealtimeHandler, handler), cast(ReachyMini, SimpleNamespace(media=media)))


def native_pipeline(mixer: SimpleNamespace) -> SimpleNamespace:
    """Preserve GI's next/result-value interface, which is not a Python iterator."""
    def iterate_recurse() -> SimpleNamespace:
        items = iter([
            (SimpleNamespace(value_nick="ok"), mixer),
            (SimpleNamespace(value_nick="done"), None),
        ])
        return SimpleNamespace(next=lambda: next(items), resync=lambda: None)

    return SimpleNamespace(iterate_recurse=iterate_recurse)


@pytest.mark.asyncio
async def test_native_stream_configures_one_coherent_jitter_window() -> None:
    """The app's native launch must buffer late PCM without resetting it early."""
    values = {"latency": 0}
    mixer = SimpleNamespace(
        get_factory=lambda: SimpleNamespace(get_name=lambda: "audiomixer"),
        get_property=lambda name: values[name],
        set_property=lambda name, value: values.__setitem__(name, value),
    )
    backend = SimpleNamespace(
        _pipeline=native_pipeline(mixer),
        GAP_RESET_NS=200_000_000,
    )
    media = BlockingMedia("")
    media.audio = backend
    stream = make_stream(media)
    try:
        assert values["latency"] == 320_000_000
        assert backend.GAP_RESET_NS == 520_000_000
    finally:
        await stream._shutdown_audio()


@pytest.mark.asyncio
async def test_native_stream_keeps_an_existing_wider_buffer() -> None:
    """Reattaching a stream must not shrink or repeatedly extend its schedule."""
    values = {"latency": 640_000_000}
    mixer = SimpleNamespace(
        get_factory=lambda: SimpleNamespace(get_name=lambda: "audiomixer"),
        get_property=lambda name: values[name],
        set_property=lambda name, value: values.__setitem__(name, value),
    )
    backend = SimpleNamespace(
        _pipeline=native_pipeline(mixer),
        GAP_RESET_NS=840_000_000,
    )
    media = BlockingMedia("")
    media.audio = backend
    first = make_stream(media)
    second = make_stream(media)
    try:
        assert values["latency"] == 640_000_000
        assert backend.GAP_RESET_NS == 840_000_000
        assert first._output_tail_delay == pytest.approx(0.82)
        assert second._output_tail_delay == first._output_tail_delay
    finally:
        await first._shutdown_audio()
        await second._shutdown_audio()


@pytest.mark.asyncio
async def test_completion_fence_waits_for_dequeued_native_push_and_tail() -> None:
    """An empty queue isn't completion while its last native write is in flight."""
    media = BlockingMedia("playback")
    stream = make_stream(media)
    stream._output_tail_delay = 0.08
    player = asyncio.create_task(stream.play_loop())
    completion: asyncio.Task[bool] | None = None
    try:
        await progress_while_blocked(media)
        assert stream.handler.output_queue.empty()
        drain = cast(Callable[[], Coroutine[object, object, bool]] | None, getattr(stream, "drain_audio", None))
        assert callable(drain), "Native playback has no ordered completion fence"
        completion = asyncio.create_task(drain())
        done, _ = await asyncio.wait({completion}, timeout=0.01)
        assert not done
        media.release.set()
        done, _ = await asyncio.wait({completion}, timeout=0.02)
        assert not done, "Completion ignored the native scheduled tail"
        assert await asyncio.wait_for(completion, 0.5) is True
        assert len(media.pushed) == 1
    finally:
        stream._stop_event.set()
        media.release.set()
        player.cancel()
        if completion is not None:
            completion.cancel()
            await asyncio.gather(completion, return_exceptions=True)
        await asyncio.gather(player, return_exceptions=True)
        await stream._shutdown_audio()


@pytest.mark.asyncio
async def test_flush_invalidates_native_completion_without_tail_wait() -> None:
    """A native tail waiter cannot delay flush or consume replacement PCM."""
    media = BlockingMedia("")
    stream = make_stream(media)
    stream._output_tail_delay = 2.0
    player = asyncio.create_task(stream.play_loop())
    completion = asyncio.create_task(stream.drain_audio())
    try:
        done, _ = await asyncio.wait({completion}, timeout=0.02)
        assert not done
        stream.clear_audio_queue()
        assert await asyncio.wait_for(completion, 0.2) is False
        assert stream._flush_task is not None
        await stream._flush_task
        stream.handler.output_queue.put_nowait((24000, np.full(8, 42, dtype=np.int16)))
        for _ in range(1000):
            if len(media.pushed) == 2:
                break
            await asyncio.sleep(0.001)
        assert len(media.pushed) == 2
        assert media.calls[-2:] == ["flush", "push"]
    finally:
        stream._stop_event.set()
        player.cancel()
        completion.cancel()
        await asyncio.gather(player, completion, return_exceptions=True)
        await stream._shutdown_audio()


async def progress_while_blocked(media: BlockingMedia) -> None:
    """Require repeated event-loop control progress before releasing the SDK call."""
    for _ in range(10000):
        if media.entered.is_set():
            break
        await asyncio.sleep(0.001)
    assert media.entered.is_set()
    for _ in range(8):
        await asyncio.sleep(0)
        assert not media.release.is_set()
        assert not media.expired, "SDK call monopolized event loop until watchdog release"


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["capture", "playback"])
async def test_controls_progress_during_synchronous_native_io(lane: str) -> None:
    """Control ticks run while either actual native loop is inside synchronous I/O."""
    media = BlockingMedia(lane)
    stream = make_stream(media)
    task = asyncio.create_task(stream.record_loop() if lane == "capture" else stream.play_loop())
    try:
        await progress_while_blocked(media)
    finally:
        stream._stop_event.set()
        media.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # Later lifecycle slices own and join the workers as well.
        cleanup = getattr(stream, "_shutdown_audio", None)
        if cleanup is not None:
            await cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["capture", "playback"])
async def test_launch_close_drains_inflight_io_before_media_stop(lane: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real launch/close retains SDK ownership after asyncio cancellation."""
    from dj_reachy.config import config

    monkeypatch.setattr(config, "VOICE_BACKEND", "subscription")
    media = BlockingMedia(lane)
    stream = make_stream(media)
    launched = threading.Thread(target=stream.launch, daemon=True)
    launched.start()
    try:
        await progress_while_blocked(media)
        stream.close()
        for _ in range(20):
            await asyncio.sleep(0.001)
        assert launched.is_alive(), "launch returned before its in-flight SDK call completed"
        assert not any(call.startswith("stop_") for call in media.calls)
    finally:
        media.release.set()
        for _ in range(1000):
            if not launched.is_alive():
                break
            await asyncio.sleep(0.001)
        launched.join(timeout=0)
    assert not launched.is_alive()
    assert media.calls.count("stop_recording") == 1
    assert media.calls.count("stop_playing") == 1
    assert not stream._capture_io._thread.is_alive()
    assert not stream._playback_io._thread.is_alive()
    assert cast(SimpleNamespace, stream.handler).shutdown_complete


@pytest.mark.asyncio
async def test_pause_flush_fences_inflight_push_and_discards_stale_queue() -> None:
    """A blocked push finishes BEFORE flush; only fresh frames may follow it."""
    media = BlockingMedia("playback")
    stream = make_stream(media)
    task = asyncio.create_task(stream.play_loop())
    try:
        await progress_while_blocked(media)
        stream.handler.output_queue.put_nowait((24000, np.ones(8, dtype=np.int16)))
        stream.clear_audio_queue()
        for _ in range(8):
            await asyncio.sleep(0)
        assert "flush" not in media.calls, "flush raced a still-owned push"
        media.release.set()
        for _ in range(1000):
            if "flush" in media.calls:
                break
            await asyncio.sleep(0.001)
        assert media.calls == ["push", "flush"]
        # Exercise queue replacement while emit() is already waiting.
        await asyncio.sleep(0.01)
        stream.clear_audio_queue()
        stream.handler.output_queue.put_nowait((24000, np.full(8, 2, dtype=np.int16)))
        for _ in range(1000):
            if media.calls.count("push") == 2:
                break
            await asyncio.sleep(0.001)
        assert media.calls == ["push", "flush", "flush", "push"]
    finally:
        stream._stop_event.set()
        media.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await stream._shutdown_audio()


@pytest.mark.asyncio
async def test_foreign_thread_flush_stays_on_playback_owner() -> None:
    """A synchronous external caller cannot bypass the live I/O fence."""
    media = BlockingMedia("playback")
    stream = make_stream(media)
    stream._asyncio_loop = asyncio.get_running_loop()
    task = asyncio.create_task(stream.play_loop())
    try:
        await progress_while_blocked(media)
        caller = threading.Thread(target=stream.clear_audio_queue)
        caller.start()
        caller.join(timeout=0.1)
        assert not caller.is_alive()
        for _ in range(8):
            await asyncio.sleep(0)
        assert "flush" not in media.calls
    finally:
        stream._stop_event.set()
        media.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await stream._shutdown_audio()
        stream._asyncio_loop = None


@pytest.mark.asyncio
async def test_flush_coalesces_and_invalidates_frame_waiting_for_native_owner() -> None:
    """A frame already emitted but not yet pushed cannot survive a pause epoch."""
    media = BlockingMedia("playback")
    stream = make_stream(media)
    blocker = asyncio.create_task(stream._playback_io.call(lambda: media._block("playback")))
    task = asyncio.create_task(stream.play_loop())
    try:
        await progress_while_blocked(media)
        # Let play_loop take the frame and queue for the single busy owner.
        for _ in range(8):
            await asyncio.sleep(0)
        assert stream.handler.output_queue.empty()
        stream.clear_audio_queue()
        first_flush = stream._flush_task
        assert first_flush is not None
        for _ in range(100):
            stream.clear_audio_queue()
            assert stream._flush_task is first_flush
        assert stream._playback_io._mailbox.maxsize == 1
        assert stream._playback_io._mailbox.qsize() <= 1
        media.release.set()
        await blocker
        await first_flush
        assert media.calls == ["flush"]
        assert media.pushed == []
    finally:
        stream._stop_event.set()
        media.release.set()
        task.cancel()
        await asyncio.gather(task, blocker, return_exceptions=True)
        await stream._shutdown_audio()


@pytest.mark.asyncio
async def test_order_format_and_single_playback_owner_are_preserved() -> None:
    """Ordered stereo int16 input still selects channel zero and resamples PCM."""
    media = BlockingMedia("none")
    stream = make_stream(media)
    stream.handler.output_queue = asyncio.Queue()
    for value in range(1, 33):
        pcm = np.array([[value] * 8, [99] * 8], dtype=np.int16)
        stream.handler.output_queue.put_nowait((12000, pcm))
    task = asyncio.create_task(stream.play_loop())
    try:
        # A streaming filter retains history and has a delayed tail; native
        # packet counts are not source packet counts. Assert every sample and
        # its order against independently converted whole PCM, including EOF.
        from av.audio.frame import AudioFrame
        from av.audio.resampler import AudioResampler

        assert await asyncio.wait_for(stream.drain_audio(), 1) is True
        source = np.repeat(np.arange(1, 33, dtype=np.float32), 8) / 32768
        reference_frame = AudioFrame.from_ndarray(source.reshape(1, -1), format="fltp", layout="mono")
        reference_frame.sample_rate = 12000
        reference = AudioResampler(format="fltp", layout="mono", rate=24000)
        expected = np.concatenate([
            item.to_ndarray().reshape(-1)
            for item in reference.resample(reference_frame) + reference.resample(None)
        ])
        actual = np.concatenate(media.pushed)
        assert actual.size == expected.size == 512
        for output_pcm in media.pushed:
            assert output_pcm.ndim == 1 and output_pcm.size <= 480
            assert output_pcm.dtype == np.float32
        assert float(actual.max()) < 50 / 32768, "Wrong stereo channel was selected"
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
        assert len(media.push_threads) == 1
        assert threading.get_ident() not in media.push_threads
    finally:
        stream._stop_event.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await stream._shutdown_audio()


@pytest.mark.parametrize("lane", ["capture", "playback", "start_recording", "stop_playing", "stop_error"])
def test_permanently_stalled_io_uses_explicit_process_kill_boundary(lane: str) -> None:
    """Python cannot kill a native thread; never unwind into main's media.close."""
    import os
    import sys
    import textwrap
    import subprocess
    from pathlib import Path

    script = textwrap.dedent(
        """
        import runpy, sys, threading
        from pathlib import Path
        from dj_reachy.console import config, __file__ as origin
        assert Path(origin).resolve() == Path(sys.argv[1]).resolve().parents[1] / 'src/dj_reachy/console.py'
        fixtures = runpy.run_path(sys.argv[1])
        media = fixtures['BlockingMedia'](sys.argv[2])
        def stalled(lane):
            if lane == media.lane:
                media.entered.set()
                media.release.wait()  # deliberately NEVER released
        media._block = stalled
        def forbidden_stop():
            print('UNSAFE_MEDIA_STOP', flush=True)
        media.stop_recording = media.stop_playing = forbidden_stop
        if media.lane == 'start_recording':
            media.start_recording = lambda: stalled('start_recording')
        stream = fixtures['make_stream'](media)
        stream._IO_SHUTDOWN_TIMEOUT = 0.03
        config.VOICE_BACKEND = 'subscription'
        if media.lane in ('stop_playing', 'stop_error'):
            import asyncio
            fault = media.lane == 'stop_error'
            media.lane = 'stop_playing'
            def stop_recording():
                if fault:
                    media.entered.wait()
                    raise RuntimeError('offline stop failure')
            media.stop_recording = stop_recording
            media.stop_playing = lambda: stalled('stop_playing')
            async def start_up():
                stream.close()
                await asyncio.Event().wait()
            stream.handler.start_up = start_up
        else:
            def stop():
                media.entered.wait()
                stream.close()
            threading.Thread(target=stop, daemon=True).start()
        stream.launch()
        print('UNSAFE_LIFESPAN_RETURN', flush=True)
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve()), lane],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        )
    except subprocess.TimeoutExpired:
        pytest.fail("native SDK stall exceeded bounded shutdown without explicit kill fallback")
    assert result.returncode == 70, result.stderr
    assert "UNSAFE_" not in result.stdout
    assert "native audio I/O" in result.stderr
