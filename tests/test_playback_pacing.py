"""Exercise sample-clock scheduling through the actual native playback loop."""

import time
import asyncio

import numpy as np
import pytest
from numpy.typing import NDArray
from test_native_audio_responsiveness import BlockingMedia, make_stream


class TimedMedia(BlockingMedia):
    """Intercept native submission without a device or provider."""

    def __init__(self) -> None:
        """Initialize bounded submission observations."""
        super().__init__("")
        self.times: list[float] = []

    def push_audio_sample(self, frame: NDArray[np.float32]) -> None:
        """Capture samples and the actual worker's monotonic submission time."""
        super().push_audio_sample(frame)
        self.times.append(time.monotonic())


@pytest.mark.asyncio
async def test_burst_frame_has_bounded_sample_clock_lookahead() -> None:
    """A large speech delta must not become an unbounded native write burst."""
    media = TimedMedia()
    stream = make_stream(media)
    stream.handler.output_queue = asyncio.Queue()
    pcm = np.arange(12000, dtype=np.int16)
    stream.handler.output_queue.put_nowait((24000, pcm))
    player = asyncio.create_task(stream.play_loop())
    try:
        async with asyncio.timeout(2):
            while sum(frame.size for frame in media.pushed) != pcm.size:
                await asyncio.sleep(0.001)
        assert max(frame.size for frame in media.pushed) <= 480, "SDK frames exceed 20ms"
        submitted = np.cumsum([frame.size for frame in media.pushed]) / 24000
        elapsed = np.array(media.times) - media.times[0]
        assert float(np.max(submitted - elapsed)) <= 0.14, "PCM submission outran the bounded lead"
        assert media.times[-1] - media.times[0] >= 0.34, "Speech was submitted as an immediate burst"
        np.testing.assert_allclose(np.concatenate(media.pushed), pcm.astype(np.float32) / 32768)
    finally:
        stream._stop_event.set()
        player.cancel()
        await asyncio.gather(player, return_exceptions=True)
        await stream._shutdown_audio()


@pytest.mark.asyncio
async def test_flush_interrupts_a_paced_frame_without_replaying_old_chunks() -> None:
    """Reset while sleeping on the sample clock and accept replacement PCM now."""
    media = TimedMedia()
    stream = make_stream(media)
    stream.handler.output_queue = asyncio.Queue()
    stream.handler.output_queue.put_nowait((24000, np.ones(24000, dtype=np.int16)))
    player = asyncio.create_task(stream.play_loop())
    try:
        async with asyncio.timeout(1):
            while sum(frame.size for frame in media.pushed) < 2400:
                await asyncio.sleep(0.001)
        stream.clear_audio_queue()
        replacement = np.full(240, -1000, dtype=np.int16)
        stream.handler.output_queue.put_nowait((24000, replacement))
        async with asyncio.timeout(0.15):
            while not any(np.any(frame < 0) for frame in media.pushed):
                await asyncio.sleep(0.001)
        assert "flush" in media.calls
        flush_at = media.calls.index("flush")
        assert media.calls[flush_at + 1:] == ["push"]
        np.testing.assert_allclose(media.pushed[-1], replacement.astype(np.float32) / 32768)
        await asyncio.sleep(0.06)
        assert media.calls[flush_at + 1:] == ["push"], "Stale chunks survived the flush"
    finally:
        stream._stop_event.set()
        player.cancel()
        await asyncio.gather(player, return_exceptions=True)
        await stream._shutdown_audio()


@pytest.mark.asyncio
async def test_idle_carrier_does_not_age_the_native_utterance_clock() -> None:
    """Long WebRTC listen silence ends a cue, not a stale never-ending utterance."""
    media = TimedMedia()
    stream = make_stream(media)
    stream._output_tail_delay = 0.5
    stream.handler.output_queue = asyncio.Queue()
    stream.handler.output_queue.put_nowait((24000, np.zeros(48000, dtype=np.int16)))
    signal = np.full(2400, 4096, dtype=np.int16)
    stream.handler.output_queue.put_nowait((24000, signal))
    player = asyncio.create_task(stream.play_loop())
    try:
        assert await asyncio.wait_for(stream.drain_audio(), 4) is True
        silent_samples = sum(frame.size for frame in media.pushed if not np.any(frame))
        assert silent_samples <= 19200, "Listening carrier kept aging the SDK's native cue clock"
        assert "flush" in media.calls, "New speech retained the idle cue's clock debt"
        nonzero = [frame for frame in media.pushed if np.any(frame)]
        np.testing.assert_array_equal(np.concatenate(nonzero), signal.astype(np.float32) / 32768)
        first_signal = next(i for i, frame in enumerate(media.pushed) if np.any(frame))
        assert media.times[first_signal] - media.times[0] >= 1.8, "Idle suppression compressed intended silence"
    finally:
        stream._stop_event.set()
        player.cancel()
        await asyncio.gather(player, return_exceptions=True)
        await stream._shutdown_audio()


@pytest.mark.asyncio
async def test_idle_reset_waits_for_existing_wider_native_tail() -> None:
    """A wider configured native buffer must never lose its preceding speech."""
    media = TimedMedia()
    stream = make_stream(media)
    stream._output_tail_delay = 0.82
    stream.handler.output_queue = asyncio.Queue()
    prefix = np.full(2400, 4096, dtype=np.int16)
    silence = np.zeros(21600, dtype=np.int16)
    suffix = np.full(2400, -4096, dtype=np.int16)
    for frame in (prefix, silence, suffix):
        stream.handler.output_queue.put_nowait((24000, frame))
    player = asyncio.create_task(stream.play_loop())
    try:
        assert await asyncio.wait_for(stream.drain_audio(), 4) is True
        assert "flush" not in media.calls, "Idle reset cut a still-buffered wider native tail"
        np.testing.assert_array_equal(np.concatenate(media.pushed), np.concatenate([prefix,silence,suffix]).astype(np.float32) / 32768)
    finally:
        stream._stop_event.set()
        player.cancel()
        await asyncio.gather(player, return_exceptions=True)
        await stream._shutdown_audio()
