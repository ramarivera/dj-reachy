"""Packetization invariance through the actual native output and EOF fence."""

import asyncio

import numpy as np
import pytest
from numpy.typing import NDArray
from test_native_audio_responsiveness import BlockingMedia, make_stream


class NativeRateMedia(BlockingMedia):
    """Expose the actual robot's native output rate without changing a device."""

    def get_output_audio_samplerate(self) -> int:
        """Return the native format used in this regression."""
        return 16000


async def render_native(pcm: NDArray[np.int16], chunk_sizes: list[int]) -> NDArray[np.float32]:
    """Use the real consumer and native completion fence with an intercepted sink."""
    media = NativeRateMedia("")
    stream = make_stream(media)
    stream.handler.output_queue = asyncio.Queue()
    cursor = 0
    index = 0
    while cursor < pcm.size:
        size = chunk_sizes[index % len(chunk_sizes)]
        stream.handler.output_queue.put_nowait((24000, pcm[cursor : cursor + size]))
        cursor += size
        index += 1
    player = asyncio.create_task(stream.play_loop())
    try:
        assert await asyncio.wait_for(stream.drain_audio(), 3) is True
        return np.concatenate([frame.reshape(-1) for frame in media.pushed])
    finally:
        stream._stop_event.set()
        player.cancel()
        await asyncio.gather(player, return_exceptions=True)
        await stream._shutdown_audio()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[480], [7, 521, 13, 77, 960, 1]])
async def test_packet_boundaries_do_not_change_native_pcm(chunks: list[int]) -> None:
    """Preserve one continuous filter phase and drain its final samples at EOF."""
    source = np.arange(24000, dtype=np.float64)
    pcm = (np.sin(2 * np.pi * 443 * source / 24000) * 8191).astype(np.int16)
    whole = await render_native(pcm, [pcm.size])
    packetized = await render_native(pcm, chunks)
    assert whole.size == packetized.size == 16000, "Native conversion lost or duplicated samples"
    np.testing.assert_allclose(packetized, whole, rtol=1e-5, atol=2e-6)


@pytest.mark.asyncio
async def test_flush_discards_resampler_history_before_replacement_pcm() -> None:
    """Neither a filter's pending tail nor its phase survives an interruption."""
    media = NativeRateMedia("")
    flush_frames: list[int] = []

    def cleared() -> None:
        flush_frames.append(len(media.pushed))
        media.calls.append("flush")

    media.audio.clear_player = cleared
    stream = make_stream(media)
    stream.handler.output_queue = asyncio.Queue()
    stream.handler.output_queue.put_nowait((24000, np.full(480, 24000, dtype=np.int16)))
    player = asyncio.create_task(stream.play_loop())
    replacement = np.full(2400, -4000, dtype=np.int16)
    try:
        async with asyncio.timeout(1):
            while not media.pushed:
                await asyncio.sleep(0.001)
        stream.clear_audio_queue()
        stream.handler.output_queue.put_nowait((24000, replacement))
        assert await asyncio.wait_for(stream.drain_audio(), 1) is True
        assert len(flush_frames) == 1
        actual = np.concatenate(media.pushed[flush_frames[0]:])
        expected = await render_native(replacement, [replacement.size])
        assert actual.size == expected.size == 1600
        np.testing.assert_array_equal(actual, expected)
    finally:
        stream._stop_event.set()
        player.cancel()
        await asyncio.gather(player, return_exceptions=True)
        await stream._shutdown_audio()
