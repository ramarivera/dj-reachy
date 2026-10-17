"""Exercise streaming conversion against real PyAV, without audio devices."""

import numpy as np
import pytest
from numpy.typing import NDArray

from dj_reachy_remix.audio.streaming_pcm import StreamingPCMConverter


def tone(rate: int, samples: int) -> NDArray[np.float32]:
    """Generate a bounded mixture with non-integer cycles at packet boundaries."""
    time = np.arange(samples, dtype=np.float64) / rate
    return np.asarray(
        0.4 * np.sin(2 * np.pi * 437.3 * time) + 0.2 * np.sin(2 * np.pi * 1291.7 * time), dtype=np.float32
    )


@pytest.mark.parametrize("input_rate", [24000, 48000, 44100])
def test_drains_resampled_tone_at_requested_rate(input_rate: int) -> None:
    """Retain the delayed tail and produce mono contiguous float32 PCM."""
    frame = tone(input_rate, input_rate + 137)
    converter = StreamingPCMConverter(input_rate, 16000)
    chunks = converter.push(frame)
    tail = converter.finish()

    assert chunks
    assert tail
    for chunk in chunks + tail:
        assert chunk.ndim == 1
        assert chunk.dtype == np.float32
        assert chunk.flags.c_contiguous
        assert chunk.size > 0
    result = np.concatenate(chunks + tail)
    assert abs(result.size - frame.size * 16000 / input_rate) <= 1
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 0.7
    np.testing.assert_allclose(result[100:-100], tone(16000, result.size)[100:-100], atol=0.002, rtol=0)


@pytest.mark.parametrize("input_rate", [24000, 48000, 44100])
@pytest.mark.parametrize("packetization", ["20ms", "irregular"])
def test_packet_boundaries_do_not_change_samples(input_rate: int, packetization: str) -> None:
    """Preserve phase across regular and tiny irregular packets, including tails."""
    frame = tone(input_rate, input_rate + 137)
    whole = StreamingPCMConverter(input_rate, 16000)
    expected = np.concatenate(whole.push(frame) + whole.finish())
    sizes = [input_rate // 50] if packetization == "20ms" else [1, 7, 313, 2, 997, 31, 819, 5]
    converter = StreamingPCMConverter(input_rate, 16000)
    chunks: list[NDArray[np.float32]] = []
    offset = 0
    packet = 0
    while offset < frame.size:
        size = sizes[packet % len(sizes)]
        chunks.extend(converter.push(frame[offset : offset + size]))
        offset += size
        packet += 1
    chunks.extend(converter.finish())
    actual = np.concatenate(chunks)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)


def test_empty_packets_do_not_change_stream_state() -> None:
    """Ignore empties before and between real packets without flushing the tail."""
    frame = tone(44100, 9871)
    whole = StreamingPCMConverter(44100, 16000)
    expected = np.concatenate(whole.push(frame) + whole.finish())
    converter = StreamingPCMConverter(44100, 16000)
    empty = np.empty(0, dtype=np.float32)
    assert converter.push(empty) == []
    chunks = converter.push(frame[:4111])
    assert converter.push(empty) == []
    chunks.extend(converter.push(frame[4111:]))
    assert converter.push(empty) == []
    chunks.extend(converter.finish())
    np.testing.assert_array_equal(np.concatenate(chunks), expected)


@pytest.mark.parametrize("has_input", [False, True])
def test_finish_closes_stream_and_never_replays_tail(has_input: bool) -> None:
    """Make finishing idempotent and reject every push after closing."""
    converter = StreamingPCMConverter(24000, 16000)
    if has_input:
        converter.push(tone(24000, 1237))
    tail = converter.finish()
    assert bool(tail) is has_input
    assert converter.finish() == []
    for frame in [np.empty(0, dtype=np.float32), tone(24000, 500)]:
        with pytest.raises(RuntimeError, match="finished"):
            converter.push(frame)


def test_fresh_instance_cannot_revive_discarded_tail() -> None:
    """Replacing a stream discards only its state, not a replacement's silence."""
    discarded = StreamingPCMConverter(44100, 16000)
    discarded.push(tone(44100, 1347))
    fresh = StreamingPCMConverter(44100, 16000)
    silence = np.zeros(9761, dtype=np.float32)
    chunks = fresh.push(silence)
    chunks.extend(fresh.finish())
    np.testing.assert_array_equal(
        np.concatenate(chunks), np.zeros(sum(chunk.size for chunk in chunks), dtype=np.float32)
    )
    assert discarded.finish()
    assert fresh.finish() == []


@pytest.mark.parametrize("bad_rate", [0, -16000, True, 16000.5, "16000", None])
@pytest.mark.parametrize("parameter", ["input_rate", "output_rate"])
def test_rejects_invalid_rates(bad_rate: int, parameter: str) -> None:
    """Reject invalid rate values before creating a PyAV stream."""
    rates = {"input_rate": 24000, "output_rate": 16000, parameter: bad_rate}
    with pytest.raises((TypeError, ValueError), match=rf"{parameter}.*positive integer"):
        StreamingPCMConverter(**rates)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (np.zeros(8, dtype=np.float64), "float32"),
        (np.zeros(8, dtype=np.int16), "float32"),
        (np.zeros((1, 8), dtype=np.float32), "one-dimensional mono"),
        (np.zeros((8, 2), dtype=np.float32), "one-dimensional mono"),
        (np.zeros((0, 2), dtype=np.float32), "one-dimensional mono"),
        (np.array(0, dtype=np.float32), "one-dimensional mono"),
        (np.array([np.nan], dtype=np.float32), "finite"),
        (np.array([np.inf], dtype=np.float32), "finite"),
        ([0.0, 0.5], "numpy.ndarray"),
    ],
)
def test_invalid_format_does_not_poison_stream(frame: NDArray[np.float32], message: str) -> None:
    """Validate packets before mutating PTS or filter state, including empties."""
    converter = StreamingPCMConverter(24000, 16000)
    valid = tone(24000, 1901)
    chunks = converter.push(valid[:877])
    with pytest.raises((TypeError, ValueError), match=message):
        converter.push(frame)
    chunks.extend(converter.push(valid[877:]))
    chunks.extend(converter.finish())
    reference = StreamingPCMConverter(24000, 16000)
    expected = np.concatenate(reference.push(valid) + reference.finish())
    np.testing.assert_array_equal(np.concatenate(chunks), expected)


def test_strided_mono_input_produces_contiguous_pcm() -> None:
    """Copy non-contiguous mono input as needed without modifying its samples."""
    original = tone(48000, 15002)
    frame = original[::2]
    assert not frame.flags.c_contiguous
    before = original.copy()
    converter = StreamingPCMConverter(24000, 16000)
    chunks = converter.push(frame) + converter.finish()
    assert all(chunk.flags.c_contiguous for chunk in chunks)
    reference = StreamingPCMConverter(24000, 16000)
    expected = np.concatenate(reference.push(frame.copy()) + reference.finish())
    np.testing.assert_array_equal(np.concatenate(chunks), expected)
    np.testing.assert_array_equal(original, before)


@pytest.mark.parametrize("input_rate", [24000, 48000, 44100])
@pytest.mark.parametrize("samples", [1, 7, 31, 63])
def test_finish_conserves_streams_shorter_than_filter_delay(input_rate: int, samples: int) -> None:
    """Drain even complete streams too short for PyAV's native flush to emit PCM."""
    frame = tone(input_rate, samples)
    whole = StreamingPCMConverter(input_rate, 16000)
    chunks = whole.push(frame) + whole.finish()
    expected_size = (samples * 16000 + input_rate - 1) // input_rate
    assert sum(chunk.size for chunk in chunks) == expected_size
    result = np.concatenate(chunks)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 0.7
    split = StreamingPCMConverter(input_rate, 16000)
    split_chunks = split.push(frame[:3]) + split.push(frame[3:]) + split.finish()
    np.testing.assert_array_equal(np.concatenate(split_chunks), result)
