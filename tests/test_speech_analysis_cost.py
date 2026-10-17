"""Regression for duration-growing speech-motion work, GitHub issue #1."""

from collections import deque
from collections.abc import Iterator

import numpy as np
import pytest

from dj_reachy_remix.audio.speech_tapper import HOP, FRAME, SwayRollRT


class CountedSamples(deque[float]):
    """Count real values traversed by the analysis window, without timing noise."""

    visited = 0

    def __iter__(self) -> Iterator[float]:
        """Count actual traversal rather than relying on a CPU-speed threshold."""
        for value in super().__iter__():
            self.visited += 1
            yield value


def test_speech_analysis_work_does_not_grow_with_utterance_length() -> None:
    """Keep per-hop analysis cost constant throughout a long utterance."""
    analysis = SwayRollRT()
    samples = CountedSamples(maxlen=analysis.samples.maxlen)
    analysis.samples = samples
    hop = np.full(HOP, 0.2, dtype=np.float32)
    for _ in range(3000):
        before = samples.visited
        analysis.feed(hop, 16000)
        assert samples.visited - before <= FRAME, (
            "Speech analysis rescans elapsed audio instead of only its current frame"
        )


@pytest.mark.parametrize("sample_rate", [8000, 16000, 24000, 48000])
def test_bounded_analysis_preserves_legacy_motion_outputs(sample_rate: int) -> None:
    """Preserve motion values across chunk boundaries, rates and reset."""
    bounded = SwayRollRT()
    legacy = SwayRollRT()
    legacy.samples = deque(maxlen=160000)
    rng = np.random.default_rng(29)
    pcm = rng.integers(-24000, 24000, size=sample_rate * 3, dtype=np.int16)
    sizes = (1, 37, 256, 480, 1024, 173)
    cursor = 0
    index = 0
    while cursor < len(pcm):
        end = min(len(pcm), cursor + sizes[index % len(sizes)])
        chunk = pcm[cursor:end]
        assert bounded.feed(chunk, sample_rate) == legacy.feed(chunk, sample_rate)
        cursor = end
        index += 1
    bounded.reset()
    legacy.reset()
    assert bounded.feed(pcm[:1024], sample_rate) == legacy.feed(pcm[:1024], sample_rate)
