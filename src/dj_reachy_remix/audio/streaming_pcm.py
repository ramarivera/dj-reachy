"""Stateful mono float32 PCM sample-rate conversion."""

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler


class StreamingPCMConverter:
    """Retain resampling phase and delayed samples across PCM packets."""

    def __init__(self, input_rate: int, output_rate: int) -> None:
        """Create an independent conversion stream with positive integer rates."""
        for name, rate in (("input_rate", input_rate), ("output_rate", output_rate)):
            if isinstance(rate, bool) or not isinstance(rate, int):
                raise TypeError(f"{name} must be a positive integer")
            if rate <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._input_rate = input_rate
        self._output_rate = output_rate
        self._time_base = Fraction(1, input_rate)
        self._pts = 0
        self._emitted_samples = 0
        self._finished = False
        self._resampler = AudioResampler(format="fltp", layout="mono", rate=output_rate)

    def push(self, frame: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        """Convert a packet while retaining the filter's delayed tail."""
        if self._finished:
            raise RuntimeError("Cannot push PCM to a finished converter")
        if not isinstance(frame, np.ndarray):
            raise TypeError("PCM must be a numpy.ndarray of float32 samples")
        if frame.dtype != np.float32:
            raise TypeError("PCM samples must have dtype float32")
        if frame.ndim != 1:
            raise ValueError("PCM must be one-dimensional mono audio")
        if not np.isfinite(frame).all():
            raise ValueError("PCM samples must be finite")
        if frame.size == 0:
            return []
        return self._resample(frame)

    def _resample(self, frame: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        """Submit PCM with continuous input timestamps to the owned filter."""
        audio = AudioFrame.from_ndarray(np.ascontiguousarray(frame.reshape(1, -1)), format="fltp", layout="mono")
        audio.sample_rate = self._input_rate
        audio.time_base = self._time_base
        audio.pts = self._pts
        self._pts += frame.size
        chunks = self._pcm(self._resampler.resample(audio))
        self._emitted_samples += sum(chunk.size for chunk in chunks)
        return chunks

    def finish(self) -> list[NDArray[np.float32]]:
        """Drain the tail and close the stream; subsequent finishes return nothing."""
        if self._finished:
            return []
        self._finished = True
        if self._pts and not self._emitted_samples:
            # FFmpeg's native flush can drop streams shorter than its filter.
            # Extend those with silence, but return only the real duration.
            remaining = (self._pts * self._output_rate + self._input_rate - 1) // self._input_rate
            padding = np.zeros(max(1, self._input_rate // 50), dtype=np.float32)
            tail: list[NDArray[np.float32]] = []
            while remaining:
                for chunk in self._resample(padding):
                    take = min(remaining, chunk.size)
                    if take:
                        tail.append(chunk[:take])
                        remaining -= take
            self._resampler.resample(None)
            return tail
        return self._pcm(self._resampler.resample(None))

    @staticmethod
    def _pcm(frames: list[AudioFrame]) -> list[NDArray[np.float32]]:
        return [np.ascontiguousarray(frame.to_ndarray().reshape(-1), dtype=np.float32) for frame in frames]
