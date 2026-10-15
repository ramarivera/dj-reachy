"""Credential-free PCM/WebRTC media adapted from Conduit's watch_audio_helper.

MIT sources and revisions are recorded in static/conduit-voice-NOTICE.txt.
"""

from __future__ import annotations
import time
import asyncio
from fractions import Fraction

import av
import numpy as np
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError

from dj_reachy.audio_diagnostics import AudioDiagnostics


class MicrophoneTrack(AudioStreamTrack):
    """Pace bounded PCM16LE mono input at 24 kHz into 48 kHz WebRTC frames."""

    def __init__(self, diagnostics: AudioDiagnostics | None = None) -> None:
        """Create an in-memory media track, never a microphone device."""
        super().__init__()
        self.diagnostics = diagnostics
        self.backlog = bytearray()
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        self.started: float | None = None
        self.timestamp = 0

    def feed(self, pcm: bytes) -> None:
        """Resample input immediately and discard oldest audio beyond two seconds."""
        if len(pcm) % 2:
            if self.diagnostics is not None:
                self.diagnostics.drop("invalid_pcm", size_bytes=len(pcm))
            raise ValueError("PCM16 input must contain whole samples")
        if not pcm or self.readyState != "live":
            if self.diagnostics is not None:
                self.diagnostics.drop("feed_empty" if not pcm else "track_ended", size_bytes=len(pcm))
            return
        input_bytes = len(pcm)
        if self.diagnostics is not None:
            self.diagnostics.audio("feed", np.frombuffer(pcm, dtype=np.int16), 24000)
        # Bound allocation even when a caller sends an oversized block.
        pcm = pcm[-96000:]
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
        frame.planes[0].update(pcm)
        frame.sample_rate = 24000
        for converted in self.resampler.resample(frame):
            self.backlog.extend(bytes(converted.planes[0])[: converted.samples * 2])
        dropped = max(0, len(self.backlog) - 192000)
        if len(self.backlog) > 192000:
            del self.backlog[:-192000]
        if self.diagnostics is not None:
            if input_bytes > 96000:
                self.diagnostics.drop("feed_input_trimmed", size_bytes=input_bytes - 96000)
            if dropped:
                self.diagnostics.drop("feed_backlog_trimmed", size_bytes=dropped)
            self.diagnostics.flow(
                "feed",
                input_bytes=input_bytes,
                backlog_bytes=len(self.backlog),
                dropped_bytes=dropped,
            )

    async def recv(self) -> av.AudioFrame:
        """Return one 20 ms frame, using silence when no input has been supplied."""
        if self.readyState != "live":
            raise MediaStreamError
        if self.started is None:
            self.started = time.monotonic()
        else:
            self.timestamp += 960
            await asyncio.sleep(max(0.0, self.started + self.timestamp / 48000 - time.monotonic()))
        raw = bytes(self.backlog[:1920])
        del self.backlog[:1920]
        frame = av.AudioFrame(format="s16", layout="mono", samples=960)
        frame.planes[0].update(raw.ljust(1920, b"\0"))
        frame.pts = self.timestamp
        frame.sample_rate = 48000
        frame.time_base = Fraction(1, 48000)
        if self.diagnostics is not None:
            self.diagnostics.audio("track", frame.to_ndarray(), 48000)
            self.diagnostics.flow(
                "track",
                input_bytes=len(raw),
                backlog_bytes=len(self.backlog),
                underrun_bytes=1920 - len(raw),
            )
        return frame

    def stop(self) -> None:
        """Release buffered input as well as ending the media track."""
        self.backlog.clear()
        super().stop()
