"""Bounded streaming PCM decoder using the app's existing PyAV dependency.

Run in an owned subprocess so stop, pause backpressure and cancellation retain
exactly the same lifecycle as an external mpg123 process. No hardware is opened.
"""

from __future__ import annotations
import sys
from collections.abc import Iterator

import av


SAMPLE_RATE = 24000


def pcm_chunks(path: str) -> Iterator[bytes]:
    """Decode one file as packed mono signed 16-bit PCM, including filter tail."""
    converter = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            for output in converter.resample(frame):
                yield output.to_ndarray().astype("<i2", copy=False).tobytes()
        for output in converter.resample(None):
            yield output.to_ndarray().astype("<i2", copy=False).tobytes()


def main() -> None:
    """Write PCM to stdout; keep errors separate from the binary stream."""
    if len(sys.argv) != 2:
        raise SystemExit("Expected one MP3 path")
    try:
        for chunk in pcm_chunks(sys.argv[1]):
            sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        # The owner deliberately closed its decoder pipe during cancellation.
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
