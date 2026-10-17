"""Keep SDK live-mixer deadlines consistent with buffered streaming audio.

SDK 1.12's default mixer can render silence before a delayed voice packet
arrives. Extending only the mixer latency is insufficient: its appsrc timestamp
reset must also allow packets to arrive within that buffering window. This
configuration is per audio instance and never edits the shared SDK or device.
"""

from typing import Protocol, runtime_checkable


class _Factory(Protocol):
    """The factory metadata needed to select the live mixer."""

    def get_name(self) -> str:
        """Return the installed GStreamer factory name."""
        ...


class _Element(Protocol):
    """The scheduling properties used on a native pipeline element."""

    def get_factory(self) -> _Factory | None:
        """Identify a native element without importing optional GI bindings."""
        ...

    def set_property(self, name: str, value: int) -> None:
        """Set a native integer scheduling property."""
        ...

    def get_property(self, name: str) -> object:
        """Read the current buffering window without shrinking custom settings."""
        ...


class _IteratorStatus(Protocol):
    """GI's native enum result, not a Python iteration sentinel."""

    @property
    def value_nick(self) -> str:
        """Return the stable native result name."""
        ...


class _Iterator(Protocol):
    """GI's explicit next/resync contract."""

    def next(self) -> tuple[_IteratorStatus, _Element | None]:
        """Read a result and element without assuming __iter__ support."""
        ...

    def resync(self) -> None:
        """Restart after the native pipeline reports a structure change."""
        ...


class _Pipeline(Protocol):
    """The per-instance pipeline, never a shared device configuration."""

    def iterate_recurse(self) -> _Iterator:
        """Enumerate elements, including nested device-output bins."""
        ...


@runtime_checkable
class _NativeAudio(Protocol):
    """SDK capabilities that require coordinated live scheduling."""

    _pipeline: _Pipeline
    GAP_RESET_NS: int


def configure_native_playback(audio: object) -> float:
    """Configure the SDK live mixer and return its conservative drain allowance.

    Other audio backends retain their existing behavior. The native path fails
    explicitly if its expected mixer is missing rather than claiming that a
    streaming buffer was installed.
    """
    if not isinstance(audio, _NativeAudio):
        return 0.0
    iterator = audio._pipeline.iterate_recurse()
    while True:
        result, element = iterator.next()
        if result.value_nick == "done":
            break
        if result.value_nick == "resync":
            iterator.resync()
            continue
        if result.value_nick != "ok" or element is None:
            raise RuntimeError("The native audio playback mixer could not be enumerated")
        factory = element.get_factory()
        if factory is not None and factory.get_name() == "audiomixer":
            current = element.get_property("latency")
            if not isinstance(current, int) or current < 0:
                raise RuntimeError("The native audio mixer has no valid playback latency")
            latency_ns = max(current, 320_000_000)
            element.set_property("latency", latency_ns)
            audio.GAP_RESET_NS = max(audio.GAP_RESET_NS, latency_ns + 200_000_000)
            # Include the native sink/silence-source latency, not only the
            # additional mixer window, before flushing a completed local song.
            return latency_ns / 1_000_000_000 + 0.18
    raise RuntimeError("The native audio backend has no streaming playback mixer")
