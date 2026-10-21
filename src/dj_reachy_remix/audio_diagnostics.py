"""Bounded aggregate voice diagnostics; no payloads or audio samples are retained."""

from __future__ import annotations
import os
import sys
import json
import time
import uuid
import logging
from io import TextIOWrapper
from typing import Literal, Callable
from pathlib import Path
from logging.handlers import RotatingFileHandler

import numpy as np
from numpy.typing import NDArray


AudioBoundary = Literal["capture", "selected", "resampled", "encoded", "append", "feed", "track"]
AUDIO_BOUNDARIES = ("capture", "selected", "resampled", "encoded", "append", "feed", "track")
INTERVAL = 5.0
PROVIDER_CATEGORIES = {
    "session.started": "session",
    "session.updated": "session",
    "session.created": "session",
    "session.closed": "session",
    "input_transcript.added": "input_transcript",
    "output_transcript.added": "output_transcript",
    "turn.done": "turn",
    "delegation.created": "delegation",
    "response.function_call_arguments.done": "tool",
    "input_audio_buffer.speech_started": "speech",
    "input_audio_buffer.speech_stopped": "speech",
    "conversation.item.input_audio_transcription.completed": "input_transcript",
    "response.audio_transcript.done": "output_transcript",
    "response.output_audio_transcript.done": "output_transcript",
    "response.audio.delta": "audio",
    "response.output_audio.delta": "audio",
    "response.done": "response",
    "error": "error",
}
SAFE_TOOLS = {"make_song_and_dance", "list_saved_songs", "play_saved_song", "pause_music", "resume_music", "stop_music"}
Lifecycle = Literal["tool", "planner", "delegation", "connection"]
Phase = Literal["started", "completed", "failed", "cancelled", "closed"]
DropReason = Literal[
    "no_connection",
    "append_failed",
    "no_microphone",
    "connection_closed",
    "capture_empty",
    "track_ended",
    "feed_empty",
    "feed_input_trimmed",
    "feed_backlog_trimmed",
    "invalid_pcm",
    "conversion_failed",
]
DROP_REASONS = {
    "no_connection",
    "append_failed",
    "no_microphone",
    "connection_closed",
    "capture_empty",
    "track_ended",
    "feed_empty",
    "feed_input_trimmed",
    "feed_backlog_trimmed",
    "invalid_pcm",
    "conversion_failed",
}


class _PrivateRotatingHandler(RotatingFileHandler):
    """Reopen every rotated file with private permissions and without symlink following."""

    def _open(self) -> TextIOWrapper:
        fd = os.open(self.baseFilename, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
            return TextIOWrapper(os.fdopen(fd, "ab"), encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise

    def doRollover(self) -> None:
        """Refuse unsafe backup targets rather than following or chmodding symlinks."""
        for index in range(self.backupCount + 1):
            path = Path(self.baseFilename + (f".{index}" if index else ""))
            if path.is_symlink():
                raise OSError("Unsafe diagnostic file")
            if path.exists():
                path.chmod(0o600)
        super().doRollover()

    def handleError(self, record: logging.LogRecord) -> None:
        """Never dump a failed record or exception body onto unrelated log sinks."""


class AudioDiagnostics:
    """Own one correlation ID and private, non-propagating aggregate-only handlers."""

    def __init__(
        self,
        *,
        home: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_bytes: int = 1024 * 1024,
        backups: int = 3,
    ) -> None:
        """Defer filesystem access until the voice lifetime explicitly starts."""
        self.run_id = uuid.uuid4().hex
        self.home = home
        self.clock = clock
        self._max_bytes = max(2048, max_bytes)
        self._backups = max(1, min(5, backups))
        self.closed = False
        self.counts: dict[str, int] = {}
        self.owned_handlers: list[logging.Handler] = []
        self._logger = logging.Logger("dj_reachy_remix.audio_diagnostics", level=logging.INFO)
        self._logger.propagate = False
        self._started = False
        self._origin = clock()
        self._last: dict[str, float] = {}
        self._formats: dict[str, tuple[str, tuple[int, ...], int]] = {}
        self._silent: dict[str, tuple[bool, ...]] = {}
        self._last_audio_boundary: str | None = None
        self._webrtc: dict[str, int | str] = {}
        self._nonzero_frames: dict[str, int] = {}
        self._samples_total: dict[str, int] = {}

    def start(self) -> None:
        """Open only owned diagnostic sinks; failures never stop a voice session."""
        if self._started or self.closed:
            return
        self._started = True
        try:
            stdout = logging.StreamHandler(sys.stdout)
            self._logger.addHandler(stdout)
            self.owned_handlers.append(stdout)
            default_home = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "dj-reachy-remix"
            home = (self.home or Path(os.environ.get("DJ_REACHY_REMIX_HOME") or default_home)).expanduser()
            directory = home / "logs"
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if directory.is_symlink():
                raise OSError("Unsafe diagnostic directory")
            directory.chmod(0o700)
            path = directory / "audio-trace.jsonl"
            handler = _PrivateRotatingHandler(
                path, maxBytes=self._max_bytes, backupCount=self._backups, encoding="utf-8"
            )
            self._logger.addHandler(handler)
            self.owned_handlers.append(handler)
        except Exception:
            self._increment("diagnostic_errors")
        self._emit("run_started")

    def _increment(self, key: str, amount: int = 1) -> int:
        self.counts[key] = self.counts.get(key, 0) + amount
        return self.counts[key]

    def _emit(self, event: str, **fields: object) -> None:
        if not self._started or self.closed:
            return
        try:
            record = {
                "event": event,
                "run_id": self.run_id,
                "elapsed_s": round(self.clock() - self._origin, 3),
                **fields,
            }
            self._logger.info(json.dumps(record, allow_nan=False, separators=(",", ":")))
        except Exception:
            self._increment("diagnostic_errors")

    def audio(self, boundary: AudioBoundary, audio: NDArray[np.generic], rate: int) -> None:
        """Measure native PCM dtype/shape and per-channel energy, never its contents."""
        if self.closed or boundary not in AUDIO_BOUNDARIES:
            return
        try:
            total = self._increment(f"audio.{boundary}")
            self._last_audio_boundary = boundary
            shape = tuple(int(size) for size in audio.shape)
            dtype = str(audio.dtype) if audio.dtype.kind in "fiub" else "unsupported"
            if audio.ndim not in (1, 2) or dtype == "unsupported":
                self._increment("diagnostic_errors")
                return
            channels = audio.reshape(-1, 1) if audio.ndim == 1 else audio
            if channels.shape[1] > channels.shape[0]:
                channels = channels.T
            energies: list[dict[str, float | int]] = []
            for channel in channels.T[:16]:
                values = channel.astype(np.float64)
                finite = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
                energies.append(
                    {
                        "peak": float(np.max(np.abs(finite))) if finite.size else 0.0,
                        "rms": float(np.sqrt(np.mean(finite * finite))) if finite.size else 0.0,
                        "nonzero": int(np.count_nonzero(finite)),
                    }
                )
            silent = tuple(energy["nonzero"] == 0 for energy in energies)
            self._nonzero_frames[boundary] = self._nonzero_frames.get(boundary, 0) + int(not all(silent))
            self._samples_total[boundary] = self._samples_total.get(boundary, 0) + audio.size
            fmt = (dtype, shape, rate)
            warning = None
            if boundary in self._formats and self._formats[boundary] != fmt:
                warning = "format_change"
            elif boundary in self._silent and self._silent[boundary] != silent:
                warning = "silence_change"
            elif boundary not in self._last and any(silent):
                warning = "silent_channel"
            self._formats[boundary] = fmt
            self._silent[boundary] = silent
            now = self.clock()
            if boundary not in self._last or now - self._last[boundary] >= INTERVAL or warning:
                # Warnings are coalesced separately to keep alternating audio bounded.
                key = f"{boundary}.{warning}" if warning else boundary
                if warning and key in self._last and now - self._last[key] < INTERVAL:
                    return
                self._last[key] = now
                self._last[boundary] = now
                self._emit(
                    "audio",
                    boundary=boundary,
                    frames_total=total,
                    dtype=dtype,
                    shape=shape,
                    rate=rate,
                    channels=energies,
                    channels_total=channels.shape[1],
                    warning=warning,
                    nonzero_frames_total=self._nonzero_frames[boundary],
                    samples_total=self._samples_total[boundary],
                )
        except Exception:
            self._increment("diagnostic_errors")

    def capture_configuration(self, *, backend: object, rate: int, channels: int) -> None:
        """Report SDK-negotiated format and a known class, never device configuration."""
        safe_backends = {"GStreamerAudio", "SoundDeviceAudio", "GstWebRTCClient", "AudioBase"}
        try:
            name = type(backend).__name__
            self._periodic(
                "capture.configuration",
                "capture_configuration",
                rate=rate,
                channels=channels,
                backend=name if name in safe_backends else "other",
            )
        except Exception:
            self.capture_configuration_unavailable()

    def capture_configuration_unavailable(self) -> None:
        """Keep unsupported SDK metadata non-fatal and payload-free."""
        self._increment("capture.configuration_unavailable")

    def webrtc(
        self,
        *,
        packets_sent: int,
        bytes_sent: int,
        connection_state: str,
        ice_state: str,
        signaling_state: str,
        channel_state: str,
    ) -> None:
        """Keep only outbound RTP counters and finite state names, not transport addresses."""
        if self.closed:
            return
        states = {
            "new",
            "connecting",
            "connected",
            "disconnected",
            "failed",
            "closed",
            "checking",
            "completed",
            "stable",
            "have-local-offer",
            "have-remote-offer",
            "have-local-pranswer",
            "have-remote-pranswer",
            "open",
            "closing",
        }
        self._webrtc = {
            "packets_sent": packets_sent,
            "bytes_sent": bytes_sent,
            "connection_state": connection_state if connection_state in states else "unknown",
            "ice_state": ice_state if ice_state in states else "unknown",
            "signaling_state": signaling_state if signaling_state in states else "unknown",
            "channel_state": channel_state if channel_state in states else "unknown",
        }
        self._increment("webrtc.stats_samples")
        self._periodic("webrtc", "webrtc_stats", **self._webrtc)

    def stats_failed(self) -> None:
        """Count stats failures without accepting an exception or provider body."""
        self._increment("webrtc.stats_failures")
        self._periodic("webrtc.failed", "webrtc_stats_unavailable", total=self.counts["webrtc.stats_failures"])

    def drop(self, reason: DropReason, *, size_bytes: int = 0) -> None:
        """Count only known discard reasons; never accept exception strings."""
        if self.closed or reason not in DROP_REASONS:
            return
        total = self._increment(f"drop.{reason}")
        byte_total = self._increment(f"drop_bytes.{reason}", size_bytes)
        self._periodic(f"drop.{reason}", "input_dropped", reason=reason, total=total, bytes_total=byte_total)

    def flow(
        self,
        boundary: Literal["append", "feed", "track"],
        *,
        input_bytes: int = 0,
        backlog_bytes: int = 0,
        dropped_bytes: int = 0,
        underrun_bytes: int = 0,
    ) -> None:
        """Count transport bytes and current/high-water queue depth independently of logs."""
        if self.closed or boundary not in {"append", "feed", "track"}:
            return
        prefix = f"flow.{boundary}"
        self._increment(f"{prefix}.calls")
        for field, value in (
            ("input_bytes", input_bytes),
            ("dropped_bytes", dropped_bytes),
            ("underrun_bytes", underrun_bytes),
        ):
            self._increment(f"{prefix}.{field}", value)
        high_water = f"{prefix}.backlog_high_water_bytes"
        self.counts[high_water] = max(backlog_bytes, self.counts.get(high_water, 0))
        self._periodic(
            prefix,
            "audio_flow",
            boundary=boundary,
            backlog_bytes=backlog_bytes,
            totals={
                key.removeprefix(prefix + "."): value
                for key, value in self.counts.items()
                if key.startswith(prefix + ".")
            },
        )

    def provider_event(self, kind: str, *, origin: Literal["received", "handled"] = "received") -> None:
        """Count fixed event categories, never arbitrary event names or payloads."""
        if self.closed:
            return
        category = PROVIDER_CATEGORIES.get(kind, "other")
        key = f"provider.handled.{category}" if origin == "handled" else f"provider.{category}"
        total = self._increment(key)
        self._periodic(key, "provider_events", category=category, origin=origin, total=total)

    def lifecycle(self, category: Lifecycle, phase: Phase, *, name: str = "") -> None:
        """Record only finite lifecycle labels and allowlisted tool names."""
        if self.closed or category not in {"tool", "planner", "delegation", "connection"}:
            return
        if phase not in {"started", "completed", "failed", "cancelled", "closed"}:
            return
        tool = name if name in SAFE_TOOLS else "other"
        key = f"{category}.{phase}.{tool}" if category == "tool" else f"{category}.{phase}"
        total = self._increment(key)
        self._periodic(key, "lifecycle", category=category, phase=phase, tool=tool, total=total)

    def _periodic(self, key: str, event: str, **fields: object) -> None:
        try:
            now = self.clock()
            if key not in self._last or now - self._last[key] >= INTERVAL:
                self._last[key] = now
                self._emit(event, **fields)
        except Exception:
            self._increment("diagnostic_errors")

    def close(self) -> None:
        """Write final totals and last observed boundary, then close only owned sinks."""
        if self.closed:
            return
        reached = [boundary for boundary in AUDIO_BOUNDARIES if f"audio.{boundary}" in self.counts]
        nonzero = [boundary for boundary in AUDIO_BOUNDARIES if self._nonzero_frames.get(boundary, 0)]
        self._emit(
            "run_summary",
            counts=dict(self.counts),
            last_audio_boundary=self._last_audio_boundary,
            furthest_audio_boundary=reached[-1] if reached else None,
            furthest_nonzero_boundary=nonzero[-1] if nonzero else None,
            nonzero_frames=dict(self._nonzero_frames),
            samples_total=dict(self._samples_total),
            webrtc=dict(self._webrtc),
        )
        self.closed = True
        for handler in self.owned_handlers:
            self._logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        self.owned_handlers.clear()
