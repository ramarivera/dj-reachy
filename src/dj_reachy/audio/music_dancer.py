"""Audio-reactive dance system for music playback.

Analyzes audio in real-time for frequency bands (bass, mid, treble) and beat detection,
then generates expressive full-body dance movements synchronized to the music.
"""

from __future__ import annotations
import math
import time
import logging
import threading
from typing import Tuple, Callable
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

# Frequency band ranges (Hz)
BASS_RANGE = (20, 250)
MID_RANGE = (250, 2000)
TREBLE_RANGE = (2000, 12000)

# Default sample rate (matches OpenAI realtime output)
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHUNK_SIZE = 2048


@dataclass
class AudioFeatures:
    """Extracted audio features for a single frame."""

    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    rms: float = 0.0
    beat_detected: bool = False
    onset_strength: float = 0.0
    bpm: float = 120.0
    beat_phase: float = 0.0  # 0-1, position within current beat cycle
    is_silent: bool = True


@dataclass
class DanceMovement:
    """Full-body dance movement output."""

    # Head pose offsets (meters for x/y/z, radians for roll/pitch/yaw)
    head_x: float = 0.0
    head_y: float = 0.0
    head_z: float = 0.0
    head_roll: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    # Antenna positions (radians)
    antenna_left: float = 0.0
    antenna_right: float = 0.0
    # Body yaw (radians)
    body_yaw: float = 0.0


class AudioAnalyzer:
    """Real-time audio analysis for beat detection and frequency bands."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        sensitivity: float = 0.6,
    ):
        """Initialize FFT bands and deterministic beat-tracking state."""
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.sensitivity = sensitivity

        # FFT setup - precompute frequency bin indices
        freqs = np.fft.rfftfreq(chunk_size, 1.0 / sample_rate)
        self.bass_bins = np.where((freqs >= BASS_RANGE[0]) & (freqs <= BASS_RANGE[1]))[0]
        self.mid_bins = np.where((freqs >= MID_RANGE[0]) & (freqs <= MID_RANGE[1]))[0]
        self.treble_bins = np.where((freqs >= TREBLE_RANGE[0]) & (freqs <= TREBLE_RANGE[1]))[0]

        # Beat tracking state
        self.energy_history: deque[float] = deque(maxlen=10)
        self.beat_times: deque[float] = deque(maxlen=50)
        self.last_beat_time = 0.0
        self.estimated_bpm = 120.0
        self.beat_interval = 0.5  # seconds between beats (60/120 BPM)

        # Timing
        self.start_time = time.monotonic()

    def reset(self) -> None:
        """Reset analyzer state."""
        self.energy_history.clear()
        self.beat_times.clear()
        self.last_beat_time = 0.0
        self.estimated_bpm = 120.0
        self.beat_interval = 0.5
        self.start_time = time.monotonic()

    def analyze(self, audio: NDArray[np.int16]) -> AudioFeatures:
        """Analyze an audio chunk and extract features.

        Args:
            audio: PCM audio data as int16 array

        Returns:
            AudioFeatures with frequency bands, beat info, etc.

        """
        # Convert to float and handle stereo -> mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        audio_float = audio.astype(np.float32) / 32768.0

        current_time = time.monotonic() - self.start_time

        # RMS energy
        rms = float(np.sqrt(np.mean(audio_float**2)))
        is_silent = rms < 0.001

        # Pad or truncate to chunk_size for FFT
        if len(audio_float) < self.chunk_size:
            audio_float = np.pad(audio_float, (0, self.chunk_size - len(audio_float)))
        else:
            audio_float = audio_float[: self.chunk_size]

        # FFT analysis with windowing
        windowed = audio_float * np.hanning(len(audio_float))
        spectrum = np.abs(np.fft.rfft(windowed))

        # Extract band energies (normalized)
        bass = float(min(np.mean(spectrum[self.bass_bins]) / 3.0, 1.0)) if len(self.bass_bins) > 0 else 0.0
        mid = float(min(np.mean(spectrum[self.mid_bins]) / 2.0, 1.0)) if len(self.mid_bins) > 0 else 0.0
        treble = float(min(np.mean(spectrum[self.treble_bins]) / 1.0, 1.0)) if len(self.treble_bins) > 0 else 0.0

        # Beat detection with sensitivity
        self.energy_history.append(rms)
        beat_detected = False
        onset_strength = 0.0
        onset_threshold = 1.1 + (1.0 - self.sensitivity) * 0.5
        min_interval = 0.2 + (1.0 - self.sensitivity) * 0.2

        if len(self.energy_history) >= 3:
            avg_energy = float(np.mean(list(self.energy_history)[:-1]))
            onset = rms / (avg_energy + 1e-10)
            onset_strength = min(onset / onset_threshold, 2.0)
            if onset > onset_threshold and (current_time - self.last_beat_time) > min_interval and rms > 0.002:
                beat_detected = True
                self.beat_times.append(current_time)
                self.last_beat_time = current_time
                self._update_bpm()

        # Calculate beat phase (0-1 position within beat cycle)
        time_since_beat = current_time - self.last_beat_time
        beat_phase = (time_since_beat / self.beat_interval) % 1.0 if self.beat_interval > 0 else 0.0

        return AudioFeatures(
            bass=bass,
            mid=mid,
            treble=treble,
            rms=min(rms * 10, 1.0),
            beat_detected=beat_detected,
            onset_strength=onset_strength,
            bpm=self.estimated_bpm,
            beat_phase=beat_phase,
            is_silent=is_silent,
        )

    def _update_bpm(self) -> None:
        """Estimate BPM from recent beat times."""
        if len(self.beat_times) < 4:
            return
        times = list(self.beat_times)
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        median = float(np.median(intervals))
        valid = [i for i in intervals if 0.5 * median < i < 2 * median]
        if valid:
            avg_interval = float(np.mean(valid))
            self.beat_interval = avg_interval
            self.estimated_bpm = max(60.0, min(200.0, 60.0 / avg_interval))

    def update_sensitivity(self, sensitivity: float) -> None:
        """Update beat detection sensitivity."""
        self.sensitivity = max(0.2, min(1.0, sensitivity))


@dataclass
class DanceStyle:
    """Movement parameters for dance style."""

    # Head movement amplitudes
    head_bob_amplitude: float = 15.0  # degrees for pitch/roll
    head_bob_speed: float = 1.0
    # Body sway
    body_sway_amplitude: float = 70.0  # degrees
    body_sway_speed: float = 1.0
    # Antenna movement
    antenna_amplitude: float = 0.8  # radians
    # Beat emphasis
    emphasis_strength: float = 25.0  # degrees for head pitch on beat
    # Smoothing (0-1, higher = more smoothing)
    movement_smoothing: float = 0.25


class DanceController:
    """Converts audio features to dance movements."""

    def __init__(self, style: DanceStyle | None = None, intensity: float = 0.7):
        """Initialize the existing local movement controller and smoothing state."""
        self.style = style or DanceStyle()
        self.intensity = intensity
        self.dance_time = 0.0

        # Smoothed movement state
        self._smooth_head_z = 0.0
        self._smooth_head_roll = 0.0
        self._smooth_head_pitch = 0.0
        self._smooth_body_yaw = 0.0
        self._smooth_antenna_l = 0.0
        self._smooth_antenna_r = 0.0

    def update_intensity(self, intensity: float) -> None:
        """Update movement intensity, including zero to disable movement."""
        self.intensity = max(0.0, min(1.0, intensity))

    def reset(self) -> None:
        """Reset smoothed state."""
        self._smooth_head_z = 0.0
        self._smooth_head_roll = 0.0
        self._smooth_head_pitch = 0.0
        self._smooth_body_yaw = 0.0
        self._smooth_antenna_l = 0.0
        self._smooth_antenna_r = 0.0
        self.dance_time = 0.0

    def get_movement(self, features: AudioFeatures, dt: float = 0.04) -> DanceMovement:
        """Calculate dance movement based on audio features.

        Args:
            features: Current audio features
            dt: Time delta since last call (seconds)

        Returns:
            DanceMovement with all body part offsets

        """
        self.dance_time += dt
        style = self.style

        # Use beat_phase synced to detected beats, convert to radians
        phase = features.beat_phase * 2 * math.pi

        # Base energy - always move, audio makes it bigger
        base_energy = 0.8 + 0.2 * features.rms
        energy = base_energy * self.intensity

        # BODY SWAY - sweeping motion driven by bass
        bass_boost = 0.8 + 0.5 * features.bass
        body_target = math.radians(style.body_sway_amplitude) * energy * bass_boost * math.sin(phase * style.body_sway_speed)

        # HEAD MOVEMENT - bobbing and rolling driven by mid frequencies
        mid_boost = 0.7 + 0.6 * features.mid
        head_z_target = (style.head_bob_amplitude / 1000.0) * energy * 1.2 * math.sin(phase * style.head_bob_speed)  # Convert to meters
        head_roll_target = math.radians(style.head_bob_amplitude * 2.0) * energy * mid_boost * math.sin(phase * 0.5)

        # Beat-triggered emphasis - punch on beats
        head_pitch_target = 0.0
        if features.beat_detected:
            strength = max(features.onset_strength, 1.2) * self.intensity
            head_pitch_target = -math.radians(style.emphasis_strength) * strength

        # ANTENNAS - bouncy and expressive driven by treble
        treble_boost = 0.6 + 0.6 * features.treble
        ant_amp = style.antenna_amplitude * energy * treble_boost * 2.0
        antenna_l_target = ant_amp * math.sin(phase * 2)
        antenna_r_target = ant_amp * math.sin(phase * 2 + math.pi)

        # Apply smoothing for fluid motion
        smooth = style.movement_smoothing
        self._smooth_head_z = smooth * self._smooth_head_z + (1 - smooth) * head_z_target
        self._smooth_head_roll = smooth * self._smooth_head_roll + (1 - smooth) * head_roll_target
        self._smooth_head_pitch = smooth * self._smooth_head_pitch + (1 - smooth) * head_pitch_target
        self._smooth_body_yaw = smooth * self._smooth_body_yaw + (1 - smooth) * body_target
        self._smooth_antenna_l = smooth * self._smooth_antenna_l + (1 - smooth) * antenna_l_target
        self._smooth_antenna_r = smooth * self._smooth_antenna_r + (1 - smooth) * antenna_r_target

        return DanceMovement(
            head_x=0.0,
            head_y=0.0,
            head_z=max(-0.02, min(0.02, self._smooth_head_z)),  # Clamp to ±20mm
            head_roll=max(-0.8, min(0.8, self._smooth_head_roll)),  # Clamp to ±45 deg
            head_pitch=max(-0.8, min(0.8, self._smooth_head_pitch)),
            head_yaw=0.0,  # Yaw handled by body
            antenna_left=max(-1.0, min(1.0, self._smooth_antenna_l)),
            antenna_right=max(-1.0, min(1.0, self._smooth_antenna_r)),
            body_yaw=max(-1.0, min(1.0, self._smooth_body_yaw)),  # Clamp to ±55 deg approx
        )

    def get_faded_movement(self, fade_factor: float) -> DanceMovement:
        """Get current movement scaled by fade factor (0-1)."""
        return DanceMovement(
            head_x=0.0,
            head_y=0.0,
            head_z=self._smooth_head_z * fade_factor,
            head_roll=self._smooth_head_roll * fade_factor,
            head_pitch=self._smooth_head_pitch * fade_factor,
            head_yaw=0.0,
            antenna_left=self._smooth_antenna_l * fade_factor,
            antenna_right=self._smooth_antenna_r * fade_factor,
            body_yaw=self._smooth_body_yaw * fade_factor,
        )


# Type alias for the offset callback
MusicDanceOffsetCallback = Callable[
    [Tuple[float, float, float, float, float, float], Tuple[float, float], float],
    None,
]


class MusicDancer:
    """High-level music dancer that coordinates analysis and movement.

    Designed to be fed audio chunks during song playback and output
    dance movements via a callback to MovementManager.
    """

    def __init__(
        self,
        set_offsets_callback: MusicDanceOffsetCallback,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        intensity: float = 0.7,
    ):
        """Initialize the music dancer.

        Args:
            set_offsets_callback: Function to call with (head_offsets, antennas, body_yaw)
                where head_offsets is (x, y, z, roll, pitch, yaw) in meters/radians
            sample_rate: Audio sample rate
            intensity: Initial movement intensity (0.1 - 1.0)

        """
        self._set_offsets = set_offsets_callback
        self._sample_rate = sample_rate

        self._analyzer = AudioAnalyzer(sample_rate=sample_rate)
        self._controller = DanceController(intensity=intensity)

        # State
        self._is_active = False
        self._is_fading = False
        self._fade_start_time = 0.0
        self._fade_duration = 0.5  # seconds

        # Threading
        self._lock = threading.Lock()
        self._last_feed_time = time.monotonic()

    @property
    def intensity(self) -> float:
        """Current movement intensity."""
        return self._controller.intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        """Set movement intensity."""
        self._controller.update_intensity(value)

    def start(self) -> None:
        """Start the dancer - call before feeding audio."""
        with self._lock:
            self._analyzer.reset()
            self._controller.reset()
            self._is_active = True
            self._is_fading = False
            self._last_feed_time = time.monotonic()
        logger.info("MusicDancer started")

    def stop(self) -> None:
        """Stop the dancer with fade-out."""
        with self._lock:
            if not self._is_active:
                return
            self._is_fading = True
            self._fade_start_time = time.monotonic()
        logger.info("MusicDancer stopping with fade-out")

        # Run fade-out in current thread (blocking)
        self._run_fade_out()

    def _run_fade_out(self) -> None:
        """Execute the fade-out over ~0.5 seconds."""
        fade_steps = 25  # 25 steps at 20ms each = 500ms
        step_duration = self._fade_duration / fade_steps

        for i in range(fade_steps):
            with self._lock:
                if not self._is_fading:
                    break
                fade_factor = 1.0 - (i + 1) / fade_steps
                movement = self._controller.get_faded_movement(fade_factor)

            # Apply faded offsets
            head_offsets = (
                movement.head_x,
                movement.head_y,
                movement.head_z,
                movement.head_roll,
                movement.head_pitch,
                movement.head_yaw,
            )
            antennas = (movement.antenna_left, movement.antenna_right)
            self._set_offsets(head_offsets, antennas, movement.body_yaw)

            time.sleep(step_duration)

        # Clear all offsets
        self._set_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0), 0.0)

        with self._lock:
            self._is_active = False
            self._is_fading = False
            self._controller.reset()

        logger.info("MusicDancer fade-out complete")

    def feed(self, audio: NDArray[np.int16]) -> None:
        """Feed an audio chunk for analysis and movement generation.

        Args:
            audio: PCM audio data as int16 array (mono or stereo)

        """
        with self._lock:
            if not self._is_active or self._is_fading:
                return

            now = time.monotonic()
            dt = now - self._last_feed_time
            self._last_feed_time = now

        # Analyze audio
        features = self._analyzer.analyze(audio)

        # Skip if silent
        if features.is_silent:
            return

        # Get movement
        movement = self._controller.get_movement(features, dt)

        # Apply offsets via callback
        head_offsets = (
            movement.head_x,
            movement.head_y,
            movement.head_z,
            movement.head_roll,
            movement.head_pitch,
            movement.head_yaw,
        )
        antennas = (movement.antenna_left, movement.antenna_right)
        self._set_offsets(head_offsets, antennas, movement.body_yaw)

    def is_active(self) -> bool:
        """Check if dancer is currently active."""
        with self._lock:
            return self._is_active and not self._is_fading

