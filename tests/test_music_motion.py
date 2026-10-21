"""Music preferences reach the existing bounded deterministic local dancer."""

from __future__ import annotations

import numpy as np
import pytest

import dj_reachy_remix.profiles.dj_reachy.make_song_and_dance as music
from dj_reachy_remix.audio.music_dancer import MusicDancer


def test_zero_music_intensity_really_disables_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Honor zero intensity rather than silently clamping it to nonzero motion."""
    offsets = []
    dancer = MusicDancer(lambda *values: offsets.append(values), intensity=0.2)
    monkeypatch.setattr(music, "_music_dancer", dancer)
    music.set_dance_intensity(0.0)
    music.set_dance_style("bounce")
    dancer.start()
    try:
        sample = (np.sin(np.arange(960) * 0.1) * 16000).astype(np.int16)
        for _ in range(4):
            dancer.feed(sample)
        assert dancer.intensity == 0.0
        assert offsets and all(values == ((0.0,) * 6, (0.0, 0.0), 0.0) for values in offsets)
    finally:
        dancer.stop()


def test_stereo_analysis_preserves_fractional_channel_mean_without_mutating_pcm() -> None:
    """Keep the existing floating downmix instead of quantizing back to int16."""
    from dj_reachy_remix.audio.music_dancer import AudioAnalyzer

    stereo = np.array([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=np.int16)
    before = stereo.copy()
    normalized = np.mean(stereo, axis=0).astype(np.float32) / 32768.0
    expected_rms = min(float(np.sqrt(np.mean(normalized**2))) * 10, 1.0)
    features = AudioAnalyzer(chunk_size=4).analyze(stereo)
    assert features.rms == expected_rms
    np.testing.assert_array_equal(stereo, before)


def test_motion_summary_reports_the_live_preset_and_scale(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Logs must show when saved settings shrink dancing below the original level."""
    dancer = MusicDancer(lambda *_values: None, intensity=0.2)
    monkeypatch.setattr(music, "_music_dancer", dancer)
    monkeypatch.setattr(music, "_music_dancer_style", "groove")
    with caplog.at_level("INFO", logger=music.__name__):
        music.set_dance_style("smooth")
        music.set_dance_intensity(0.05)
    summary = music.motion_summary()
    assert summary == {"style": "smooth", "intensity": 0.05, "percent_of_original": 25}
    assert "smooth" in caplog.text and "0.05" in caplog.text
