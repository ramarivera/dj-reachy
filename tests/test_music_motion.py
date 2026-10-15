"""Music preferences reach the existing bounded deterministic local dancer."""

from __future__ import annotations

import numpy as np

import dj_reachy.profiles.dj_reachy.make_song_and_dance as music
from dj_reachy.audio.music_dancer import MusicDancer


def test_zero_music_intensity_really_disables_offsets(monkeypatch):
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
