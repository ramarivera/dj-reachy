"""Audio-only launch must not initialize camera capture."""

import logging
from unittest.mock import Mock

import pytest

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend, MediaManager
from dj_reachy_remix import main


def test_audio_only_initializes_audio_without_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the real SDK media type and intercept only the hardware initializer."""
    media = object.__new__(MediaManager)
    media.backend = MediaBackend.NO_MEDIA
    media.camera = None
    media.audio = None
    media.logger = logging.getLogger("audio-only-fixture")
    initialized: list[MediaBackend] = []

    def initialize_audio(owner: MediaManager, log_level: str) -> None:
        assert owner.camera is None
        assert log_level == "INFO"
        initialized.append(owner.backend)
        owner.audio = Mock()

    monkeypatch.setattr(MediaManager, "_init_audio", initialize_audio)
    robot = Mock(spec=ReachyMini)
    robot.media = media
    initialize = getattr(main, "_initialize_audio_only", None)
    assert callable(initialize), "no-camera native launch has no audio initializer"
    initialize(robot, prefer_gstreamer=True)
    assert media.camera is None
    assert media.audio is not None
    assert initialized == [MediaBackend.__members__.get("LOCAL", MediaBackend.GSTREAMER_NO_VIDEO)]
    initialize(robot, prefer_gstreamer=True)
    assert len(initialized) == 1
