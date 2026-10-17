"""Startup must prepare real motor control before the movement loop sends poses."""

import numpy as np
import pytest

from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
import dj_reachy_remix.main as main


@pytest.mark.parametrize("sleeping", [True, False])
def test_motor_startup_enables_before_waking(
    monkeypatch: pytest.MonkeyPatch, sleeping: bool,
) -> None:
    """Intercept hardware methods on the actual SDK type, not fake protocol data."""
    robot = object.__new__(ReachyMini)
    calls: list[str] = []
    pose = SLEEP_HEAD_POSE.copy() if sleeping else np.eye(4)
    monkeypatch.setattr(robot, "get_current_head_pose", lambda: pose)
    monkeypatch.setattr(robot, "enable_motors", lambda: calls.append("enable"))
    monkeypatch.setattr(robot, "wake_up", lambda: calls.append("wake"))
    main._prepare_robot_motion(robot)
    assert calls == (["enable", "wake"] if sleeping else ["enable"])
