"""Honor camera privacy before the SDK-managed app constructs its robot."""

import sys
import contextlib
from types import SimpleNamespace
from collections.abc import Iterator

import pytest

from reachy_mini import ReachyMiniApp
import dj_reachy.main as main


@pytest.mark.parametrize(("flags", "expected_backend"), [(["--no-camera"], "no_media"), (["--settings-only"], "no_media"), ([], "default")])
def test_sdk_managed_robot_backend_honors_camera_flag(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], expected_backend: str
) -> None:
    """Exercise the real SDK wrapper but intercept the device constructor."""
    monkeypatch.setattr(sys, "argv", ["dj-reachy", *flags])
    monkeypatch.setattr(main.ReachyMiniConversationApp, "dont_start_webserver", True)
    for name in ("_check_daemon_on_localhost", "_check_systemd_service_exists"):
        if hasattr(ReachyMiniApp, name):
            monkeypatch.setattr(ReachyMiniApp, name, lambda _self: False)
    monkeypatch.setattr(main.ReachyMiniConversationApp, "run", lambda _self, _robot, _stop: None)
    constructed: list[dict[str, object]] = []

    @contextlib.contextmanager
    def device_constructor(**kwargs: object) -> Iterator[SimpleNamespace]:
        """No camera, microphone, socket or robot is opened by this fixture."""
        constructed.append(kwargs)
        yield SimpleNamespace()

    sdk_module = sys.modules[ReachyMiniApp.__module__]
    monkeypatch.setattr(sdk_module, "ReachyMini", device_constructor)
    app = main.ReachyMiniConversationApp()
    app.wrapped_run()
    assert len(constructed) == 1
    assert constructed[0]["media_backend"] == expected_backend
