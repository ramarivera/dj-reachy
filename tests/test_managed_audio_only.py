"""Honor camera privacy before the SDK-managed app constructs its robot."""

import sys
import contextlib
from types import SimpleNamespace
from collections.abc import Iterator

import pytest

from reachy_mini import ReachyMiniApp
import dj_reachy_remix.main as main


@pytest.mark.parametrize(("flags", "expected_backend", "daemon_local"), [
    (["--no-camera"], "no_media", False),
    (["--settings-only"], "no_media", False),
    ([], "default", False),
    ([], "gstreamer", True),
    (["--no-camera"], "no_media", True),
    (["--settings-only"], "no_media", True),
])
def test_sdk_managed_robot_backend_honors_camera_flag(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], expected_backend: str, daemon_local: bool
) -> None:
    """Exercise the real SDK wrapper but intercept the device constructor."""
    monkeypatch.setattr(sys, "argv", ["dj-reachy-remix", *flags])
    monkeypatch.setattr(main.ReachyMiniConversationApp, "dont_start_webserver", True)
    # Older supported SDKs lack the locality attribute; model the current
    # SDK's discovery result while still exercising its real managed wrapper.
    monkeypatch.setattr(main.ReachyMiniConversationApp, "daemon_on_localhost", daemon_local, raising=False)
    for name in ("_check_daemon_on_localhost", "_check_systemd_service_exists"):
        if hasattr(ReachyMiniApp, name):
            local = daemon_local if name == "_check_daemon_on_localhost" else False
            monkeypatch.setattr(ReachyMiniApp, name, lambda _self, value=local: value)
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
