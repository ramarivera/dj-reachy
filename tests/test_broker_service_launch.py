"""Systemd credential bridge qualification, without sockets or providers."""

from __future__ import annotations
import os
import runpy
from pathlib import Path

import pytest


BRIDGE = Path(__file__).resolve().parents[1] / "deployment" / "run_broker.py"


def test_service_bridge_uses_only_its_scoped_credential(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Resolve the systemd credential, overriding stale inherited token state."""
    import dj_reachy_remix.subscription_broker as broker

    token = "fixture-scoped-broker-token-for-service-testing"
    (tmp_path / "broker-token").write_text(token + "\n")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_TOKEN", "stale-inherited-token")
    observed: list[str] = []

    def fake_main() -> None:
        observed.append(os.environ["DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_TOKEN"])

    monkeypatch.setattr(broker, "main", fake_main)
    runpy.run_path(str(BRIDGE), run_name="__main__")
    assert observed == [token]


def test_service_bridge_refuses_missing_scoped_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not use an inherited token when the service credential is absent."""
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setenv("DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_TOKEN", "stale-inherited-token")
    with pytest.raises(RuntimeError, match="systemd credential directory"):
        runpy.run_path(str(BRIDGE), run_name="__main__")
