from __future__ import annotations
import json
from pathlib import Path

import pytest

from dj_reachy_remix.runtime_settings import RuntimeSettingsStore


def test_runtime_settings_persist_separate_validated_groups_atomically(tmp_path: Path) -> None:
    """Persist separate groups without losing normalized settings."""
    store = RuntimeSettingsStore(tmp_path / "settings.json")
    settings = store.update({
        "voice": {"backend": "subscription", "voice": "cove", "language": "es"},
        "music": {"model_id": "music_v2_5", "duration_ms": 30000, "output_format": "mp3_48000_192"},
        "motion": {"style": "groove", "intensity": 0.2},
    })
    assert settings["voice"]["backend"] == "subscription"
    assert settings["music"]["duration_ms"] == 30000
    assert settings["motion"]["intensity"] == 0.2
    assert json.loads((tmp_path / "settings.json").read_text()) == settings


def test_invalid_or_secret_settings_do_not_replace_saved_values(tmp_path: Path) -> None:
    """Reject credential fields without replacing the existing JSON."""
    path = tmp_path / "settings.json"
    store = RuntimeSettingsStore(path)
    original = store.update({"music": {"duration_ms": 60000}})
    content_before = path.read_text()
    with pytest.raises(ValueError):
        store.update({"voice": {"backend": "api_key", "api_key": "never-persist"}})
    assert store.read() == original
    assert path.read_text() == content_before
    assert "never-persist" not in content_before


def test_http_updates_validate_shapes_and_preserve_state(tmp_path: Path) -> None:
    """Route validation rejects malformed values without partial writes."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from dj_reachy_remix.runtime_settings import register_runtime_settings_routes

    path = tmp_path / "settings.json"
    app = FastAPI()
    register_runtime_settings_routes(app, path)
    with TestClient(app) as client:
        response = client.put("/api/dj-reachy-remix/settings", json={"music": {"duration_ms": 10000}})
        assert response.status_code == 200, response.text
        assert response.json()["music"]["duration_ms"] == 10000
        before = path.read_bytes()
        invalid: object
        for invalid in ([], {"music": []}, {"music": {"model_id": []}},
                        {"voice": {"backend": {"nested": True}}}, {"motion": {"intensity": True}},
                        {"voice": {"token": "fixture-secret"}}, {"extra": {}}):
            response = client.put("/api/dj-reachy-remix/settings", json=invalid)
            assert response.status_code == 400, response.text
            assert "fixture-secret" not in response.text
            assert path.read_bytes() == before
        assert client.put("/api/dj-reachy-remix/settings", content="{").status_code == 400


def test_backend_switch_resets_incompatible_defaults_and_rejects_cross_billing(tmp_path: Path) -> None:
    """Backend selection cannot retain a subscription model under API billing."""
    store = RuntimeSettingsStore(tmp_path / "settings.json")
    switched = store.update({"voice": {"backend": "api_key"}})
    assert switched["voice"]["model"] == "gpt-realtime"
    assert switched["voice"]["voice"] == "cedar"
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="subscription"):
        store.update({"voice": {"model": "gpt-live-1-codex", "voice": "cove"}})
    assert store.path.read_bytes() == before
    reverted = store.update({"voice": {"backend": "subscription"}})
    assert reverted["voice"]["model"] == "gpt-live-1-codex"
    assert reverted["voice"]["voice"] == "cove"


def test_persona_reload_keeps_paid_confirmation_and_separates_languages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reload persona additions without dropping the paid-generation policy."""
    from dj_reachy_remix.config import config
    from dj_reachy_remix.prompts import get_session_voice, get_session_instructions
    monkeypatch.setattr(config, "APP_HOME", tmp_path)
    monkeypatch.setattr(config, "VOICE_NAME", "cedar", raising=False)
    monkeypatch.setattr(config, "CONVERSATION_LANGUAGE", "Spanish", raising=False)
    (tmp_path / "instructions.md").write_text("Speak like a jazz radio host.")
    instructions = get_session_instructions()
    assert "jazz radio host" in instructions
    assert "Spanish" in instructions
    assert "confirmation" in instructions
    assert "English" not in instructions
    assert get_session_voice() == "cedar"
    (tmp_path / "instructions.md").write_text("Be a quiet announcer.")
    assert "quiet announcer" in get_session_instructions()
