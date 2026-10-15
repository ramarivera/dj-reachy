from __future__ import annotations

import json

import pytest

from dj_reachy.runtime_settings import RuntimeSettingsStore


def test_runtime_settings_persist_separate_validated_groups_atomically(tmp_path) -> None:
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


def test_invalid_or_secret_settings_do_not_replace_saved_values(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = RuntimeSettingsStore(path)
    original = store.update({"music": {"duration_ms": 60000}})
    content_before = path.read_text()
    with pytest.raises(ValueError):
        store.update({"voice": {"backend": "api_key", "api_key": "never-persist"}})
    assert store.read() == original
    assert path.read_text() == content_before
    assert "never-persist" not in content_before
