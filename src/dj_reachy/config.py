"""Nonsecret runtime settings and process-local provider credentials."""

import os
import sys
from typing import Any
from pathlib import Path

from dj_reachy.runtime_settings import RuntimeSettingsStore


class Config:
    """Keep user state outside the installed package and secrets outside JSON."""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    DEFAULT_PROFILE = "dj_reachy"
    REACHY_MINI_CUSTOM_PROFILE = DEFAULT_PROFILE
    APP_HOME = Path(os.getenv("DJ_REACHY_HOME") or
                    Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "dj-reachy").expanduser()
    SONGS_DIR = APP_HOME / "songs"
    RUNTIME_SETTINGS_PATH = APP_HOME / "settings.json"
    HF_HOME = os.getenv("HF_HOME") or str(Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "dj-reachy/hf")
    LOCAL_VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    ELEVEN_MUSIC_TIMEOUT_S = 180
    RUNTIME_SETTINGS: dict[str, dict[str, Any]] = {}
    MODEL_NAME = "gpt-live-1-codex"
    VOICE_BACKEND = "subscription"
    VOICE_MODEL = "gpt-live-1-codex"
    VOICE_NAME = "cove"
    CONVERSATION_LANGUAGE = "auto"
    MUSIC_MODEL_ID = "music_v2_5"
    ELEVEN_MUSIC_LENGTH_MS = 30000
    MUSIC_OUTPUT_FORMAT = "mp3_48000_192"
    MUSIC_POSITIVE_STYLES: list[str] = []
    MUSIC_NEGATIVE_STYLES: list[str] = []
    MUSIC_LYRICS = ""
    MUSIC_LANGUAGE = "auto"
    MUSIC_FORCE_INSTRUMENTAL = False
    MUSIC_FINETUNE_ID: str | None = None
    MOTION_STYLE = "groove"
    MOTION_INTENSITY = 0.2


config = Config()
runtime_settings_store = RuntimeSettingsStore(config.RUNTIME_SETTINGS_PATH)


def apply_runtime_settings(settings: dict[str, dict[str, Any]]) -> None:
    """Update the provider configuration and any already-created local dancer."""
    voice, music, motion = settings["voice"], settings["music"], settings["motion"]
    config.RUNTIME_SETTINGS = settings
    config.VOICE_BACKEND = voice["backend"]
    config.VOICE_MODEL = voice["model"]
    config.MODEL_NAME = voice["model"]
    config.VOICE_NAME = voice["voice"]
    config.CONVERSATION_LANGUAGE = voice["language"]
    config.MUSIC_MODEL_ID = music["model_id"]
    config.ELEVEN_MUSIC_LENGTH_MS = music["duration_ms"]
    config.MUSIC_OUTPUT_FORMAT = music["output_format"]
    config.MUSIC_POSITIVE_STYLES = list(music["positive_styles"])
    config.MUSIC_NEGATIVE_STYLES = list(music["negative_styles"])
    config.MUSIC_LYRICS = music["lyrics"]
    config.MUSIC_LANGUAGE = music["language"]
    config.MUSIC_FORCE_INSTRUMENTAL = music["force_instrumental"]
    config.MUSIC_FINETUNE_ID = music["finetune_id"]
    config.MOTION_STYLE = motion["style"]
    config.MOTION_INTENSITY = motion["intensity"]
    # Don't import tools while config itself is initializing. Existing dancers update now;
    # dancers created later consume the same bounded config values.
    music_module = sys.modules.get("dj_reachy.profiles.dj_reachy.make_song_and_dance")
    if music_module is not None and hasattr(music_module, "set_dance_intensity"):
        music_module.set_dance_intensity(config.MOTION_INTENSITY)
        music_module.set_dance_style(config.MOTION_STYLE)


def update_runtime_settings(updates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and persist before mutating live provider or motion configuration."""
    settings = RuntimeSettingsStore(config.RUNTIME_SETTINGS_PATH).update(updates)
    apply_runtime_settings(settings)
    return settings


apply_runtime_settings(runtime_settings_store.read())


def set_custom_profile(profile: str | None) -> None:
    """Preserve the fixed DJ tool registry; persona lives in instructions.md."""
    if profile not in (None, "", Config.DEFAULT_PROFILE):
        raise ValueError("DJ Reachy uses a fixed tool profile; edit user instructions instead.")


def set_api_keys(openai_key: str | None = None, elevenlabs_key: str | None = None) -> None:
    """Accept explicitly supplied keys without selecting a billing backend."""
    for name, value in (("OPENAI_API_KEY", openai_key), ("ELEVENLABS_API_KEY", elevenlabs_key)):
        if value and value.strip():
            setattr(config, name, value.strip())
            os.environ[name] = value.strip()
