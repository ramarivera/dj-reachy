import os
import logging
from pathlib import Path


from dj_reachy.runtime_settings import RuntimeSettingsStore
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for the conversation app.
    
    API keys (OPENAI_API_KEY, ELEVENLABS_API_KEY) are provided via the web UI.
    Other settings are hardcoded with sensible defaults.
    """

    # API Keys - provided via web UI, fall back to env vars if set
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

    # Model settings (hardcoded)
    MODEL_NAME = "gpt-4o-realtime-preview"

    # ElevenLabs music settings (hardcoded)
    ELEVEN_MUSIC_LENGTH_MS = 30000
    ELEVEN_MUSIC_TIMEOUT_S = 180

    # Vision model settings (hardcoded)
    HF_HOME = "./cache"
    LOCAL_VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    # Profile settings - always use dj_reachy profile
    # This is the default DJ profile and cannot be changed via environment
    DEFAULT_PROFILE = "dj_reachy"
    REACHY_MINI_CUSTOM_PROFILE = DEFAULT_PROFILE

    # User-owned data, never source-package files.
    APP_HOME = Path(
        os.environ.get("DJ_REACHY_HOME")
        or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "dj-reachy"
    )
    SONGS_DIR = APP_HOME / "songs"
    RUNTIME_SETTINGS_PATH = APP_HOME / "settings.json"

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")


config = Config()


runtime_settings_store = RuntimeSettingsStore(config.RUNTIME_SETTINGS_PATH)


def apply_runtime_settings(settings: dict) -> None:
    """Apply nonsecret settings to the existing provider and dance config."""
    voice = settings["voice"]
    music = settings["music"]
    motion = settings["motion"]
    config.RUNTIME_SETTINGS = settings
    config.VOICE_BACKEND = voice["backend"]
    config.VOICE_MODEL = voice["model"]
    config.VOICE_NAME = voice["voice"]
    config.CONVERSATION_LANGUAGE = voice["language"]
    config.MUSIC_MODEL_ID = music["model_id"]
    config.ELEVEN_MUSIC_LENGTH_MS = music["duration_ms"]
    config.MUSIC_OUTPUT_FORMAT = music["output_format"]
    config.MUSIC_POSITIVE_STYLES = music["positive_styles"]
    config.MUSIC_NEGATIVE_STYLES = music["negative_styles"]
    config.MUSIC_LYRICS = music["lyrics"]
    config.MUSIC_LANGUAGE = music["language"]
    config.MUSIC_FORCE_INSTRUMENTAL = music["force_instrumental"]
    config.MUSIC_FINETUNE_ID = music["finetune_id"]
    config.MOTION_STYLE = motion["style"]
    config.MOTION_INTENSITY = motion["intensity"]


def update_runtime_settings(updates: dict) -> dict:
    settings = runtime_settings_store.update(updates)
    apply_runtime_settings(settings)
    return settings


apply_runtime_settings(runtime_settings_store.read())


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime.

    Note: This function is a no-op - the profile is always fixed to dj_reachy.
    This ensures the DJ Reachy experience is consistent for all users.
    """
    # Profile is locked to dj_reachy - ignore any attempts to change it
    pass


def set_api_keys(openai_key: str | None = None, elevenlabs_key: str | None = None) -> None:
    """Update API keys at runtime (from web UI)."""
    if openai_key:
        config.OPENAI_API_KEY = openai_key
        os.environ["OPENAI_API_KEY"] = openai_key
    if elevenlabs_key:
        config.ELEVENLABS_API_KEY = elevenlabs_key
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
