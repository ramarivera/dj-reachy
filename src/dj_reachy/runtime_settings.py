"""User-owned runtime settings and narrowly scoped app-manager routes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "voice": {"backend": "subscription", "model": "gpt-live-1-codex", "voice": "cove", "language": "auto"},
    "music": {
        "model_id": "music_v2_5", "duration_ms": 30000, "output_format": "mp3_48000_192",
        "positive_styles": [], "negative_styles": [], "lyrics": "", "language": "auto",
        "force_instrumental": False, "finetune_id": None,
    },
    "motion": {"style": "groove", "intensity": 0.2},
}

_ALLOWED_KEYS = {
    "voice": {"backend", "model", "voice", "language"},
    "music": {"model_id", "duration_ms", "output_format", "positive_styles", "negative_styles", "lyrics", "language", "force_instrumental", "finetune_id"},
    "motion": {"style", "intensity"},
}
_ALLOWED_FORMATS = {
    "auto", "mp3_48000_128", "mp3_48000_192", "mp3_48000_240", "mp3_48000_320",
    "mp3_22050_32", "mp3_24000_48", "mp3_44100_32", "mp3_44100_64", "mp3_44100_96",
    "mp3_44100_128", "mp3_44100_192", "pcm_8000", "pcm_16000", "pcm_22050", "pcm_24000",
    "pcm_32000", "pcm_44100", "pcm_48000", "ulaw_8000", "alaw_8000", "opus_48000_32",
    "opus_48000_64", "opus_48000_96", "opus_48000_128", "opus_48000_192",
}


def _reject_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if any(secret_word in str(key).lower() for secret_word in ("key", "token", "secret", "credential", "password")):
                raise ValueError("credentials are not accepted by runtime settings")
            _reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item)


def _validate(value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if set(value) != set(_ALLOWED_KEYS):
        raise ValueError("settings must contain voice, music, and motion groups")
    for group, allowed in _ALLOWED_KEYS.items():
        settings = value[group]
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise ValueError(f"invalid fields in {group} settings")
    voice, music, motion = value["voice"], value["music"], value["motion"]
    if voice["backend"] not in ("subscription", "api_key"):
        raise ValueError("voice.backend must be subscription or api_key")
    for field, limit in (("model", 100), ("voice", 32), ("language", 64)):
        text = voice[field]
        if not isinstance(text, str) or not text.strip() or len(text) > limit:
            raise ValueError(f"voice.{field} is invalid")
    if music["model_id"] not in ("music_v1", "music_v2", "music_v2_5"):
        raise ValueError("music.model_id is unsupported")
    duration = music["duration_ms"]
    if isinstance(duration, bool) or not isinstance(duration, int) or not 3000 <= duration <= 600000:
        raise ValueError("music.duration_ms must be between 3000 and 600000")
    if music["output_format"] not in _ALLOWED_FORMATS:
        raise ValueError("music.output_format is unsupported")
    for field in ("positive_styles", "negative_styles"):
        styles = music[field]
        if not isinstance(styles, list) or len(styles) > 50 or any(not isinstance(item, str) or len(item) > 200 for item in styles):
            raise ValueError(f"music.{field} must be a list of at most 50 short strings")
    for field in ("lyrics", "language"):
        text = music[field]
        if not isinstance(text, str) or len(text) > 6000:
            raise ValueError(f"music.{field} is too long or invalid")
    if not isinstance(music["force_instrumental"], bool):
        raise ValueError("music.force_instrumental must be boolean")
    if music["force_instrumental"] and music["lyrics"].strip():
        raise ValueError("instrumental music cannot include original lyrics")
    finetune = music["finetune_id"]
    if finetune is not None and (not isinstance(finetune, str) or len(finetune) > 128):
        raise ValueError("music.finetune_id is invalid")
    if motion["style"] not in ("groove", "bounce", "smooth", "classic"):
        raise ValueError("motion.style is unsupported")
    intensity = motion["intensity"]
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)) or not 0 <= intensity <= 0.25:
        raise ValueError("motion.intensity must be between 0 and 0.25")
    return value


class RuntimeSettingsStore:
    """Strict JSON settings persisted with an atomic same-directory replace."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> dict[str, dict[str, Any]]:
        settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        try:
            disk = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return settings
        if not isinstance(disk, dict):
            raise ValueError("runtime settings file must contain an object")
        for group, values in disk.items():
            if group not in settings or not isinstance(values, dict):
                raise ValueError("runtime settings file has an unknown group")
            settings[group].update(values)
        _reject_secrets(settings)
        return _validate(settings)

    def update(self, updates: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if not isinstance(updates, dict) or set(updates) - set(DEFAULT_SETTINGS):
            raise ValueError("only voice, music, and motion settings may be changed")
        _reject_secrets(updates)
        current = self.read()
        merged = json.loads(json.dumps(current))
        for group, values in updates.items():
            if not isinstance(values, dict):
                raise ValueError(f"{group} settings must be an object")
            merged[group].update(values)
        validated = _validate(merged)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(validated, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        return validated


def register_runtime_settings_routes(app: Any, path: Path) -> None:
    """Expose validated nonsecret settings and user-owned persona operations."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    store = RuntimeSettingsStore(path)
    persona_path = path.parent / "instructions.md"

    @app.get("/api/dj-reachy/settings")
    async def get_runtime_settings() -> dict[str, Any]:
        return store.read()

    @app.put("/api/dj-reachy/settings")
    async def put_runtime_settings(request: Request) -> Any:
        try:
            body = await request.json()
            from dj_reachy.config import apply_runtime_settings, config
            settings = store.update(body)
            if store.path == config.RUNTIME_SETTINGS_PATH:
                apply_runtime_settings(settings)
            return settings
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/dj-reachy/instructions")
    async def get_runtime_instructions() -> dict[str, str]:
        return {"instructions": persona_path.read_text(encoding="utf-8") if persona_path.exists() else ""}

    @app.put("/api/dj-reachy/instructions")
    async def put_runtime_instructions(request: Request) -> Any:
        try:
            body = await request.json()
            text = body.get("instructions") if isinstance(body, dict) else None
            if not isinstance(text, str) or len(text) > 12000:
                raise ValueError("instructions must be a string of at most 12000 characters")
            persona_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=".instructions.", suffix=".tmp", dir=persona_path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, persona_path)
            finally:
                Path(temp_name).unlink(missing_ok=True)
            return {"status": "saved", "reload": "Reload instructions from the app settings page or reconnect voice."}
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
