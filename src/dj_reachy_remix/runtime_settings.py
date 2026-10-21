"""User-owned runtime settings and narrowly scoped app-manager routes."""

from __future__ import annotations
import os
import re
import json
import tempfile
import threading
from typing import Any
from pathlib import Path

from fastapi import FastAPI, Request


_STORE_LOCK = threading.RLock()


def atomic_write_private(path: Path, text: str) -> None:
    """Replace a user-owned file without following a destination symlink."""
    if path.is_symlink():
        raise ValueError("refusing to replace a symbolic link")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def save_instructions(path: Path, text: Any) -> None:
    """Validate the whole persona before replacing the previous value."""
    if not isinstance(text, str) or len(text) > 12000:
        raise ValueError("instructions must be a string of at most 12000 characters")
    atomic_write_private(path, text)


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
    "mp3_48000_128", "mp3_48000_192", "mp3_48000_240", "mp3_48000_320",
    "mp3_22050_32", "mp3_24000_48", "mp3_44100_32", "mp3_44100_64",
    "mp3_44100_96", "mp3_44100_128", "mp3_44100_192",
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
    if voice["backend"] == "api_key" and (
        voice["model"].startswith("gpt-live") or voice["voice"] == "cove"
    ):
        raise ValueError("subscription model/voice cannot be used with API-key billing")
    if voice["backend"] == "subscription" and (
        not voice["model"].startswith("gpt-live") or voice["voice"] != "cove"
    ):
        raise ValueError("subscription requires a qualified gpt-live model and cove voice")
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

    def __init__(self, path: Path) -> None:
        """Bind one user-owned JSON settings file."""
        self.path = Path(path)

    def read(self) -> dict[str, dict[str, Any]]:
        """Read validated settings, refusing a redirected settings file."""
        if self.path.is_symlink():
            raise ValueError("runtime settings cannot be a symbolic link")
        settings: dict[str, dict[str, Any]] = json.loads(json.dumps(DEFAULT_SETTINGS))
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
        """Serialize read/validate/replace so concurrent HTTP updates don't lose groups."""
        with _STORE_LOCK:
            try:
                return self._update(updates)
            except TypeError as exc:
                raise ValueError("settings values have invalid types") from exc

    def _update(self, updates: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if not isinstance(updates, dict) or set(updates) - set(DEFAULT_SETTINGS):
            raise ValueError("only voice, music, and motion settings may be changed")
        _reject_secrets(updates)
        current = self.read()
        merged = json.loads(json.dumps(current))
        for group, values in updates.items():
            if not isinstance(values, dict):
                raise ValueError(f"{group} settings must be an object")
            if group == "voice" and values.get("backend", current["voice"]["backend"]) != current["voice"]["backend"]:
                backend = values["backend"]
                merged["voice"].update(
                    {"model": "gpt-realtime", "voice": "cedar"} if backend == "api_key"
                    else {"model": "gpt-live-1-codex", "voice": "cove"}
                )
            merged[group].update(values)
        validated = _validate(merged)
        atomic_write_private(self.path, json.dumps(validated, ensure_ascii=False, indent=2) + "\n")
        return validated


def validate_generation_request(arguments: dict[str, Any], settings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Validate UI input through the same offline builder used by music tools."""
    from inspect import signature

    from dj_reachy_remix.profiles.dj_reachy.make_song_and_dance import build_music_payload

    allowed = _ALLOWED_KEYS["music"] | {"prompt", "composition_plan", "music_length_ms"}
    if set(arguments) - allowed or "duration_ms" in arguments:
        raise ValueError("unknown music fields; use music_length_ms")
    options = dict(settings["music"])
    options.update({key: value for key, value in arguments.items() if key in options})
    options["duration_ms"] = arguments.get("music_length_ms", options["duration_ms"])
    full = {**settings, "music": options}
    _validate(full)
    prompt, plan = arguments.get("prompt"), arguments.get("composition_plan")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 6000):
        raise ValueError("prompt must be nonempty text, up to 6000 characters")
    if plan is not None:
        if not isinstance(plan, dict):
            raise ValueError("composition_plan must be an object")
        items = plan.get("sections" if options["model_id"] == "music_v1" else "chunks")
        if not isinstance(items, list) or not items or not all(isinstance(item, dict) for item in items):
            raise ValueError("plan must contain versioned sections or chunks")
    kwargs = {key: options[key] for key in
              ("model_id", "positive_styles", "negative_styles", "lyrics", "language", "force_instrumental")}
    finetune = options["finetune_id"]
    if finetune:
        if "finetune_id" not in signature(build_music_payload).parameters:
            raise ValueError("this music implementation does not support qualified finetune_id")
        kwargs["finetune_id"] = finetune
    build_music_payload(prompt=prompt, composition_plan=plan, music_length_ms=options["duration_ms"], **kwargs)
    result = {**kwargs, "music_length_ms": options["duration_ms"], "output_format": options["output_format"]}
    result["composition_plan" if plan is not None else "prompt"] = plan if plan is not None else prompt
    return result


def register_runtime_settings_routes(app: FastAPI, path: Path) -> None:
    """Expose validated nonsecret settings and user-owned persona operations."""
    # Request must be module-global for FastAPI's postponed annotation resolution.
    from fastapi.responses import JSONResponse

    if getattr(app.state, "dj_settings_registered", False):
        return
    app.state.dj_settings_registered = True

    store = RuntimeSettingsStore(path)
    persona_path = path.parent / "instructions.md"

    @app.get("/api/dj-reachy-remix/settings")
    async def get_runtime_settings() -> Any:
        try:
            return store.read()
        except (ValueError, TypeError):
            return JSONResponse({"error": "The settings file is invalid. Correct its voice/music/motion values."}, status_code=400)

    @app.put("/api/dj-reachy-remix/settings")
    async def put_runtime_settings(request: Request) -> Any:
        try:
            body = await request.json()
            from dj_reachy_remix.config import config, apply_runtime_settings
            settings = store.update(body)
            if store.path == config.RUNTIME_SETTINGS_PATH:
                apply_runtime_settings(settings)
            return settings
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/dj-reachy-remix/instructions")
    async def get_runtime_instructions() -> Any:
        if persona_path.is_symlink():
            return JSONResponse({"error": "Instructions cannot be a symbolic link."}, status_code=400)
        return {"instructions": persona_path.read_text(encoding="utf-8") if persona_path.exists() else ""}

    @app.put("/api/dj-reachy-remix/instructions")
    async def put_runtime_instructions(request: Request) -> Any:
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"instructions"}:
                raise ValueError("provide exactly an instructions string")
            save_instructions(persona_path, body["instructions"])
            return {"status": "saved", "reconnect_required": True}
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    from fastapi.responses import FileResponse

    @app.get("/api/dj-reachy-remix/status")
    async def runtime_status() -> dict[str, Any]:
        from dj_reachy_remix.config import config
        return {
            "backend": config.VOICE_BACKEND,
            "billing": "ChatGPT subscription" if config.VOICE_BACKEND == "subscription" else "OpenAI API key",
            "openai_key_configured": bool(config.OPENAI_API_KEY),
            "elevenlabs_key_configured": bool(config.ELEVENLABS_API_KEY),
            "provider_access": "unverified",
            "voice_changes": "Reconnect voice after changing conversation settings or instructions.",
            "browser_voice_available": hasattr(app.state, "dj_voice_stream"),
            **(app.state.dj_voice_status() if hasattr(app.state, "dj_voice_status") else
               {"voice_state": "not_connected", "voice_error": None, "voice_reconnect_available": False}),
        }

    @app.post("/api/dj-reachy-remix/voice/reconnect")
    async def reconnect_voice(request: Request) -> Any:
        # A POST is deliberate: status polling and settings saves never connect.
        reconnect = getattr(app.state, "dj_voice_reconnect", None)
        if reconnect is None:
            return JSONResponse({"error": "Local voice reconnect is unavailable."}, status_code=409)
        try:
            if await request.json() != {}:
                raise ValueError("Reconnect takes no options")
            return JSONResponse(await reconnect(), status_code=202)
        except (ValueError, TypeError):
            return JSONResponse({"error": "Reconnect requires an empty object."}, status_code=400)
        except RuntimeError:
            return JSONResponse({"error": "Voice is active, stopping, or unavailable. Wait for offline status."}, status_code=409)

    @app.post("/api/dj-reachy-remix/credentials")
    async def save_credential(request: Request) -> Any:
        from dotenv import dotenv_values

        from dj_reachy_remix.config import config
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"provider", "key"}:
                raise ValueError("provide exactly provider and key")
            provider, key = body["provider"], body["key"]
            if provider not in ("openai", "elevenlabs"):
                raise ValueError("provider must be openai or elevenlabs")
            if not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,512}", key) is None:
                raise ValueError("invalid key")
            env_path = path.parent / ".env"
            if env_path.is_symlink():
                raise ValueError("credential file cannot be a symbolic link")
            with _STORE_LOCK:
                existing = dotenv_values(env_path, interpolate=False) if env_path.exists() else {}
                name = "OPENAI_API_KEY" if provider == "openai" else "ELEVENLABS_API_KEY"
                existing[name] = key
                text = "".join(f"{field}={json.dumps(existing[field])}\n"
                               for field in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY") if existing.get(field))
                atomic_write_private(env_path, text)
            os.environ[name] = key
            setattr(config, name, key)
            dependencies = getattr(app.state, "dj_dependencies", None)
            if provider == "elevenlabs" and dependencies is not None:
                dependencies.elevenlabs_api_key = key
            return {"status": "saved", "reconnect_required": provider == "openai"}
        except (ValueError, TypeError):
            return JSONResponse({"error": "Invalid credential request. Use provider openai or elevenlabs and a single key."}, status_code=400)
        except OSError:
            return JSONResponse({"error": "Could not persist credentials under the user app home."}, status_code=503)

    def library() -> Any:
        from dj_reachy_remix.gradio_dj_ui import DJReachyUI
        return DJReachyUI()

    @app.get("/songs")
    async def list_saved_songs() -> dict[str, Any]:
        return {"songs": [{"filename": row[0], "title": row[1], "size": row[2]}
                          for row in library()._list_songs() if row[0]]}

    @app.get("/songs/{filename}")
    async def get_saved_song(filename: str) -> Any:
        song = library()._get_song_path(filename)
        if song is None:
            return JSONResponse({"error": "Saved track not found."}, status_code=404)
        return FileResponse(song, media_type="audio/mpeg", filename=filename)

    @app.delete("/songs/{filename}")
    async def delete_saved_song(filename: str) -> Any:
        ui = library()
        if ui._get_song_path(filename) is None:
            return JSONResponse({"error": "Saved track not found."}, status_code=404)
        _, status = ui._delete_song(filename)
        if ui._get_song_path(filename) is not None:
            return JSONResponse({"error": status}, status_code=409)
        return {"status": "deleted"}

    @app.post("/api/dj-reachy-remix/music/{action}")
    async def music_action(action: str, request: Request) -> Any:
        tools = {"generate": "make_song_and_dance", "replay": "play_saved_song",
                 "pause": "pause_music", "resume": "resume_music", "stop": "stop_music"}
        if action not in tools:
            return JSONResponse({"error": "Unknown music action."}, status_code=404)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("music arguments must be an object")
            if action in ("pause", "resume", "stop") and body:
                raise ValueError("playback controls do not accept arguments")
            if action == "replay" and (set(body) != {"filename"} or not isinstance(body["filename"], str)
                                       or library()._get_song_path(body["filename"]) is None):
                raise ValueError("choose a regular saved track by filename")
            if action == "generate":
                body = validate_generation_request(body, store.read())
            invoke = getattr(app.state, "dj_invoke", None)
            if invoke is None:
                return JSONResponse({"error": "Connect the audio session before using playback."}, status_code=409)
            result = await invoke(tools[action], body)
            return {key: result[key] for key in ("status", "message", "error", "regenerated")
                    if key in result and isinstance(result[key], (str, bool))}
        except (ValueError, TypeError, KeyError):
            return JSONResponse({"error": "Invalid music request. Check the model, duration, prompt/plan and settings."}, status_code=400)
        except RuntimeError:
            return JSONResponse({"error": "Audio session is not running. Connect voice first."}, status_code=409)


def register_settings_page(app: FastAPI, path: Path) -> None:
    """Mount the wheel-packaged control panel on CLI and app-manager apps."""
    from fastapi.responses import FileResponse
    from starlette.staticfiles import StaticFiles
    register_runtime_settings_routes(app, path)
    static = Path(__file__).parent / "static"
    if not any(getattr(route, "path", None) == "/static" for route in app.routes):
        app.mount("/static", StaticFiles(directory=static), name="dj-static")
    if not any(getattr(route, "path", None) == "/" for route in app.routes):
        @app.get("/")
        async def settings_page() -> Any:
            return FileResponse(static / "index.html")
