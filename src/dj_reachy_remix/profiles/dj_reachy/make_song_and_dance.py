import os
import re
import sys
import json
import stat
import time
import shutil
import signal
import asyncio
import logging
import threading
from uuid import uuid4
from typing import Any, Dict, Optional, cast
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Callable, Awaitable

import httpx
import numpy as np

from dj_reachy_remix.tools.core_tools import Tool, ToolDependencies
from dj_reachy_remix.audio.music_dancer import DanceStyle, MusicDancer


_playback_pause: asyncio.Event | None = None
_playback_stop: asyncio.Event | None = None
_generation_cancel: threading.Event | None = None
_active_decoder: asyncio.subprocess.Process | None = None
_playback_owner: str | None = None
_playback_motion_lock: asyncio.Lock | None = None
_playback_done: asyncio.Event | None = None


logger = logging.getLogger(__name__)

# Module-level music dancer instance (lazily initialized)
_music_dancer: Optional[MusicDancer] = None
_music_dancer_deps: Optional[ToolDependencies] = None


def _get_music_dancer(deps: ToolDependencies) -> MusicDancer:
    """Get or create the module-level MusicDancer instance."""
    global _music_dancer, _music_dancer_deps
    from dj_reachy_remix.config import config

    if _music_dancer is None or _music_dancer_deps is not deps:
        _music_dancer = MusicDancer(
            set_offsets_callback=deps.movement_manager.set_music_dance_offsets,
            sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            intensity=float(getattr(config, "MOTION_INTENSITY", 0.2)),
        )
        _music_dancer._controller.style = _dance_style(getattr(config, "MOTION_STYLE", "groove"))
        _music_dancer_deps = deps
        logger.info("Created new MusicDancer instance")
    return _music_dancer


def get_dance_intensity() -> float:
    """Get current dance intensity for the UI."""
    if _music_dancer is not None:
        return _music_dancer.intensity
    from dj_reachy_remix.config import config

    return float(getattr(config, "MOTION_INTENSITY", 0.2))


def set_dance_intensity(intensity: float) -> None:
    """Set dance intensity for the UI."""
    if _music_dancer is not None:
        _music_dancer.intensity = max(0.0, min(0.25, intensity))
        logger.info("Dance intensity updated")


def _dance_style(name: str) -> DanceStyle:
    presets = {
        "groove": {
            "head_bob_amplitude": 15.0,
            "head_bob_speed": 1.0,
            "body_sway_amplitude": 70.0,
            "body_sway_speed": 1.0,
            "antenna_amplitude": 0.8,
            "emphasis_strength": 25.0,
            "movement_smoothing": 0.25,
        },
        "bounce": {
            "head_bob_amplitude": 22.0,
            "head_bob_speed": 1.3,
            "body_sway_amplitude": 55.0,
            "body_sway_speed": 1.2,
            "antenna_amplitude": 0.9,
            "emphasis_strength": 28.0,
            "movement_smoothing": 0.2,
        },
        "smooth": {
            "head_bob_amplitude": 9.0,
            "head_bob_speed": 0.7,
            "body_sway_amplitude": 38.0,
            "body_sway_speed": 0.7,
            "antenna_amplitude": 0.5,
            "emphasis_strength": 12.0,
            "movement_smoothing": 0.55,
        },
        "classic": {
            "head_bob_amplitude": 13.0,
            "head_bob_speed": 0.9,
            "body_sway_amplitude": 58.0,
            "body_sway_speed": 0.9,
            "antenna_amplitude": 0.65,
            "emphasis_strength": 20.0,
            "movement_smoothing": 0.32,
        },
    }
    return DanceStyle(**presets.get(name, presets["groove"]))


def set_dance_style(style: str) -> None:
    """Apply one allow-listed motion preset without changing controller bounds."""
    if _music_dancer is not None:
        _music_dancer._controller.style = _dance_style(style)


ELEVEN_MUSIC_URL = "https://api.elevenlabs.io/v1/music"
SUPPORTED_MUSIC_MODELS = {"music_v1", "music_v2", "music_v2_5"}
OUTPUT_FORMATS = {
    "auto",
    "mp3_48000_128",
    "mp3_48000_192",
    "mp3_48000_240",
    "mp3_48000_320",
    "mp3_22050_32",
    "mp3_24000_48",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",
}


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _styles(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > 50
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{name} must be a list of at most 50 nonempty strings")
    return value


def _lyrics_lines(value: Any) -> None:
    if (
        not isinstance(value, list)
        or len(value) > 30
        or any(not isinstance(line, str) or len(line) > 200 for line in value)
    ):
        raise ValueError("lyrics must contain at most 30 lines of 200 characters")


def _time_range(value: Any) -> int:
    if not isinstance(value, dict) or set(value) != {"start_ms", "end_ms"}:
        raise ValueError("audio ranges require start_ms and end_ms")
    start = _integer(value["start_ms"], "start_ms", 0, 600000)
    end = _integer(value["end_ms"], "end_ms", 1, 600000)
    if end <= start:
        raise ValueError("audio range end_ms must exceed start_ms")
    return end - start


def _audio_ref(value: Any, *, source: bool = False) -> None:
    if value is None:
        return
    allowed = {"song_id", "range"} | ({"negative_ranges"} if source else set())
    if not isinstance(value, dict) or not {"song_id", "range"} <= set(value) or set(value) - allowed:
        raise ValueError("audio references require song_id and range only")
    if not isinstance(value["song_id"], str) or not value["song_id"].strip():
        raise ValueError("song_id must be a nonempty string")
    duration = _time_range(value["range"])
    if not source and duration > 30000:
        raise ValueError("conditioning reference cannot exceed 30000 ms")
    if "negative_ranges" in value:
        if not isinstance(value["negative_ranges"], list):
            raise ValueError("negative_ranges must be a list")
        for item in value["negative_ranges"]:
            _time_range(item)


def _validate_plan(plan: Any, model_id: str) -> int:
    if not isinstance(plan, dict):
        raise ValueError("composition_plan must be an object")
    v1 = model_id == "music_v1"
    keys = {"sections", "positive_global_styles", "negative_global_styles"} if v1 else {"chunks"}
    if set(plan) != keys:
        raise ValueError(f"{model_id} requires {'sections and global styles' if v1 else 'chunks only'}")
    if v1:
        _styles(plan["positive_global_styles"], "positive_global_styles")
        _styles(plan["negative_global_styles"], "negative_global_styles")
    items = plan["sections" if v1 else "chunks"]
    if not isinstance(items, list) or not 1 <= len(items) <= 30:
        raise ValueError("composition plan requires 1 to 30 sections/chunks")
    total = 0
    for item in items:
        required = (
            {"section_name", "duration_ms", "positive_local_styles", "negative_local_styles", "lines"}
            if v1
            else {"text", "duration_ms", "positive_styles"}
        )
        allowed = required | (
            {"source_from"}
            if v1
            else {"negative_styles", "context_adherence", "conditioning_ref", "condition_strength"}
        )
        if not isinstance(item, dict) or not required <= set(item) or set(item) - allowed:
            raise ValueError("composition plan has missing or unsupported fields")
        total += _integer(item["duration_ms"], "duration_ms", 3000, 120000)
        if v1:
            if not isinstance(item["section_name"], str) or not 1 <= len(item["section_name"]) <= 100:
                raise ValueError("section_name must contain 1 to 100 characters")
            _styles(item["positive_local_styles"], "positive_local_styles")
            _styles(item["negative_local_styles"], "negative_local_styles")
            _lyrics_lines(item["lines"])
            _audio_ref(item.get("source_from"), source=True)
        else:
            if not isinstance(item["text"], str):
                raise ValueError("chunk text must be a string")
            _lyrics_lines(item["text"].splitlines())
            if item["text"].startswith("["):
                section = item["text"].split("]", 1)
                if len(section) != 2 or not 1 <= len(section[0][1:]) <= 100:
                    raise ValueError("chunk section name must contain 1 to 100 characters")
            _styles(item["positive_styles"], "positive_styles")
            _styles(item.get("negative_styles", []), "negative_styles")
            if item.get("context_adherence", "high") not in ("low", "medium", "high"):
                raise ValueError("invalid context_adherence")
            if item.get("condition_strength") not in (None, "low", "medium", "high", "xhigh"):
                raise ValueError("invalid condition_strength")
            _audio_ref(item.get("conditioning_ref"))
    _integer(total, "plan total duration", 3000, 600000)
    return total


def build_music_payload(
    *,
    prompt: str | None,
    music_length_ms: int,
    model_id: str = "music_v2_5",
    composition_plan: dict[str, Any] | None = None,
    force_instrumental: bool = False,
    positive_styles: list[str] | None = None,
    negative_styles: list[str] | None = None,
    lyrics: str | None = None,
    language: str | None = None,
    finetune_id: str | None = None,
) -> dict[str, Any]:
    """Validate and build a request using only documented compose-body fields."""
    if not isinstance(model_id, str) or model_id not in SUPPORTED_MUSIC_MODELS:
        raise ValueError("Unsupported Eleven Music model")
    _integer(music_length_ms, "music_length_ms", 3000, 600000)
    if type(force_instrumental) is not bool:
        raise ValueError("force_instrumental must be a boolean")
    if finetune_id is not None and (
        not isinstance(finetune_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", finetune_id)
    ):
        raise ValueError("finetune_id must be an existing Music finetune identifier")
    extra = {"finetune_id": finetune_id} if finetune_id is not None else {}
    for value, name in ((lyrics, "lyrics"), (language, "language")):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
    if lyrics:
        _lyrics_lines(lyrics.splitlines())
    if language and ("\n" in language or len(language) > 100):
        raise ValueError("language must be one line of at most 100 characters")
    for styles, name in ((positive_styles, "positive_styles"), (negative_styles, "negative_styles")):
        if styles is not None:
            _styles(styles, name)
    if (prompt is None) == (composition_plan is None):
        raise ValueError("provide exactly one of prompt or composition plan")
    if composition_plan is not None and force_instrumental:
        raise ValueError("force_instrumental is only valid with prompt mode, not a composition plan")
    if force_instrumental and lyrics:
        raise ValueError("instrumental music cannot include original lyrics")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError("prompt must be a nonempty string")

    if prompt is not None:
        lines = [prompt.strip()]
        if positive_styles:
            lines.append("Musical styles and instrumentation: " + ", ".join(positive_styles))
        if negative_styles:
            lines.append("Avoid: " + ", ".join(negative_styles))
        if language:
            lines.append(f"Lyrics language: {language}")
        if lyrics:
            lines.append("Use these original lyrics:\n" + lyrics)
        return {
            "model_id": model_id,
            "prompt": "\n\n".join(lines),
            "music_length_ms": music_length_ms,
            "force_instrumental": bool(force_instrumental),
            **extra,
        }

    _validate_plan(composition_plan, model_id)
    plan = json.loads(json.dumps(composition_plan))
    if model_id == "music_v1":
        plan["positive_global_styles"] = list(dict.fromkeys(plan["positive_global_styles"] + (positive_styles or [])))
        plan["negative_global_styles"] = list(dict.fromkeys(plan["negative_global_styles"] + (negative_styles or [])))
        first = plan["sections"][0]
        if lyrics:
            first["lines"].extend(lyrics.splitlines())
        if language:
            first["positive_local_styles"].append(f"Lyrics language: {language}")
    else:
        first = plan["chunks"][0]
        first["positive_styles"] = list(dict.fromkeys(first["positive_styles"] + (positive_styles or [])))
        if negative_styles:
            first["negative_styles"] = list(dict.fromkeys(first.get("negative_styles", []) + negative_styles))
        if lyrics:
            first["text"] = first["text"] + "\n" + lyrics
        if language:
            first["positive_styles"].append(f"Lyrics language: {language}")
    total_duration = _validate_plan(plan, model_id)
    if total_duration != music_length_ms:
        raise ValueError("composition plan total duration must match music_length_ms")
    return {"model_id": model_id, "composition_plan": plan, **extra}


ELEVEN_COMPOSE_URL = ELEVEN_MUSIC_URL


def _get_elevenlabs_key(deps: ToolDependencies | None = None) -> str:
    """Get ElevenLabs API key from deps, config (set via UI), or environment."""
    key = deps.elevenlabs_api_key if deps is not None else None
    from dj_reachy_remix.config import config

    key = key or config.ELEVENLABS_API_KEY or os.environ.get("ELEVENLABS_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError("Missing ELEVENLABS_API_KEY. Configure it in the app settings.")
    return key


def _eleven_headers(deps: ToolDependencies | None = None) -> Dict[str, str]:
    return {"xi-api-key": _get_elevenlabs_key(deps), "Content-Type": "application/json", "Accept": "audio/mpeg"}


class LocalGenerationCancelled(RuntimeError):
    """Local cancellation won; upstream provider work may have continued."""


async def _compose_music(
    prompt: str | None,
    timeout_s: int = 180,
    deps: ToolDependencies | None = None,
    *,
    music_length_ms: int = 30000,
    model_id: str = "music_v2_5",
    composition_plan: dict[str, Any] | None = None,
    force_instrumental: bool = False,
    positive_styles: list[str] | None = None,
    negative_styles: list[str] | None = None,
    lyrics: str | None = None,
    language: str | None = None,
    cancel_event: threading.Event | None = None,
    output_format: str = "mp3_48000_192",
    finetune_id: str | None = None,
) -> str:
    """Generate a track and atomically commit it to the user data directory."""
    from dj_reachy_remix.config import config

    payload = build_music_payload(
        prompt=prompt,
        music_length_ms=music_length_ms,
        model_id=model_id,
        composition_plan=composition_plan,
        force_instrumental=force_instrumental,
        positive_styles=positive_styles,
        negative_styles=negative_styles,
        lyrics=lyrics,
        language=language,
        finetune_id=finetune_id,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise LocalGenerationCancelled("Local result discarded; provider work may have continued.")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported Eleven Music output_format: {output_format}")
    headers = _eleven_headers(deps)
    async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as client:
        request = asyncio.create_task(
            client.post(ELEVEN_MUSIC_URL, params={"output_format": output_format}, json=payload, headers=headers)
        )
        try:
            deadline = asyncio.get_running_loop().time() + timeout_s
            while not request.done():
                if cancel_event is not None and cancel_event.is_set():
                    raise LocalGenerationCancelled("Local HTTP request cancelled; provider work may have continued.")
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError("ElevenLabs music request timed out")
                await asyncio.wait({request}, timeout=0.02)
            try:
                response = request.result()
            except httpx.HTTPError:
                raise RuntimeError("ElevenLabs music HTTP transport failed") from None
        finally:
            if not request.done():
                request.cancel()
            await asyncio.gather(request, return_exceptions=True)
    if cancel_event is not None and cancel_event.is_set():
        raise LocalGenerationCancelled("Local result discarded; provider work may have continued.")
    if response.status_code >= 400:
        raise RuntimeError(f"ElevenLabs music request failed with HTTP {response.status_code}")
    if response.headers.get("Content-Type", "").split(";", 1)[0].lower() not in ("audio/mpeg", "audio/mp3"):
        raise RuntimeError("ElevenLabs music request returned a non-MP3 response")
    songs_dir = Path(config.SONGS_DIR)
    if songs_dir.is_symlink():
        raise ValueError("Music library must not be a symlink")
    songs_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    title = prompt or "planned_track"
    safe_prompt = re.sub(r"[^a-zA-Z0-9_-]", "_", title[:40]).strip("_") or "track"
    track_id = uuid4().hex
    path = songs_dir / f"{stamp}_{track_id}_{safe_prompt}.mp3"
    temp_path = path.with_suffix(".mp3.part")
    metadata_path = path.with_suffix(".json")
    metadata_temp = metadata_path.with_suffix(".json.part")
    selected_format = (
        output_format if output_format != "auto" else ("mp3_44100_128" if model_id == "music_v1" else "mp3_48000_192")
    )
    song_id = response.headers.get("song-id")
    if song_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", song_id):
        song_id = None
    metadata = {
        "id": track_id,
        "title": title,
        "created_at_ms": stamp,
        "provider": "elevenlabs",
        "request": payload,
        "output_format": selected_format,
        "requested_output_format": output_format,
        "provider_song_id": song_id,
    }
    committed = False
    try:
        with temp_path.open("xb") as audio:
            audio.write(response.content)
            audio.flush()
            os.fsync(audio.fileno())
        if temp_path.stat().st_size < 1024:
            raise RuntimeError("ElevenLabs returned an unexpectedly small audio file")
        with metadata_temp.open("x", encoding="utf-8") as meta:
            json.dump(metadata, meta, ensure_ascii=False)
            meta.flush()
            os.fsync(meta.fileno())
        if cancel_event is not None and cancel_event.is_set():
            raise LocalGenerationCancelled("Local generated file was not committed.")
        # Audio is the library visibility marker. Publish its metadata first.
        os.replace(metadata_temp, metadata_path)
        os.replace(temp_path, path)
        if cancel_event is not None and cancel_event.is_set():
            raise LocalGenerationCancelled("Local generated file was not committed.")
        committed = True
    finally:
        temp_path.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
        if not committed:
            path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
    return str(path)


def _compose_music_to_mp3(
    prompt: str | None, timeout_s: int = 180, deps: ToolDependencies | None = None, **options: Any
) -> str:
    """Compose synchronously for callers outside an event loop."""
    return asyncio.run(_compose_music(prompt, timeout_s, deps, **options))


LIST_SAVED_SONGS_LIMIT = 20
LIST_SAVED_TITLE_CHARS = 160

OPEN_AI_OUTPUT_SAMPLE_RATE = 24000  # must match openai_realtime.py

# Chunk size: 960 samples @ 24kHz = 40ms of audio
CHUNK_SAMPLES = 960
CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16 = 2 bytes per sample
CHUNK_DURATION_S = CHUNK_SAMPLES / OPEN_AI_OUTPUT_SAMPLE_RATE  # 0.04 seconds


def _discard_local_track(path: str) -> None:
    """Remove only the exact generated pair after a late local cancellation."""
    for candidate in (path, os.path.splitext(path)[0] + ".json"):
        try:
            os.unlink(candidate)
        except FileNotFoundError:
            pass


async def _drain_pipe(stream: asyncio.StreamReader | None) -> None:
    if stream is not None:
        while await stream.read(16384):
            pass


async def _reap_decoder(proc: asyncio.subprocess.Process, stderr_task: asyncio.Task[None]) -> None:
    if proc.returncode is None:
        try:
            proc.send_signal(signal.SIGCONT)
            proc.terminate()
        except ProcessLookupError:
            pass
    drained = asyncio.gather(proc.wait(), _drain_pipe(proc.stdout), stderr_task)
    try:
        await asyncio.wait_for(asyncio.shield(drained), 1)
    except asyncio.TimeoutError:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await asyncio.wait_for(drained, 1)


async def _play_mp3(
    path: str, deps: ToolDependencies, cancel_event: threading.Event | None = None, *, source_fd: int | None = None
) -> bool:
    """Stream one cancellable, pausable decoder into the audio handler."""
    global _playback_pause, _playback_stop, _active_decoder, _playback_owner, _playback_motion_lock
    global _playback_done
    handler = getattr(deps, "openai_realtime_handler", None)
    if handler is None:
        raise RuntimeError("Realtime handler is not connected.")
    if _playback_pause is not None:
        raise RuntimeError("A music playback session is already active.")
    dancer = _get_music_dancer(deps)
    owner = f"music-{uuid4().hex}"
    pause_event = asyncio.Event()
    pause_event.set()
    stop_event = asyncio.Event()
    _playback_pause, _playback_stop = pause_event, stop_event
    _playback_owner = owner
    motion_lock = asyncio.Lock()
    _playback_motion_lock = motion_lock
    done_event = asyncio.Event()
    _playback_done = done_event
    proc: asyncio.subprocess.Process | None = None
    stderr_task: asyncio.Task[None] | None = None
    output_drained = False
    try:
        handler.mute_output_audio(owner)
        executable = shutil.which("mpg123")
        command = (
            [executable, "-q", "-s", "--mono", "--rate", str(OPEN_AI_OUTPUT_SAMPLE_RATE), path]
            if executable else [sys.executable, "-m", "dj_reachy_remix.audio.decode_mp3", path]
        )
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            pass_fds=(source_fd,) if source_fd is not None else (),
        )
        _active_decoder = proc
        assert proc.stdout is not None
        stderr_task = asyncio.create_task(_drain_pipe(proc.stderr))
        dancer.start()
        loop = asyncio.get_running_loop()
        pacing_deadline: float | None = None
        data = b""
        while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
            if not pause_event.is_set():
                # Consumer-side backpressure holds the decoder's bounded read-ahead.
                # No SIGSTOP: a stopped process cannot handle SIGTERM during shutdown.
                pacing_deadline = None
                await asyncio.sleep(0.02)
                continue
            if not data:
                try:
                    data = await asyncio.wait_for(proc.stdout.readexactly(CHUNK_BYTES), 0.1)
                except asyncio.TimeoutError:
                    continue
                except asyncio.IncompleteReadError as exc:
                    data = exc.partial
                    if not data:
                        break
            async with motion_lock:
                if not pause_event.is_set() or stop_event.is_set():
                    continue
                if len(data) % 2:
                    raise RuntimeError("mpg123 returned incomplete PCM samples")
                pcm = np.frombuffer(data, dtype=np.int16)
                if pacing_deadline is None:
                    pacing_deadline = loop.time()
                dancer.feed(pcm)
                await handler.output_queue.put((OPEN_AI_OUTPUT_SAMPLE_RATE, pcm.reshape(1, -1)))
                handler.last_activity_time = loop.time()
                duration = pcm.size / OPEN_AI_OUTPUT_SAMPLE_RATE
                data = b""
            # Account for processing inside the sample budget, as the original
            # player did. Sleeping a full duration here accumulates drift on
            # every chunk. The paused branch rebases this clock before resume.
            pacing_deadline += duration
            delay = max(0.0, pacing_deadline - loop.time())
            try:
                if delay:
                    await asyncio.wait_for(stop_event.wait(), delay)
                else:
                    await asyncio.sleep(0)
            except asyncio.TimeoutError:
                pass
        stopped = stop_event.is_set() or bool(cancel_event and cancel_event.is_set())
        if not stopped:
            return_code = await asyncio.wait_for(proc.wait(), 1)
            if return_code != 0:
                raise RuntimeError(f"mpg123 failed with exit code {return_code}")
            drain_output = cast(Callable[[], Awaitable[bool]] | None, getattr(handler, "_drain_queue", None))
            if callable(drain_output):
                while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
                    if not pause_event.is_set():
                        await asyncio.sleep(0.02)
                        continue
                    pending_drain = asyncio.ensure_future(drain_output())
                    try:
                        while not pending_drain.done():
                            if stop_event.is_set() or (cancel_event and cancel_event.is_set()) or not pause_event.is_set():
                                break
                            await asyncio.wait({pending_drain}, timeout=0.02)
                        output_drained = bool(await pending_drain) if pending_drain.done() else False
                    finally:
                        if not pending_drain.done():
                            pending_drain.cancel()
                        await asyncio.gather(pending_drain, return_exceptions=True)
                    if not pause_event.is_set():
                        output_drained = False
                    if output_drained:
                        break
                    # An old epoch may finish after a rapid pause/resume. Keep
                    # the owner and obtain a fresh fence, never revive its PCM.
                    await asyncio.sleep(0.02)
                stopped = stop_event.is_set() or bool(cancel_event and cancel_event.is_set()) or not output_drained
                output_drained = output_drained and not stopped
        return stopped
    finally:

        async def finish() -> None:
            global _playback_pause, _playback_stop, _active_decoder, _playback_owner
            global _playback_motion_lock, _playback_done
            try:
                if proc is not None:
                    drain = stderr_task or asyncio.create_task(_drain_pipe(proc.stderr))
                    await _reap_decoder(proc, drain)
            finally:
                try:
                    if not output_drained:
                        _clear_handler_audio(handler)
                    async with motion_lock:
                        await asyncio.to_thread(dancer.stop)
                finally:
                    handler.unmute_output_audio(owner)
                    if _active_decoder is proc:
                        _active_decoder = None
                    if _playback_pause is pause_event:
                        _playback_pause = None
                        _playback_stop = None
                        _playback_owner = None
                        _playback_motion_lock = None
                        _playback_done = None

        cleanup = asyncio.create_task(finish())
        cancelled_during_cleanup = False
        try:
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled_during_cleanup = True
            await cleanup
        finally:
            done_event.set()
        if cancelled_during_cleanup:
            raise asyncio.CancelledError


def _clear_handler_audio(handler: Any) -> None:
    """Discard queued PCM while preserving chat and status events."""
    queue = getattr(handler, "output_queue", None)
    retained: list[Any] = []
    if queue is not None:
        while not queue.empty():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], np.ndarray)):
                retained.append(item)
        for item in retained:
            queue.put_nowait(item)
    clear_backend = getattr(handler, "_clear_queue", None)
    if callable(clear_backend):
        clear_backend()


class MakeSongAndDance(Tool):
    """Own one generation-to-playback job without blocking the voice receiver."""

    name = "make_song_and_dance"
    description = "Generate and play one song, then dance. Request a music duration from 3 to 600 seconds."
    parameters_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "music_length_ms": {"type": "integer", "minimum": 3000, "maximum": 600000},
            "model_id": {"type": "string", "enum": ["music_v1", "music_v2", "music_v2_5"]},
            "composition_plan": {"type": "object"},
            "positive_styles": {"type": "array", "items": {"type": "string"}},
            "negative_styles": {"type": "array", "items": {"type": "string"}},
            "lyrics": {"type": "string"},
            "language": {"type": "string"},
            "force_instrumental": {"type": "boolean"},
            "output_format": {"type": "string"},
            "finetune_id": {
                "type": "string",
                "description": "Existing permission-gated Music finetune ID, not a TTS voice_id or guaranteed Vocal identity.",
            },
        },
        "required": ["music_length_ms"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Validate, generate, save and play one track using settings or explicit overrides."""
        global _generation_cancel
        prompt = kwargs.get("prompt")
        plan = kwargs.get("composition_plan")
        if prompt is None and plan is None:
            return {"error": "Provide a prompt or composition plan."}
        if _generation_cancel is not None or _playback_pause is not None:
            return {"status": "busy", "message": "A music job is already active. Stop it first."}
        from dj_reachy_remix.config import config

        cancel_event = threading.Event()
        _generation_cancel = cancel_event
        options = {
            "music_length_ms": kwargs.get("music_length_ms", config.ELEVEN_MUSIC_LENGTH_MS),
            "model_id": kwargs.get("model_id", getattr(config, "MUSIC_MODEL_ID", "music_v2_5")),
            "composition_plan": plan,
            "positive_styles": kwargs.get("positive_styles", getattr(config, "MUSIC_POSITIVE_STYLES", [])),
            "negative_styles": kwargs.get("negative_styles", getattr(config, "MUSIC_NEGATIVE_STYLES", [])),
            "lyrics": kwargs.get("lyrics", getattr(config, "MUSIC_LYRICS", "")),
            "language": kwargs.get("language", getattr(config, "MUSIC_LANGUAGE", "")),
            "force_instrumental": kwargs.get("force_instrumental", getattr(config, "MUSIC_FORCE_INSTRUMENTAL", False)),
            "output_format": kwargs.get("output_format", getattr(config, "MUSIC_OUTPUT_FORMAT", "mp3_48000_192")),
            "finetune_id": kwargs.get("finetune_id", getattr(config, "MUSIC_FINETUNE_ID", None)) or None,
            "cancel_event": cancel_event,
        }
        try:
            path = await _compose_music(prompt, config.ELEVEN_MUSIC_TIMEOUT_S, deps, **options)
            if cancel_event.is_set():
                _discard_local_track(path)
                return {"status": "cancelled", "message": "Local result discarded; provider work may have continued."}
            notify = getattr(deps.openai_realtime_handler, "notify_song_ready", None)
            if callable(notify):
                notice_drained = await cast(Callable[[], Awaitable[bool | None]], notify)()
                if notice_drained is False:
                    return {
                        "status": "stopped", "song_path": path,
                        "message": "Saved song retained; readiness announcement was interrupted.",
                    }
            if cancel_event.is_set():
                return {"status": "stopped", "song_path": path, "message": "Saved song retained; playback cancelled."}
            if _generation_cancel is cancel_event:
                _generation_cancel = None
            stopped = await _play_mp3(path, deps, cancel_event)
            if cancel_event.is_set():
                _discard_local_track(path)
                return {
                    "status": "cancelled",
                    "message": "Local playback was stopped; the provider job had already finished.",
                }
            return {"status": "stopped" if stopped else "completed", "song_path": path}
        except LocalGenerationCancelled as exc:
            return {"status": "cancelled", "message": str(exc)}
        except asyncio.CancelledError:
            cancel_event.set()
            raise
        finally:
            if _generation_cancel is cancel_event:
                _generation_cancel = None


class PauseMusic(Tool):
    """Pause local PCM delivery and release conversation muting."""

    name = "pause_music"
    description = "Pause the current song at its current position."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, deps: ToolDependencies, **_kwargs: Any) -> Dict[str, Any]:
        """Flush buffered music and fade motion to zero before acknowledging pause."""
        pause, lock = _playback_pause, _playback_motion_lock
        if pause is None or lock is None:
            return {"status": "not_playing"}
        async with lock:
            if _playback_pause is not pause:
                return {"status": "not_playing"}
            if not pause.is_set():
                return {"status": "paused"}
            pause.clear()
            handler = deps.openai_realtime_handler
            _clear_handler_audio(handler)
            if handler is not None and _playback_owner is not None:
                handler.unmute_output_audio(_playback_owner)
            await asyncio.to_thread(_get_music_dancer(deps).stop)
        return {"status": "paused"}


class ResumeMusic(Tool):
    """Resume the held decoder position with fresh pacing and mute ownership."""

    name = "resume_music"
    description = "Resume the paused song from its held position."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, deps: ToolDependencies, **_kwargs: Any) -> Dict[str, Any]:
        """Resume only the current playback owner, never a replacement session."""
        pause, lock = _playback_pause, _playback_motion_lock
        if pause is None or lock is None:
            return {"status": "not_playing"}
        async with lock:
            if _playback_pause is not pause:
                return {"status": "not_playing"}
            if _playback_stop is not None and _playback_stop.is_set():
                return {"status": "stopped"}
            if pause.is_set():
                return {"status": "resumed"}
            handler = deps.openai_realtime_handler
            _clear_handler_audio(handler)
            if handler is not None and _playback_owner is not None:
                handler.mute_output_audio(_playback_owner)
            _get_music_dancer(deps).start()
            pause.set()
        return {"status": "resumed"}


class StopMusic(Tool):
    """Cancel local generation or reap playback before returning."""

    name = "stop_music"
    description = "Stop the local decoder, or discard an in-flight generation result."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, deps: ToolDependencies, **_kwargs: Any) -> Dict[str, Any]:
        """Stop local work without asserting provider cancellation or credit refunds."""
        done_event = _playback_done
        generation = _generation_cancel
        if _generation_cancel is not None:
            _generation_cancel.set()
        if _playback_stop is not None:
            _playback_stop.set()
        if _playback_pause is not None:
            _playback_pause.set()
        if _active_decoder is not None and _active_decoder.returncode is None:
            try:
                _active_decoder.send_signal(signal.SIGCONT)
                _active_decoder.terminate()
            except ProcessLookupError:
                pass
        handler = getattr(deps, "openai_realtime_handler", None)
        if handler is not None:
            _clear_handler_audio(handler)
        if done_event is not None:
            await done_event.wait()
        if generation is not None:
            return {
                "status": "cancellation_requested",
                "message": "Provider work may continue; local result will be discarded.",
            }
        return {"status": "stopped", "message": "Local decoder stopped and queued audio cleared."}


def _library_root() -> Path:
    from dj_reachy_remix.config import config

    root = Path(config.SONGS_DIR)
    if root.is_symlink():
        raise ValueError("Music library must not be a symlink")
    return root.resolve(strict=True)


def resolve_saved_song(filename: str) -> Path:
    """Resolve an exact MP3 filename, rejecting traversal, links and special files."""
    if (
        not isinstance(filename, str)
        or not filename
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or Path(filename).suffix.lower() != ".mp3"
    ):
        raise ValueError("Invalid saved-track filename")
    path = _library_root() / filename
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Saved track must be a regular file, not a symlink")
    metadata = path.with_suffix(".json")
    if metadata.is_symlink() or (metadata.exists() and not stat.S_ISREG(metadata.lstat().st_mode)):
        raise ValueError("Saved metadata must be a regular file, not a symlink")
    return path


def _open_regular(path: Path) -> int:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("Library entries must be regular files")
    return fd


def load_song_metadata(filename: str) -> dict[str, Any]:
    """Read an optional per-track sidecar without following symlinks."""
    path = resolve_saved_song(filename).with_suffix(".json")
    try:
        fd = _open_regular(path)
    except FileNotFoundError:
        return {}
    with os.fdopen(fd, encoding="utf-8") as stream:
        content = stream.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024:
        raise ValueError("Saved metadata exceeds 1 MiB")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Saved metadata must be an object")
    return data


def list_saved_songs() -> list[dict[str, Any]]:
    """List regular saved MP3s, including legacy tracks without sidecars."""
    try:
        root = _library_root()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for entry in root.iterdir():
        try:
            path = resolve_saved_song(entry.name)
            metadata = load_song_metadata(entry.name)
            info = path.stat()
        except (OSError, ValueError):
            continue
        title = metadata.get("title")
        rows.append(
            {
                "filename": path.name,
                "title": title if isinstance(title, str) else path.stem,
                "size_bytes": info.st_size,
                "created_at_ms": int(info.st_mtime * 1000),
            }
        )
    return sorted(rows, key=lambda row: row["created_at_ms"], reverse=True)


def delete_saved_song(filename: str) -> bool:
    """Delete only an exact validated audio/sidecar pair in the music library."""
    path = resolve_saved_song(filename)
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return True


class ListSavedSongs(Tool):
    """Let voice sessions discover the library instead of guessing filenames."""

    name = "list_saved_songs"
    description = (
        "List the newest saved library tracks with their exact filenames and short titles. "
        "Use before play_saved_song when the filename isn't already known from this tool."
    )
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, deps: ToolDependencies, **_kwargs: Any) -> Dict[str, Any]:
        """Read only local library metadata; no provider or playback side effects."""
        rows = await asyncio.to_thread(list_saved_songs)
        songs = []
        for row in rows[:LIST_SAVED_SONGS_LIMIT]:
            title = " ".join(str(row["title"]).split())
            if len(title) > LIST_SAVED_TITLE_CHARS:
                title = title[: LIST_SAVED_TITLE_CHARS - 1].rstrip() + "\u2026"
            created = datetime.fromtimestamp(row["created_at_ms"] / 1000, timezone.utc)
            songs.append({"filename": row["filename"], "title": title, "created_at": created.isoformat(timespec="minutes")})
        logger.info("Listed %d of %d saved tracks for voice", len(songs), len(rows))
        return {"count": len(rows), "songs": songs}


class PlaySavedSong(Tool):
    """Replay a validated saved MP3 through the same exclusive playback owner."""

    name = "play_saved_song"
    description = (
        "Replay one saved library track by its exact filename without regeneration. "
        "Call list_saved_songs first if you don't have the exact filename."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"filename": {"type": "string"}},
        "required": ["filename"],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, filename: str = "", **_kwargs: Any) -> Dict[str, Any]:
        """Replay an open no-follow file descriptor without contacting a provider."""
        if _generation_cancel is not None or _playback_pause is not None:
            return {"status": "busy", "message": "A music job is already active. Stop it first."}

        try:
            path = resolve_saved_song(filename)
            fd = _open_regular(path)
        except (OSError, ValueError):
            return {"error": "Saved track not found or unsafe filename."}
        try:
            stopped = await _play_mp3(f"/dev/fd/{fd}", deps, source_fd=fd)
        finally:
            os.close(fd)
        return {"status": "stopped" if stopped else "completed", "song_path": str(path), "regenerated": False}
