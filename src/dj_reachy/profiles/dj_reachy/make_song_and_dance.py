import asyncio
import signal
import logging
import os
import re
import time
from typing import Any, Dict, Optional

import numpy as np
import requests
import json
import threading

_playback_pause: asyncio.Event | None = None
_playback_stop: asyncio.Event | None = None
_generation_cancel: threading.Event | None = None
_active_decoder: asyncio.subprocess.Process | None = None


from dj_reachy.audio.music_dancer import DanceStyle, MusicDancer
from dj_reachy.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

# Module-level music dancer instance (lazily initialized)
_music_dancer: Optional[MusicDancer] = None
_music_dancer_deps: Optional[ToolDependencies] = None


def _get_music_dancer(deps: ToolDependencies) -> MusicDancer:
    """Get or create the module-level MusicDancer instance."""
    global _music_dancer, _music_dancer_deps
    from dj_reachy.config import config

    if _music_dancer is None or _music_dancer_deps is not deps:
        _music_dancer = MusicDancer(
            set_offsets_callback=deps.movement_manager.set_music_dance_offsets,
            sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            intensity=config.MOTION_INTENSITY,
        )
        _music_dancer._controller.style = _dance_style(config.MOTION_STYLE)
        _music_dancer_deps = deps
        logger.info("Created new MusicDancer instance")
    return _music_dancer


def get_dance_intensity() -> float:
    """Get current dance intensity for the UI."""
    if _music_dancer is not None:
        return _music_dancer.intensity
    from dj_reachy.config import config
    return config.MOTION_INTENSITY


def set_dance_intensity(intensity: float) -> None:
    """Set dance intensity for the UI."""
    if _music_dancer is not None:
        _music_dancer.intensity = max(0.0, min(0.25, intensity))
        logger.info("Dance intensity updated")

def _dance_style(name: str) -> DanceStyle:
    presets = {
        "groove": {"head_bob_amplitude": 15.0, "head_bob_speed": 1.0, "body_sway_amplitude": 70.0,
                   "body_sway_speed": 1.0, "antenna_amplitude": 0.8, "emphasis_strength": 25.0,
                   "movement_smoothing": 0.25},
        "bounce": {"head_bob_amplitude": 22.0, "head_bob_speed": 1.3, "body_sway_amplitude": 55.0,
                   "body_sway_speed": 1.2, "antenna_amplitude": 0.9, "emphasis_strength": 28.0,
                   "movement_smoothing": 0.2},
        "smooth": {"head_bob_amplitude": 9.0, "head_bob_speed": 0.7, "body_sway_amplitude": 38.0,
                   "body_sway_speed": 0.7, "antenna_amplitude": 0.5, "emphasis_strength": 12.0,
                   "movement_smoothing": 0.55},
        "classic": {"head_bob_amplitude": 13.0, "head_bob_speed": 0.9, "body_sway_amplitude": 58.0,
                    "body_sway_speed": 0.9, "antenna_amplitude": 0.65, "emphasis_strength": 20.0,
                    "movement_smoothing": 0.32},
    }
    return DanceStyle(**presets.get(name, presets["groove"]))


def set_dance_style(style: str) -> None:
    """Apply one allow-listed motion preset without changing controller bounds."""
    if _music_dancer is not None:
        _music_dancer._controller.style = _dance_style(style)
ELEVEN_MUSIC_URL = "https://api.elevenlabs.io/v1/music"
SUPPORTED_MUSIC_MODELS = {"music_v1", "music_v2", "music_v2_5"}
OUTPUT_FORMATS = {
    "auto", "mp3_48000_128", "mp3_48000_192", "mp3_48000_240", "mp3_48000_320",
    "mp3_22050_32", "mp3_24000_48", "mp3_44100_32", "mp3_44100_64", "mp3_44100_96",
    "mp3_44100_128", "mp3_44100_192", "pcm_8000", "pcm_16000", "pcm_22050", "pcm_24000",
    "pcm_32000", "pcm_44100", "pcm_48000", "ulaw_8000", "alaw_8000",
    "opus_48000_32", "opus_48000_64", "opus_48000_96", "opus_48000_128", "opus_48000_192",
}


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
) -> dict[str, Any]:
    """Validate and build a request using only documented compose-body fields."""
    if model_id not in SUPPORTED_MUSIC_MODELS:
        raise ValueError(f"Unsupported Eleven Music model: {model_id}")
    if isinstance(music_length_ms, bool) or not 3000 <= music_length_ms <= 600000:
        raise ValueError("music_length_ms must be between 3000 and 600000")
    if (prompt is None) == (composition_plan is None):
        raise ValueError("provide exactly one of prompt or composition plan")
    if composition_plan is not None and force_instrumental:
        raise ValueError("force_instrumental is only valid with prompt mode, not a composition plan")
    if force_instrumental and lyrics:
        raise ValueError("instrumental music cannot include original lyrics")
    if prompt is not None and not prompt.strip():
        raise ValueError("prompt must not be empty")

    if composition_plan is None:
        lines = [prompt.strip()]  # type: ignore[union-attr]
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
        }

    plan = json.loads(json.dumps(composition_plan))
    if model_id == "music_v1":
        if "sections" not in plan or "chunks" in plan:
            raise ValueError("music_v1 plans use sections, not v2/v2.5 chunks")
        durations = [section.get("duration_ms") for section in plan.get("sections", [])]
        if positive_styles:
            plan.setdefault("positive_global_styles", [])
            plan["positive_global_styles"] = list(dict.fromkeys(plan["positive_global_styles"] + positive_styles))
        if negative_styles:
            plan.setdefault("negative_global_styles", [])
            plan["negative_global_styles"] = list(dict.fromkeys(plan["negative_global_styles"] + negative_styles))
        sections = plan.get("sections", [])
        if lyrics:
            if not sections:
                raise ValueError("lyrics require at least one plan section")
            sections[0].setdefault("lines", []).extend(lyrics.splitlines())
        if language and sections:
            sections[0].setdefault("lines", []).insert(0, f"[Lyrics language: {language}]")
    else:
        if "chunks" not in plan or "sections" in plan:
            raise ValueError("music_v2/music_v2_5 plans use chunks, not music_v1 sections")
        chunks = plan.get("chunks", [])
        durations = [chunk.get("duration_ms") for chunk in chunks]
        if positive_styles:
            chunks[0]["positive_styles"] = list(dict.fromkeys(chunks[0].get("positive_styles", []) + positive_styles)) if chunks else []
        if negative_styles:
            chunks[0]["negative_styles"] = list(dict.fromkeys(chunks[0].get("negative_styles", []) + negative_styles)) if chunks else []
        if lyrics:
            if not chunks:
                raise ValueError("lyrics require at least one plan chunk")
            chunks[0]["text"] = chunks[0].get("text", "") + "\n" + lyrics
        if language and chunks:
            chunks[0]["text"] = f"[Lyrics language: {language}]\n" + chunks[0].get("text", "")
    if not durations or any(isinstance(d, bool) or not isinstance(d, int) or not 3000 <= d <= 120000 for d in durations):
        raise ValueError("each composition plan section must have a duration from 3000 to 120000 ms")
    total_duration = sum(durations)
    if len(durations) > 30 or not 3000 <= total_duration <= 600000:
        raise ValueError("composition plan must contain up to 30 sections and total 3000 to 600000 ms")
    if total_duration != music_length_ms:
        raise ValueError("composition plan total duration must match music_length_ms")
    return {"model_id": model_id, "composition_plan": plan}


ELEVEN_COMPOSE_URL = ELEVEN_MUSIC_URL


def _get_elevenlabs_key(deps: ToolDependencies | None = None) -> str:
    """Get ElevenLabs API key from deps, config (set via UI), or environment."""
    if deps is not None and getattr(deps, "elevenlabs_api_key", None):
        return deps.elevenlabs_api_key
    from dj_reachy.config import config
    key = config.ELEVENLABS_API_KEY or os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("Missing ELEVENLABS_API_KEY. Configure it in the app settings.")
    return key


def _eleven_headers(deps: ToolDependencies | None = None) -> Dict[str, str]:
    return {"xi-api-key": _get_elevenlabs_key(deps), "Content-Type": "application/json", "Accept": "audio/mpeg"}


class LocalGenerationCancelled(RuntimeError):
    """Local cancellation won; upstream provider work may have continued."""

def _compose_music_to_mp3(
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
) -> str:
    """Generate a track and atomically commit it to the user data directory."""
    from dj_reachy.config import config
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
    )
    if cancel_event is not None and cancel_event.is_set():
        raise LocalGenerationCancelled("Local result discarded; provider work may have continued.")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported Eleven Music output_format: {output_format}")
    response = requests.post(
        ELEVEN_MUSIC_URL, params={"output_format": output_format}, json=payload,
        headers=_eleven_headers(deps), timeout=timeout_s
    )
    if cancel_event is not None and cancel_event.is_set():
        raise LocalGenerationCancelled("Local result discarded; provider work may have continued.")
    if response.status_code >= 400:
        raise RuntimeError(f"ElevenLabs music request failed with HTTP {response.status_code}")
    if "audio" not in (response.headers.get("Content-Type") or "").lower():
        raise RuntimeError("ElevenLabs music request returned a non-audio response")
    songs_dir = config.SONGS_DIR
    songs_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    title = prompt or "planned_track"
    safe_prompt = re.sub(r"[^a-zA-Z0-9_-]", "_", title[:40]).strip("_") or "track"
    path = songs_dir / f"{stamp}_{safe_prompt}.mp3"
    temp_path = path.with_suffix(".mp3.part")
    metadata_path = path.with_suffix(".json")
    metadata_temp = metadata_path.with_suffix(".json.part")
    try:
        temp_path.write_bytes(response.content)
        if temp_path.stat().st_size < 1024:
            raise RuntimeError("ElevenLabs returned an unexpectedly small audio file")
        metadata_temp.write_text(json.dumps({"title": title, "request": payload, "provider": "elevenlabs"}), encoding="utf-8")
        if cancel_event is not None and cancel_event.is_set():
            raise LocalGenerationCancelled("Local generated file was not committed.")
        os.replace(temp_path, path)
        os.replace(metadata_temp, metadata_path)
    finally:
        temp_path.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return str(path)


async def _run(cmd: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    msg = (err_b or out_b).decode("utf-8", errors="replace")
    return proc.returncode, msg


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



async def _play_mp3(path: str, deps: ToolDependencies, cancel_event: threading.Event | None = None) -> bool:
    """Stream one cancellable, pausable decoder into the audio handler."""
    global _playback_pause, _playback_stop, _active_decoder
    handler = getattr(deps, "openai_realtime_handler", None)
    if handler is None:
        raise RuntimeError("Realtime handler is not connected.")
    dancer = _get_music_dancer(deps)
    owner = f"music-{id(asyncio.current_task())}"
    pause_event = asyncio.Event()
    pause_event.set()
    stop_event = asyncio.Event()
    _playback_pause, _playback_stop = pause_event, stop_event
    handler.mute_output_audio(owner)
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "mpg123", "-q", "-s", "--mono", "--rate", str(OPEN_AI_OUTPUT_SAMPLE_RATE), path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _active_decoder = proc
        assert proc.stdout is not None
        dancer.start()
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        paused_total = 0.0
        pause_started: float | None = None
        chunks_sent = 0
        while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
            if not pause_event.is_set():
                if pause_started is None:
                    pause_started = loop.time()
                    if proc.returncode is None:
                        proc.send_signal(signal.SIGSTOP)
                _clear_handler_audio(handler)
                resume_task = asyncio.create_task(pause_event.wait())
                stop_task = asyncio.create_task(stop_event.wait())
                done, pending = await asyncio.wait(
                    (resume_task, stop_task), return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if stop_event.is_set():
                    break
                if proc.returncode is None:
                    proc.send_signal(signal.SIGCONT)
                paused_total += loop.time() - pause_started
                pause_started = None
            data = await proc.stdout.read(CHUNK_BYTES)
            if not data:
                break
            data = data[: len(data) - (len(data) % 2)]
            if not data:
                continue
            pcm = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
            dancer.feed(pcm.flatten())
            await handler.output_queue.put((OPEN_AI_OUTPUT_SAMPLE_RATE, pcm))
            chunks_sent += 1
            handler.last_activity_time = loop.time()
            expected = start_time + paused_total + chunks_sent * CHUNK_DURATION_S
            if expected > loop.time():
                await asyncio.sleep(expected - loop.time())
        if (stop_event.is_set() or (cancel_event and cancel_event.is_set())) and proc.returncode is None:
            proc.terminate()
        return_code = await asyncio.wait_for(proc.wait(), timeout=2)
        if return_code not in (0, -15) and not (stop_event.is_set() or (cancel_event and cancel_event.is_set())):
            raise RuntimeError(f"mpg123 failed with exit code {return_code}")
        return stop_event.is_set() or bool(cancel_event and cancel_event.is_set())
    finally:
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=1)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        _clear_handler_audio(handler)
        await asyncio.to_thread(dancer.stop)
        handler.unmute_output_audio(owner)
        if _active_decoder is proc:
            _active_decoder = None
        if _playback_pause is pause_event:
            _playback_pause = None
            _playback_stop = None


def _clear_handler_audio(handler: Any) -> None:
    """Discard queued PCM while preserving chat and status events."""
    queue = getattr(handler, "output_queue", None)
    if queue is None:
        return
    retained: list[Any] = []
    while not queue.empty():
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], np.ndarray)):
            retained.append(item)
    for item in retained:
        queue.put_nowait(item)



class MakeSongAndDance(Tool):
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
        },
        "required": ["music_length_ms"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        global _generation_cancel
        prompt = kwargs.get("prompt")
        plan = kwargs.get("composition_plan")
        if prompt is None and plan is None:
            return {"error": "Provide a prompt or composition plan."}
        if _generation_cancel is not None:
            return {"status": "busy", "message": "A track is already generating."}
        from dj_reachy.config import config
        cancel_event = threading.Event()
        _generation_cancel = cancel_event
        options = {
            "music_length_ms": kwargs.get("music_length_ms", config.ELEVEN_MUSIC_LENGTH_MS),
            "model_id": kwargs.get("model_id", config.MUSIC_MODEL_ID),
            "composition_plan": plan,
            "positive_styles": kwargs.get("positive_styles", config.MUSIC_POSITIVE_STYLES),
            "negative_styles": kwargs.get("negative_styles", config.MUSIC_NEGATIVE_STYLES),
            "lyrics": kwargs.get("lyrics", config.MUSIC_LYRICS),
            "language": kwargs.get("language", config.MUSIC_LANGUAGE),
            "force_instrumental": kwargs.get("force_instrumental", config.MUSIC_FORCE_INSTRUMENTAL),
            "output_format": kwargs.get("output_format", config.MUSIC_OUTPUT_FORMAT),
            "cancel_event": cancel_event,
        }
        try:
            path = await asyncio.to_thread(
                _compose_music_to_mp3, prompt, config.ELEVEN_MUSIC_TIMEOUT_S, deps, **options
            )
            if cancel_event.is_set():
                _discard_local_track(path)
                return {"status": "cancelled", "message": "Local result discarded; provider work may have continued."}
            if _generation_cancel is cancel_event:
                _generation_cancel = None
            stopped = await _play_mp3(path, deps, cancel_event)
            if cancel_event.is_set():
                _discard_local_track(path)
                return {"status": "cancelled", "message": "Local playback was stopped; the provider job had already finished."}
            return {"status": "stopped" if stopped else "completed", "song_path": path}
        except LocalGenerationCancelled as exc:
            return {"status": "cancelled", "message": str(exc)}
        except asyncio.CancelledError:
            cancel_event.set()
            if _playback_pause is not None:
                _playback_pause.set()
            if _playback_stop is not None:
                _playback_stop.set()
            if _active_decoder is not None and _active_decoder.returncode is None:
                _active_decoder.terminate()
            raise
        finally:
            if _generation_cancel is cancel_event:
                _generation_cancel = None
class PauseMusic(Tool):
    name = "pause_music"
    description = "Pause the current song at its current position."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, _deps: ToolDependencies, **_kwargs: Any) -> Dict[str, Any]:
        if _playback_pause is None:
            return {"status": "not_playing"}
        _playback_pause.clear()
        return {"status": "paused"}


class ResumeMusic(Tool):
    name = "resume_music"
    description = "Resume the paused song from its held position."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, _deps: ToolDependencies, **_kwargs: Any) -> Dict[str, Any]:
        if _playback_pause is None:
            return {"status": "not_playing"}
        _playback_pause.set()
        return {"status": "resumed"}


class StopMusic(Tool):
    name = "stop_music"
    description = "Stop the local decoder, or discard an in-flight generation result."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, deps: ToolDependencies, **_kwargs: Any) -> Dict[str, Any]:
        if _generation_cancel is not None:
            _generation_cancel.set()
        if _playback_stop is not None:
            _playback_stop.set()
        if _playback_pause is not None:
            _playback_pause.set()
        if _active_decoder is not None and _active_decoder.returncode is None:
            _active_decoder.terminate()
        handler = getattr(deps, "openai_realtime_handler", None)
        if handler is not None:
            _clear_handler_audio(handler)
        if _generation_cancel is not None:
            return {"status": "cancellation_requested", "message": "Provider work may continue; local result will be discarded."}
        return {"status": "stopped", "message": "Local decoder stopped and queued audio cleared."}
class PlaySavedSong(Tool):
    name = "play_saved_song"
    description = "Replay one saved library track by its exact filename without regeneration."
    parameters_schema = {
        "type": "object",
        "properties": {"filename": {"type": "string"}},
        "required": ["filename"],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, filename: str, **_kwargs: Any) -> Dict[str, Any]:
        from dj_reachy.config import config

        if not filename or os.path.basename(filename) != filename:
            return {"error": "Invalid library filename."}
        path = config.SONGS_DIR / filename
        try:
            if path.is_symlink() or path.suffix.lower() != ".mp3" or not path.is_file():
                return {"error": "Saved track not found."}
            if path.resolve(strict=True).parent != config.SONGS_DIR.resolve(strict=True):
                return {"error": "Saved track is outside the library."}
        except (OSError, RuntimeError):
            return {"error": "Saved track not found."}
        stopped = await _play_mp3(str(path), deps)
        return {"status": "stopped" if stopped else "completed", "song_path": str(path), "regenerated": False}
