from __future__ import annotations

import pytest

from dj_reachy.tools.core_tools import ALL_TOOLS
from dj_reachy.profiles.dj_reachy.make_song_and_dance import build_music_payload

ALLOWED_BODY_FIELDS = {
    "prompt", "composition_plan", "music_length_ms", "model_id", "seed", "force_instrumental",
    "finetune_id", "respect_sections_durations", "store_for_inpainting", "sign_with_c2pa",
}


def test_prompt_preferences_compile_into_valid_documented_request() -> None:
    payload = build_music_payload(
        prompt="upbeat funk", music_length_ms=31_000, model_id="music_v2_5",
        positive_styles=["warm bass", "bright brass"], negative_styles=["metal"],
        lyrics="Move together", language="Spanish",
    )
    assert set(payload) <= ALLOWED_BODY_FIELDS
    assert payload["model_id"] == "music_v2_5" and payload["music_length_ms"] == 31_000
    assert all(text in payload["prompt"] for text in ("upbeat funk", "warm bass", "bright brass", "Avoid: metal", "Move together", "Spanish"))
    assert not {"positive_styles", "negative_styles", "lyrics", "language", "lyrics_language", "voice_id"} & set(payload)


def test_plan_preferences_compile_into_v2_chunk_schema() -> None:
    payload = build_music_payload(
        prompt=None, music_length_ms=30_000, model_id="music_v2_5",
        composition_plan={"chunks": [{"text": "[Verse]\nHola", "duration_ms": 30000, "positive_styles": ["pop"]}]},
        positive_styles=["guitar"], negative_styles=["dark"], lyrics="Hola mundo", language="Spanish",
    )
    assert set(payload) <= ALLOWED_BODY_FIELDS
    chunk = payload["composition_plan"]["chunks"][0]
    assert all(text in chunk["text"] for text in ("Hola mundo", "Spanish"))
    assert "guitar" in chunk["positive_styles"] and "dark" in chunk["negative_styles"]
    assert "music_length_ms" not in payload


def test_plan_model_version_and_instrumental_constraints() -> None:
    with pytest.raises(ValueError, match="music_v1"):
        build_music_payload(prompt=None, music_length_ms=30_000, model_id="music_v1", composition_plan={"chunks": []})
    with pytest.raises(ValueError, match="instrumental"):
        build_music_payload(prompt="vocal pop", music_length_ms=30_000, force_instrumental=True, lyrics="Sing this")


def test_registered_voice_tools_include_music_controls() -> None:
    from dj_reachy.tools.core_tools import ALL_TOOLS

    assert {"make_song_and_dance", "play_saved_song", "pause_music", "resume_music", "stop_music"} <= set(ALL_TOOLS)

def test_http_request_uses_only_documented_fields_and_output_query(monkeypatch, tmp_path) -> None:
    from dj_reachy.config import config
    import dj_reachy.profiles.dj_reachy.make_song_and_dance as music

    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "fixture-only")
    captured = {}

    class Response:
        status_code = 200
        headers = {"Content-Type": "audio/mpeg", "history-item-id": "song-fixture"}
        content = b"x" * 2048

    def post(url, *, params, json, headers, timeout):
        captured.update(url=url, params=params, payload=json, timeout=timeout)
        assert headers["xi-api-key"] == "fixture-only"
        return Response()

    monkeypatch.setattr(music.requests, "post", post)
    path = music._compose_music_to_mp3(
        "salsa", deps=None, music_length_ms=47_000, model_id="music_v2_5",
        positive_styles=["bright brass"], lyrics="Baila", language="Spanish",
        output_format="mp3_48000_192",
    )
    assert path.endswith(".mp3")
    assert captured["url"] == "https://api.elevenlabs.io/v1/music"
    assert captured["params"] == {"output_format": "mp3_48000_192"}
    assert set(captured["payload"]) <= ALLOWED_BODY_FIELDS
    assert captured["payload"]["music_length_ms"] == 47_000
    assert captured["payload"]["model_id"] == "music_v2_5"
    assert all(value in captured["payload"]["prompt"] for value in ("bright brass", "Baila", "Spanish"))
