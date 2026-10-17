from __future__ import annotations

import pytest

from dj_reachy_remix.profiles.dj_reachy.make_song_and_dance import build_music_payload


ALLOWED_BODY_FIELDS = {
    "prompt",
    "composition_plan",
    "music_length_ms",
    "model_id",
    "seed",
    "force_instrumental",
    "finetune_id",
    "respect_sections_durations",
    "store_for_inpainting",
    "sign_with_c2pa",
}


def test_prompt_preferences_compile_into_valid_documented_request() -> None:
    """Compile preferences without inventing unsupported REST fields."""
    payload = build_music_payload(
        prompt="upbeat funk",
        music_length_ms=31_000,
        model_id="music_v2_5",
        positive_styles=["warm bass", "bright brass"],
        negative_styles=["metal"],
        lyrics="Move together",
        language="Spanish",
    )
    assert set(payload) <= ALLOWED_BODY_FIELDS
    assert payload["model_id"] == "music_v2_5" and payload["music_length_ms"] == 31_000
    assert all(
        text in payload["prompt"]
        for text in ("upbeat funk", "warm bass", "bright brass", "Avoid: metal", "Move together", "Spanish")
    )
    assert not {"positive_styles", "negative_styles", "lyrics", "language", "lyrics_language", "voice_id"} & set(
        payload
    )


def test_plan_preferences_compile_into_v2_chunk_schema() -> None:
    """Preserve v2 chunk lyrics and put language directions in musical styles."""
    payload = build_music_payload(
        prompt=None,
        music_length_ms=30_000,
        model_id="music_v2_5",
        composition_plan={"chunks": [{"text": "[Verse]\nHola", "duration_ms": 30000, "positive_styles": ["pop"]}]},
        positive_styles=["guitar"],
        negative_styles=["dark"],
        lyrics="Hola mundo",
        language="Spanish",
    )
    assert set(payload) <= ALLOWED_BODY_FIELDS
    chunk = payload["composition_plan"]["chunks"][0]
    assert "Hola mundo" in chunk["text"]
    assert "guitar" in chunk["positive_styles"] and "dark" in chunk["negative_styles"]
    assert "Lyrics language: Spanish" in chunk["positive_styles"]
    assert "music_length_ms" not in payload


def test_plan_model_version_and_instrumental_constraints() -> None:
    """Reject model/plan mismatches and contradictory instrumental lyrics."""
    with pytest.raises(ValueError, match="music_v1"):
        build_music_payload(prompt=None, music_length_ms=30_000, model_id="music_v1", composition_plan={"chunks": []})
    with pytest.raises(ValueError, match="instrumental"):
        build_music_payload(prompt="vocal pop", music_length_ms=30_000, force_instrumental=True, lyrics="Sing this")
