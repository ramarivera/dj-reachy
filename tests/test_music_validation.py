"""Offline request validation, including nested provider inputs."""

from __future__ import annotations

import pytest

from dj_reachy.profiles.dj_reachy.make_song_and_dance import build_music_payload


@pytest.mark.parametrize(
    "changes",
    [
        {"music_length_ms": 10000.5},
        {"prompt": 42},
        {"force_instrumental": "false"},
        {"positive_styles": "pop"},
        {"negative_styles": [False]},
        {"lyrics": ["hello"]},
        {"language": 2},
        {"prompt": None, "composition_plan": {"chunks": [{"duration_ms": 10000}]}},
        {"prompt": None, "composition_plan": {"chunks": [{"duration_ms": 10000, "text": 99, "positive_styles": []}]}},
        {
            "prompt": None,
            "composition_plan": {"chunks": [{"duration_ms": 10000, "text": "Hi", "positive_styles": "pop"}]},
        },
        {
            "prompt": None,
            "composition_plan": {
                "chunks": [{"duration_ms": 10000, "text": "Hi", "positive_styles": [], "voice_id": "tts"}]
            },
        },
        {
            "prompt": None,
            "composition_plan": {"chunks": [{"duration_ms": 10000, "text": "x" * 201, "positive_styles": []}]},
        },
        {
            "prompt": None,
            "composition_plan": {"chunks": [{"duration_ms": 10000, "text": "Hi", "positive_styles": ["pop"] * 51}]},
        },
        {
            "prompt": None,
            "composition_plan": {
                "chunks": [
                    {
                        "duration_ms": 10000,
                        "text": "Hi",
                        "positive_styles": [],
                        "conditioning_ref": {"song_id": "s", "range": {"start_ms": 10, "end_ms": 0}},
                    }
                ]
            },
        },
    ],
)
def test_malformed_request_is_rejected_locally(changes):
    """Reject malformed provider inputs, including all nested plan field types."""
    options = {"prompt": "fixture", "music_length_ms": 10000}
    options.update(changes)
    with pytest.raises(ValueError):
        build_music_payload(**options)


@pytest.mark.parametrize("output_format", ["pcm_24000", "opus_48000_128", "ulaw_8000", "alaw_8000", "invalid"])
def test_undecodable_output_format_is_rejected_before_credentials(monkeypatch, output_format):
    """Reject codecs the MP3 decoder cannot replay before loading credentials."""
    import dj_reachy.profiles.dj_reachy.make_song_and_dance as music

    def no_key(*_args):
        raise AssertionError("Unsupported formats must fail before loading credentials")

    monkeypatch.setattr(music, "_get_elevenlabs_key", no_key)
    with pytest.raises(ValueError, match="format"):
        music._compose_music_to_mp3("fixture", music_length_ms=10000, output_format=output_format)
