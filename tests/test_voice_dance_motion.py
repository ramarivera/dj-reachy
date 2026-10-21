"""Voice sessions can steer the local dancer's style and level, within local bounds."""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import dj_reachy_remix.profiles.dj_reachy.make_song_and_dance as music
from dj_reachy_remix.voice_planner import app_tool_specs, validate_app_call
from dj_reachy_remix.audio.music_dancer import MusicDancer


def test_planner_exposes_bounded_motion_and_skip_confirmation() -> None:
    """The subscription planner can change motion and mark an explicit no-confirm request."""
    tools = {tool["name"]: tool for tool in app_tool_specs()[0]["tools"]}
    motion = tools["set_dance_motion"]["parameters"]["properties"]
    assert motion["style"]["enum"] == ["bounce", "classic", "groove", "smooth", None]
    assert motion["level_percent"]["maximum"] == 125
    assert "skip_confirmation" in tools["make_song_and_dance"]["parameters"]["properties"]

    assert validate_app_call("set_dance_motion", {"style": "bounce", "level_percent": None}) == (
        "set_dance_motion", {"style": "bounce"}
    )
    assert validate_app_call("set_dance_motion", {"style": None, "level_percent": 60}) == (
        "set_dance_motion", {"level_percent": 60}
    )
    for bad in ({"style": None, "level_percent": None}, {"style": "wild"}, {"level_percent": 126},
                {"level_percent": -1}, {"level_percent": True}, {"style": "groove", "speed": 2}):
        with pytest.raises(ValueError):
            validate_app_call("set_dance_motion", bad)

    prompt = {"prompt": "song", "music_length_ms": None}
    assert validate_app_call("make_song_and_dance", {**prompt, "skip_confirmation": True})[1] == {
        "prompt": "song", "skip_confirmation": True
    }
    assert validate_app_call("make_song_and_dance", {**prompt, "skip_confirmation": None})[1] == {"prompt": "song"}
    assert validate_app_call("make_song_and_dance", {**prompt, "skip_confirmation": False})[1] == {"prompt": "song"}
    with pytest.raises(ValueError):
        validate_app_call("make_song_and_dance", {**prompt, "skip_confirmation": "yes"})


@pytest.mark.asyncio
async def test_dispatched_motion_change_updates_the_live_dancer_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registered tool changes this session's dancing; saved panel defaults stay put."""
    from dj_reachy_remix.config import config
    from dj_reachy_remix.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call

    assert "set_dance_motion" in {item["name"] for item in get_tool_specs()}
    dancer = MusicDancer(lambda *_values: None, intensity=0.2)
    monkeypatch.setattr(music, "_music_dancer", dancer)
    monkeypatch.setattr(music, "_music_dancer_style", "groove")
    monkeypatch.setattr(config, "MOTION_STYLE", "groove")
    monkeypatch.setattr(config, "MOTION_INTENSITY", 0.2)
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(config, "RUNTIME_SETTINGS_PATH", settings_file)
    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())

    result = await dispatch_tool_call("set_dance_motion", json.dumps({"style": "bounce", "level_percent": 125}), deps)
    assert result["motion"] == {"style": "bounce", "intensity": 0.25, "percent_of_original": 125}
    assert dancer._controller.style.head_bob_amplitude == music._dance_style("bounce").head_bob_amplitude
    assert (config.MOTION_STYLE, config.MOTION_INTENSITY) == ("bounce", 0.25)
    assert not settings_file.exists()

    calmer = await dispatch_tool_call("set_dance_motion", json.dumps({"level_percent": 50}), deps)
    assert calmer["motion"] == {"style": "bounce", "intensity": 0.1, "percent_of_original": 50}

    rejected = await dispatch_tool_call("set_dance_motion", json.dumps({"level_percent": 500}), deps)
    assert "error" in rejected and dancer.intensity == 0.1


def test_motion_tool_is_registered_traced_and_documented() -> None:
    """Profile, diagnostics and both voice instructions know the motion tool."""
    from dj_reachy_remix.audio_diagnostics import SAFE_TOOLS
    from dj_reachy_remix.subscription_broker import compose_live_instructions

    root = Path(__file__).resolve().parents[1] / "src/dj_reachy_remix/profiles/dj_reachy"
    assert "set_dance_motion" in (root / "tools.txt").read_text().splitlines()
    assert "set_dance_motion" in SAFE_TOOLS
    assert "set_dance_motion" in (root / "instructions.txt").read_text()
    live = compose_live_instructions("Synthetic persona.").lower()
    assert "dance" in live and "skip" in live
