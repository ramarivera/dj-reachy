"""Voice sessions can discover saved tracks instead of guessing exact filenames."""

import json
import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from dj_reachy_remix.config import config
from dj_reachy_remix.voice_planner import app_tool_specs, validate_app_call


def _save_track(root: Path, stem: str, title: str, mtime: int) -> str:
    song = root / f"{stem}.mp3"
    song.write_bytes(b"ID3fixture")
    (root / f"{stem}.json").write_text(json.dumps({"title": title}))
    import os

    os.utime(song, (mtime, mtime))
    return song.name


def test_list_saved_songs_is_exposed_to_both_voice_registries() -> None:
    """The subscription planner and the API-key tool registry both list the library."""
    tools = {tool["name"]: tool for tool in app_tool_specs()[0]["tools"]}
    assert "list_saved_songs" in tools
    assert "list_saved_songs" in tools["play_saved_song"]["description"]
    assert validate_app_call("list_saved_songs", {"query": None}) == ("list_saved_songs", {})
    assert validate_app_call("list_saved_songs", {"query": "ares"}) == ("list_saved_songs", {"query": "ares"})
    with pytest.raises(ValueError):
        validate_app_call("list_saved_songs", {"filename": "x.mp3"})
    with pytest.raises(ValueError):
        validate_app_call("list_saved_songs", {"query": "x" * 201})
    profile = Path(__file__).resolve().parents[1] / "src/dj_reachy_remix/profiles/dj_reachy/tools.txt"
    assert "list_saved_songs" in profile.read_text().splitlines()


def test_live_policy_delegates_saved_library_questions() -> None:
    """Live must delegate "what songs do you have" rather than answer from memory."""
    from dj_reachy_remix.subscription_broker import compose_live_instructions

    text = compose_live_instructions("Synthetic custom persona.").lower()
    assert "list" in text and "saved songs" in text


@pytest.mark.asyncio
async def test_dispatched_listing_returns_playable_newest_first_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the real registered tool against a fixture library and replay its answer."""
    import dj_reachy_remix.profiles.dj_reachy.make_song_and_dance as music
    from dj_reachy_remix.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call

    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    older = _save_track(tmp_path, "1_old", "Birthday song for Ares", 1_700_000_000)
    newer = _save_track(tmp_path, "2_new", "Upbeat intro " + "very long prompt " * 40, 1_800_000_000)
    (tmp_path / "notes.txt").write_text("not a track")

    assert "list_saved_songs" in {item["name"] for item in get_tool_specs()}
    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())
    result = await dispatch_tool_call("list_saved_songs", "{}", deps)

    assert result["count"] == 2
    assert [song["filename"] for song in result["songs"]] == [newer, older]
    assert result["songs"][1]["title"] == "Birthday song for Ares"
    assert len(result["songs"][0]["title"]) <= 160
    for song in result["songs"]:
        assert music.resolve_saved_song(song["filename"]).parent == tmp_path.resolve()


@pytest.mark.asyncio
async def test_listing_is_bounded_and_survives_a_missing_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large library is truncated with a total; a missing library is an empty answer."""
    from dj_reachy_remix.tools.core_tools import ToolDependencies, dispatch_tool_call

    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path / "missing")
    empty = await dispatch_tool_call("list_saved_songs", "{}", deps)
    assert empty["count"] == 0 and empty["songs"] == []

    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    for index in range(30):
        _save_track(tmp_path, f"{index:02d}_track", f"Track {index}", 1_700_000_000 + index)
    listed = await asyncio.wait_for(dispatch_tool_call("list_saved_songs", "{}", deps), 5)
    assert listed["count"] == 30
    assert len(listed["songs"]) == 20
    assert listed["songs"][0]["title"] == "Track 29"
    assert "more" in listed

    oldest = await dispatch_tool_call("list_saved_songs", '{"query": "track 0"}', deps)
    assert [song["title"] for song in oldest["songs"]][-1] == "Track 0"
    assert "more" not in oldest


@pytest.mark.asyncio
async def test_listing_uses_the_panel_titles_for_legacy_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tracks named only in legacy metadata.json keep the title the UI shows."""
    from dj_reachy_remix.tools.core_tools import ToolDependencies, dispatch_tool_call

    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    (tmp_path / "old_01.mp3").write_bytes(b"ID3fixture")
    (tmp_path / "metadata.json").write_text(json.dumps({"old_01.mp3": {"title": "Moonlight Waltz"}}))
    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())
    result = await dispatch_tool_call("list_saved_songs", '{"query": "moonlight"}', deps)
    assert result["songs"] == [{"filename": "old_01.mp3", "title": "Moonlight Waltz"}]


def test_listing_is_traced_by_name() -> None:
    """Voice diagnostics must show whether the model actually consulted the library."""
    from dj_reachy_remix.audio_diagnostics import SAFE_TOOLS

    assert "list_saved_songs" in SAFE_TOOLS


@pytest.mark.asyncio
async def test_newest_first_follows_save_time_not_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recent track whose filename sorts low must still lead the capped list."""
    from dj_reachy_remix.tools.core_tools import ToolDependencies, dispatch_tool_call

    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    for index in range(25):
        _save_track(tmp_path, f"z{index:02d}", f"Old {index}", 1_700_000_000 + index)
    recent = _save_track(tmp_path, "a_recent", "Brand new", 1_900_000_000)
    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())
    result = await dispatch_tool_call("list_saved_songs", "{}", deps)
    assert result["songs"][0]["filename"] == recent


@pytest.mark.asyncio
async def test_uppercase_mp3_tracks_are_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay accepts .MP3, so discovery must too, in the panel and in voice."""
    from dj_reachy_remix.tools.core_tools import ToolDependencies, dispatch_tool_call

    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    (tmp_path / "LOUD.MP3").write_bytes(b"ID3fixture")
    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())
    result = await dispatch_tool_call("list_saved_songs", "{}", deps)
    assert [song["filename"] for song in result["songs"]] == ["LOUD.MP3"]
