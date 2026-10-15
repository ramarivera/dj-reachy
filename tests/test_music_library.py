"""Saved-track operations never regenerate or follow library symlinks."""

from __future__ import annotations
import json
import asyncio
from pathlib import Path

import pytest
import test_music_lifecycle as lifecycle

import dj_reachy.profiles.dj_reachy.make_song_and_dance as music
from dj_reachy.config import config


actor = lifecycle.actor
real_mp3 = lifecycle.real_mp3


@pytest.mark.asyncio
async def test_safe_library_list_replay_and_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, actor: lifecycle.PlaybackActor, real_mp3: Path
) -> None:
    """List, decode and delete exact legacy filenames without regenerating."""
    root = tmp_path / "songs"
    root.mkdir()
    monkeypatch.setattr(config, "SONGS_DIR", root)
    song = root / "original.mp3"
    song.write_bytes(real_mp3.read_bytes())
    song.with_suffix(".json").write_text(json.dumps({"title": "Original fixture"}))
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(real_mp3.read_bytes())
    (root / "linked.mp3").symlink_to(outside)

    def no_provider(*_args: object) -> str:
        raise AssertionError("Replay must never read provider credentials or regenerate")

    monkeypatch.setattr(music, "_get_elevenlabs_key", no_provider)
    rows = music.list_saved_songs()
    assert [row["filename"] for row in rows] == ["original.mp3"]
    assert rows[0]["title"] == "Original fixture"
    result = await asyncio.wait_for(music.PlaySavedSong()(actor.deps, filename="original.mp3"), 5)
    assert result["status"] == "completed" and result["regenerated"] is False
    assert not actor.owners and not actor.dancer.running
    assert music.delete_saved_song("original.mp3") is True
    assert not song.exists() and not song.with_suffix(".json").exists()
    assert outside.read_bytes() == real_mp3.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename",
    [
        "../outside.mp3",
        "/outside.mp3",
        "a/../outside.mp3",
        "a\\outside.mp3",
        "linked.mp3",
        "bad-metadata.mp3",
        "fifo.mp3",
        "wrong.wav",
        "\x00.mp3",
    ],
)
async def test_unsafe_library_entries_are_rejected_everywhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, actor: lifecycle.PlaybackActor, filename: str
) -> None:
    """Reject traversal, symlinks and special files across all library operations."""
    import os

    root = tmp_path / "songs"
    root.mkdir()
    monkeypatch.setattr(config, "SONGS_DIR", root)
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"never follow this")
    (root / "linked.mp3").symlink_to(outside)
    (root / "bad-metadata.mp3").write_bytes(b"fixture")
    (root / "bad-metadata.json").symlink_to(outside)
    os.mkfifo(root / "fifo.mp3")
    with pytest.raises((ValueError, OSError)):
        music.resolve_saved_song(filename)
    with pytest.raises((ValueError, OSError)):
        music.delete_saved_song(filename)
    assert "error" in await music.PlaySavedSong()(actor.deps, filename=filename)
    assert music.list_saved_songs() == []
    assert outside.read_bytes() == b"never follow this"


def test_library_root_symlink_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reject a redirected library root instead of following it."""
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-root"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(config, "SONGS_DIR", link)
    with pytest.raises(ValueError, match="symlink"):
        music.list_saved_songs()


@pytest.mark.asyncio
async def test_replay_pins_no_follow_file_before_decoder_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, actor: lifecycle.PlaybackActor, real_mp3: Path
) -> None:
    """Keep decoding the validated inode when its filename changes before spawn."""
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    song = tmp_path / "pinned.mp3"
    song.write_bytes(real_mp3.read_bytes())
    outside = tmp_path / "invalid.mp3"
    outside.write_bytes(b"not audio")
    spawn = asyncio.create_subprocess_exec

    async def replace_before_spawn(
        *args: str, stdout: int, stderr: int, pass_fds: tuple[int, ...] = ()
    ) -> asyncio.subprocess.Process:
        song.unlink()
        song.symlink_to(outside)
        return await spawn(*args, stdout=stdout, stderr=stderr, pass_fds=pass_fds)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", replace_before_spawn)
    result = await music.PlaySavedSong()(actor.deps, filename="pinned.mp3")
    assert result["status"] == "completed" and actor.dancer.feed_count > 50
    assert outside.read_bytes() == b"not audio"
