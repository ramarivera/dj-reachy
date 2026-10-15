"""Cancellation, admission, HTTP errors and atomic library commit boundaries."""

from __future__ import annotations
import json
import asyncio
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest
import test_music_lifecycle as lifecycle
from test_music_generation import compose_server

import dj_reachy.profiles.dj_reachy.make_song_and_dance as music
from dj_reachy.config import config


actor = lifecycle.actor
real_mp3 = lifecycle.real_mp3


@pytest.mark.asyncio
async def test_task_shutdown_closes_partial_http_body(monkeypatch, tmp_path):
    """Close sockets and discard partial audio during app task shutdown."""
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    deps = SimpleNamespace(elevenlabs_api_key="fixture-only")
    async with compose_server(monkeypatch, stall=True, partial=True) as server:
        task = asyncio.create_task(music.MakeSongAndDance()(deps, prompt="fixture", music_length_ms=10000))
        await asyncio.wait_for(server.received.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.6)
        await asyncio.wait_for(server.closed.wait(), 0.5)
        assert music._generation_cancel is None
        assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_generation_admission_rejects_second_generation_and_replay(monkeypatch, tmp_path, actor):
    """Reserve one music job across generation and replay admission."""
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    actor.deps.elevenlabs_api_key = "fixture-only"
    async with compose_server(monkeypatch, stall=True) as server:
        task = asyncio.create_task(music.MakeSongAndDance()(actor.deps, prompt="first", music_length_ms=10000))
        try:
            await asyncio.wait_for(server.received.wait(), 2)
            second = await music.MakeSongAndDance()(actor.deps, prompt="second", music_length_ms=10000)
            replay = await music.PlaySavedSong()(actor.deps, filename="anything.mp3")
            assert second["status"] == replay["status"] == "busy"
            assert len(server.requests) == 1
            await music.StopMusic()(actor.deps)
            assert (await asyncio.wait_for(task, 0.6))["status"] == "cancelled"
        finally:
            server.release.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,mime,body",
    [
        (403, "application/json", b'{"secret":"do-not-echo"}'),
        (200, "audio/wav", b"x" * 2048),
        (200, "audio/mpeg", b"x"),
    ],
)
async def test_provider_errors_do_not_commit_or_echo_body(monkeypatch, tmp_path, status, mime, body):
    """Reject HTTP/content failures without saving files or echoing error bodies."""
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    deps = SimpleNamespace(elevenlabs_api_key="fixture-only")
    async with compose_server(monkeypatch, status=status, mime=mime, body=body):
        with pytest.raises(RuntimeError) as error:
            await music._compose_music("fixture", deps=deps, music_length_ms=10000)
        assert "fixture-only" not in str(error.value) and "do-not-echo" not in str(error.value)
        assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_atomic_commit_rolls_back_when_audio_publish_fails(monkeypatch, tmp_path, real_mp3):
    """Remove staged metadata when the audio visibility marker cannot publish."""
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    replace = music.os.replace

    def fail_audio(source, target):
        if str(target).endswith(".mp3"):
            raise OSError("fixture-atomic-publish-error")
        return replace(source, target)

    monkeypatch.setattr(music.os, "replace", fail_audio)
    async with compose_server(monkeypatch, body=real_mp3.read_bytes()):
        with pytest.raises(OSError, match="fixture-atomic-publish-error"):
            await music._compose_music(
                "fixture", deps=SimpleNamespace(elevenlabs_api_key="fixture-only"), music_length_ms=10000
            )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_same_timestamp_generations_get_distinct_complete_pairs(monkeypatch, tmp_path, real_mp3):
    """Keep simultaneous same-title outputs distinct and record auto's codec."""
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    monkeypatch.setattr(music.time, "time", lambda: 1234.0)
    async with compose_server(monkeypatch, body=real_mp3.read_bytes()):
        paths = await asyncio.gather(
            *(
                music._compose_music(
                    "same",
                    deps=SimpleNamespace(elevenlabs_api_key="fixture-only"),
                    music_length_ms=10000,
                    output_format="auto",
                )
                for _ in range(2)
            )
        )
    assert paths[0] != paths[1]
    for path in map(Path, paths):
        assert path.read_bytes() == real_mp3.read_bytes()
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["output_format"] == "mp3_48000_192"
        assert metadata["requested_output_format"] == "auto"
    assert len(list(tmp_path.iterdir())) == 4


def test_already_cancelled_queued_generation_does_not_load_credentials(monkeypatch):
    """Reject already-cancelled queued work before any key lookup."""
    event = threading.Event()
    event.set()

    def no_key(*_args):
        raise AssertionError("Cancelled queued generation reached the credential/provider boundary")

    monkeypatch.setattr(music, "_get_elevenlabs_key", no_key)
    with pytest.raises(music.LocalGenerationCancelled):
        music._compose_music_to_mp3("fixture", music_length_ms=10000, cancel_event=event)
