"""Music generation against real loopback HTTP, without provider access."""

from __future__ import annotations
import json
import asyncio
from types import SimpleNamespace
from pathlib import Path
from contextlib import asynccontextmanager

import pytest
import test_music_lifecycle as lifecycle

import dj_reachy.profiles.dj_reachy.make_song_and_dance as music
from dj_reachy.config import config


actor = lifecycle.actor
real_mp3 = lifecycle.real_mp3


@asynccontextmanager
async def compose_server(monkeypatch, *, body=b"x" * 2048, stall=False, partial=False, status=200, mime="audio/mpeg"):
    """Serve bounded real HTTP fixtures and observe client disconnection."""
    received = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    requests = []
    connections = set()

    async def serve(reader, writer):
        current = asyncio.current_task()
        connections.add(current)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode().split("\r\n")
            headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
            data = await reader.readexactly(int(headers["Content-Length"]))
            requests.append((lines[0], headers, json.loads(data)))
            received.set()
            response_head = (
                f"HTTP/1.1 {status} Fixture\r\nContent-Type: {mime}\r\n"
                f"Song-Id: provider-song-fixture\r\nX-Secret: do-not-save\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            ).encode()
            if partial:
                writer.write(response_head + body[:10])
                await writer.drain()
            if stall:
                disconnected = asyncio.create_task(reader.read())
                ready = asyncio.create_task(release.wait())
                done, pending = await asyncio.wait((disconnected, ready), return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if disconnected in done:
                    closed.set()
                    return
            writer.write(body[10:] if partial else response_head + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            connections.discard(current)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(music, "ELEVEN_MUSIC_URL", f"http://127.0.0.1:{port}/v1/music")
    try:
        yield SimpleNamespace(received=received, release=release, closed=closed, requests=requests)
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        if connections:
            await asyncio.gather(*connections, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_cancels_stalled_http_and_discards_track(monkeypatch, tmp_path):
    """Close a stalled local request without waiting for the provider timeout."""
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    deps = SimpleNamespace(elevenlabs_api_key="fixture-only", openai_realtime_handler=None)
    async with compose_server(monkeypatch, stall=True) as server:
        task = asyncio.create_task(music.MakeSongAndDance()(deps, prompt="fixture", music_length_ms=10000))
        try:
            await asyncio.wait_for(server.received.wait(), 2)
            await music.StopMusic()(deps)
            done, _ = await asyncio.wait({task}, timeout=0.6)
            assert task in done, "Stop left a blocking HTTP worker alive until its long provider timeout"
            assert (await task)["status"] == "cancelled"
            await asyncio.wait_for(server.closed.wait(), 0.5)
            assert list(tmp_path.iterdir()) == [], "Canceled generation committed files later"
        finally:
            server.release.set()
            await asyncio.wait_for(task, 3)


@pytest.mark.asyncio
async def test_tool_generates_decodes_and_saves_normalized_metadata(monkeypatch, tmp_path, actor, real_mp3):
    """Exercise tool defaults and overrides through HTTP, disk and real decoding."""
    monkeypatch.setattr(config, "SONGS_DIR", tmp_path)
    monkeypatch.setattr(config, "MUSIC_MODEL_ID", "music_v1")
    monkeypatch.setattr(config, "MUSIC_POSITIVE_STYLES", ["default jazz"])
    monkeypatch.setattr(config, "MUSIC_NEGATIVE_STYLES", ["default metal"])
    monkeypatch.setattr(config, "MUSIC_FINETUNE_ID", "default-finetune")
    actor.deps.elevenlabs_api_key = "fixture-only"
    async with compose_server(monkeypatch, body=real_mp3.read_bytes()) as server:
        task = asyncio.create_task(
            music.MakeSongAndDance()(
                actor.deps,
                prompt="salsa",
                music_length_ms=47000,
                model_id="music_v2_5",
                positive_styles=["brass"],
                lyrics="Baila",
                language="Spanish",
                output_format="mp3_48000_192",
                finetune_id="selected-finetune",
            )
        )
        await asyncio.wait_for(actor.handler.output_queue.get(), 2)
        result = await asyncio.wait_for(task, 5)
        assert result["status"] == "completed"
        path = Path(result["song_path"])
        assert path.read_bytes() == real_mp3.read_bytes()
        metadata = json.loads(path.with_suffix(".json").read_text())
        request_line, headers, payload = server.requests[0]
        assert request_line == "POST /v1/music?output_format=mp3_48000_192 HTTP/1.1"
        assert payload["model_id"] == "music_v2_5"
        assert payload["music_length_ms"] == 47000
        assert payload["finetune_id"] == "selected-finetune"
        assert "brass" in payload["prompt"] and "default jazz" not in payload["prompt"]
        assert "default metal" in payload["prompt"] and "Baila" in payload["prompt"]
        assert "Spanish" in payload["prompt"]
        assert set(payload) == {"model_id", "prompt", "music_length_ms", "force_instrumental", "finetune_id"}
        assert metadata["request"] == payload
        assert metadata["output_format"] == "mp3_48000_192"
        assert metadata["provider_song_id"] == "provider-song-fixture"
        assert "fixture-only" not in path.with_suffix(".json").read_text()
        assert "do-not-save" not in path.with_suffix(".json").read_text()
        assert not actor.owners and not actor.dancer.running
        assert len(list(tmp_path.iterdir())) == 2
