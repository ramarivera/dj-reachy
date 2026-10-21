"""Generation-ready speech must not occupy or mute the live receiver."""

import json
import base64
import asyncio
import inspect
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from collections.abc import Awaitable

import pytest

from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler, voice_config
from dj_reachy_remix.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_missing_voice_connection_does_not_authorize_music_handover() -> None:
    """An unavailable announcer is a refusal, not the legacy optional success."""
    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    try:
        assert await app.notify_song_ready() is False
        assert app._music_ready_notice is None
    finally:
        app.diagnostics.close()


@pytest.mark.asyncio
async def test_missing_native_drain_does_not_authorize_music_handover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A speech final is insufficient when no native playout fence is installed."""
    monkeypatch.setattr(voice_config, "VOICE_BACKEND", "subscription")
    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    connection = SimpleNamespace(send_context=AsyncMock())
    app.connection = connection
    task = asyncio.create_task(app.notify_song_ready())
    try:
        for _ in range(50):
            if connection.send_context.await_count:
                break
            await asyncio.sleep(0)
        assert connection.send_context.await_count == 1
        await app._handle_event(SimpleNamespace(
            type="turn.done", turn={"role": "assistant", "transcript": "Your song is ready."},
        ))
        assert await asyncio.wait_for(task, 1) is False
        assert app._music_ready_notice is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        app.diagnostics.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["connection", "work-epoch"])
async def test_ready_native_drain_cannot_release_replaced_work(
    monkeypatch: pytest.MonkeyPatch, replacement: str,
) -> None:
    """Ownership must still be current after the asynchronous native drain."""
    monkeypatch.setattr(voice_config, "VOICE_BACKEND", "subscription")
    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    connection = SimpleNamespace(send_context=AsyncMock())
    app.connection = connection
    drain_started, release_drain = asyncio.Event(), asyncio.Event()

    async def drain() -> bool:
        drain_started.set()
        await release_drain.wait()
        return True

    app._drain_queue = drain
    task = asyncio.create_task(app.notify_song_ready())
    try:
        for _ in range(50):
            if connection.send_context.await_count:
                break
            await asyncio.sleep(0)
        assert connection.send_context.await_count == 1
        await app._handle_event(SimpleNamespace(
            type="turn.done", turn={"role": "assistant", "transcript": "Your song is ready."},
        ))
        await asyncio.wait_for(drain_started.wait(), 1)
        if replacement == "connection":
            app.connection = SimpleNamespace(send_context=AsyncMock())
        else:
            app._work_epoch += 1
        release_drain.set()
        assert await asyncio.wait_for(task, 1) is False
        assert app._music_ready_notice is None
        assert not app._mute_output_audio
    finally:
        release_drain.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        app.diagnostics.close()


@pytest.mark.asyncio
async def test_ready_notice_stays_audible_and_drains_before_music(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receive voice events while the background song waits for its announcement."""
    monkeypatch.setattr(voice_config, "VOICE_BACKEND", "subscription")
    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    connection = SimpleNamespace(send_context=AsyncMock())
    app.connection = connection
    drain = AsyncMock(return_value=True)
    app._drain_queue = drain
    task = asyncio.create_task(app.notify_song_ready())
    try:
        for _ in range(20):
            if connection.send_context.await_count:
                break
            await asyncio.sleep(0)
        assert connection.send_context.await_count == 1
        assert connection.send_context.call_args.args[1] == "speakable"
        assert "ready" in connection.send_context.call_args.args[0].lower()
        notice = json.loads(connection.send_context.call_args.args[0])
        assert notice["status"] == "generated"
        assert notice["playback_status"] == "not_started"
        assert not task.done() and not app._mute_output_audio
        await app._handle_event(SimpleNamespace(type="response.audio.delta", delta=base64.b64encode(b"\x01\x00" * 480).decode()))
        assert not app.output_queue.empty()
        drain.assert_not_awaited()
        await app._handle_event(SimpleNamespace(type="turn.done", turn={"role": "assistant", "transcript": "Your song is ready."}))
        await asyncio.wait_for(task, 1)
        drain.assert_awaited_once()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        app.diagnostics.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("speech_timeout", [False, True], ids=["interrupted-drain", "speech-timeout"])
async def test_interrupted_ready_drain_keeps_song_saved_without_starting_playback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, speech_timeout: bool,
) -> None:
    """Neither an invalidated drain nor missing speech completion permits playback."""
    from dj_reachy_remix.profiles.dj_reachy import make_song_and_dance as music

    monkeypatch.setattr(voice_config, "VOICE_BACKEND", "subscription")
    monkeypatch.setattr(music, "_generation_cancel", None)
    monkeypatch.setattr(music, "_playback_pause", None)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    app = OpenaiRealtimeHandler(deps)
    deps.openai_realtime_handler = app
    connection = SimpleNamespace(send_context=AsyncMock())
    app.connection = connection
    drain = AsyncMock(return_value=speech_timeout)
    app._drain_queue = drain
    if speech_timeout:
        original_wait_for = asyncio.wait_for

        async def expire_notice(awaitable: Awaitable[object], timeout: float | None = None) -> object:
            if timeout == 8:
                if inspect.iscoroutine(awaitable):
                    awaitable.close()
                raise asyncio.TimeoutError
            return await original_wait_for(awaitable, timeout)

        monkeypatch.setattr(asyncio, "wait_for", expire_notice)
    track = tmp_path / "generated-fixture.mp3"
    track.write_bytes(b"synthetic-saved-track-fixture")
    play = AsyncMock(return_value=False)
    discard = MagicMock()
    monkeypatch.setattr(music, "_compose_music", AsyncMock(return_value=str(track)))
    monkeypatch.setattr(music, "_play_mp3", play)
    monkeypatch.setattr(music, "_discard_local_track", discard)
    task = asyncio.create_task(music.MakeSongAndDance()(deps, prompt="Synthetic fixture", music_length_ms=3000))
    try:
        for _ in range(50):
            if connection.send_context.await_count:
                break
            await asyncio.sleep(0)
        assert connection.send_context.await_count == 1
        if not speech_timeout:
            await app._handle_event(SimpleNamespace(
                type="turn.done", turn={"role": "assistant", "transcript": "Your song is ready."},
            ))
        result = await asyncio.wait_for(task, 1)
        assert result["status"] == "stopped", "An interrupted readiness fence started the saved song"
        assert result["song_path"] == str(track)
        play.assert_not_awaited()
        discard.assert_not_called()
        if speech_timeout:
            drain.assert_not_awaited()
        else:
            drain.assert_awaited_once()
        assert track.read_bytes() == b"synthetic-saved-track-fixture"
        assert music._generation_cancel is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        app.diagnostics.close()
