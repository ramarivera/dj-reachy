"""Generation-ready speech must not occupy or mute the live receiver."""

import base64
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler, voice_config
from dj_reachy_remix.tools.core_tools import ToolDependencies


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
