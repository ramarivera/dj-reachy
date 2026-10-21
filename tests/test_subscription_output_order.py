"""Fence locally queued speech before assistant completion, not future WebRTC media."""

import base64
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler, voice_config
from dj_reachy_remix.tools.core_tools import ToolDependencies
from dj_reachy_remix.subscription_voice import SubscriptionRealtimeAdapter


def assistant_done() -> SimpleNamespace:
    """Create an explicitly synthetic terminal event in the consumer dialect."""
    return SimpleNamespace(type="turn.done", turn={"role": "assistant", "transcript": "Synthetic fixture."})


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_turn_id", ["synthetic-provider-turn", None])
async def test_provider_turn_identity_is_not_replaced_by_the_local_user_counter(
    wire_turn_id: str | None,
) -> None:
    """Keep observed wire identity separate from the app's unrelated user counter."""
    connection = SubscriptionRealtimeAdapter().connect()
    fields = {"type": "turn.delta", "delta": "Synthetic fixture.", "provider_turn_id": "untrusted-field"}
    if wire_turn_id is not None:
        fields["turn_id"] = wire_turn_id
    event = SimpleNamespace(**fields)
    try:
        connection._enqueue(event)
        received = await connection.__anext__()
        assert received.provider_turn_id == wire_turn_id
        assert received.turn_id == 0, "Provider identity must not impersonate a local confirmation turn"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_assistant_completion_follows_previously_queued_pcm() -> None:
    """An assistant final must not overtake PCM already admitted by the adapter."""
    connection = SubscriptionRealtimeAdapter().connect()
    audio = SimpleNamespace(type="response.audio.delta", delta="AQACAA==")
    done = assistant_done()
    try:
        connection._enqueue(audio)
        connection._enqueue(done)
        assert await connection.__anext__() is audio
        assert await connection.__anext__() is done
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_user_stop_still_preempts_fenced_completion_and_audio() -> None:
    """Fencing assistant metadata cannot put a user stop behind a media backlog."""
    connection = SubscriptionRealtimeAdapter().connect()
    audio = SimpleNamespace(type="response.audio.delta", delta="AQACAA==")
    done = assistant_done()
    stop = SimpleNamespace(type="turn.done", turn={"role": "user", "transcript": "stop music"})
    try:
        connection._enqueue(audio)
        connection._enqueue(done)
        connection._enqueue(stop)
        assert await connection.__anext__() is stop
        assert await connection.__anext__() is audio
        assert await connection.__anext__() is done
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_completion_fence_does_not_wait_for_audio_enqueued_later() -> None:
    """Use a finite local watermark, not a wait for an endless remote media stream."""
    connection = SubscriptionRealtimeAdapter().connect()
    before = SimpleNamespace(type="response.audio.delta", delta="AQACAA==")
    after = SimpleNamespace(type="response.audio.delta", delta="AwAEAA==")
    done = assistant_done()
    try:
        connection._enqueue(before)
        connection._enqueue(done)
        connection._enqueue(after)
        assert await connection.__anext__() is before
        assert await connection.__anext__() is done
        assert await connection.__anext__() is after
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_excess_completion_backlog_closes_instead_of_dropping_terminal_metadata() -> None:
    """Preserve the existing fail-closed control-overflow boundary."""
    connection = SubscriptionRealtimeAdapter().connect()
    try:
        for _ in range(257):
            connection._enqueue(assistant_done())
        with pytest.raises(ConnectionError, match="session closed"):
            await asyncio.wait_for(connection.__anext__(), 1)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_admitted_notice_pcm_precedes_the_handlers_native_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the actual adapter and handler around an intercepted native fence."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(voice_config, "VOICE_BACKEND", "subscription")
    connection = SubscriptionRealtimeAdapter().connect()
    connection._channel = SimpleNamespace(readyState="open", send=lambda _text: None)
    connection.ready.set()
    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    app.connection = connection
    received: list[bytes] = []
    expected = [b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"]

    async def drain() -> bool:
        while not app.output_queue.empty():
            item = app.output_queue.get_nowait()
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], np.ndarray):
                received.append(item[1].tobytes())
        assert received == expected, "Native fence overtook locally admitted announcement PCM"
        return True

    app._drain_queue = drain
    task = asyncio.create_task(app.notify_song_ready())
    try:
        for _ in range(50):
            if app._music_ready_notice is not None:
                break
            await asyncio.sleep(0)
        assert app._music_ready_notice is not None
        for packet in expected:
            connection._enqueue(SimpleNamespace(
                type="response.audio.delta", delta=base64.b64encode(packet).decode("ascii"),
            ))
        connection._enqueue(assistant_done())
        audio_delivered = 0
        for _ in range(3):
            event = await connection.__anext__()
            if event.type == "turn.done":
                assert audio_delivered == len(expected)
            else:
                audio_delivered += 1
            await app._handle_event(event)
        assert await asyncio.wait_for(task, 1) is True
        assert received == expected
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await connection.close()
        app.diagnostics.close()


def test_provider_audio_overflow_is_reported_to_diagnostics() -> None:
    """Dropped provider audio was a silent counter (PR #8 review); count it by reason."""
    from dj_reachy_remix.audio_diagnostics import DROP_REASONS

    assert "provider_audio_overflow" in DROP_REASONS
