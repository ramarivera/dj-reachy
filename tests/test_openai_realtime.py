import asyncio
import logging
from typing import Any
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import dj_reachy_remix.openai_realtime as rt_mod
from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler
from dj_reachy_remix.tools.core_tools import ToolDependencies


def _build_handler(loop: asyncio.AbstractEventLoop) -> OpenaiRealtimeHandler:
    asyncio.set_event_loop(loop)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return OpenaiRealtimeHandler(deps)


def test_format_timestamp_uses_wall_clock() -> None:
    """Use wall-clock years rather than interpreting monotonic uptime as a date."""
    loop = asyncio.new_event_loop()
    try:
        handler = _build_handler(loop)
        formatted = handler.format_timestamp()
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    assert int(formatted[1:5]) == datetime.now(timezone.utc).year


@pytest.mark.asyncio
async def test_start_up_retries_on_abrupt_close(monkeypatch: Any, caplog: Any) -> None:
    """Reconnect explicit API sessions after an unexpected transport close."""
    caplog.set_level(logging.WARNING)
    from dj_reachy_remix.config import config

    monkeypatch.setattr(config, "VOICE_BACKEND", "api_key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "explicit-test-fixture")
    monkeypatch.setattr(config, "VOICE_MODEL", "gpt-realtime")
    fake_cce = type("FakeCCE", (Exception,), {})
    monkeypatch.setattr(rt_mod, "ConnectionClosedError", fake_cce)

    async def fast_sleep(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep, raising=False)
    attempt_counter = {"n": 0}

    class FakeConn:
        def __init__(self, mode: str):
            self._mode = mode

            class Session:
                async def update(self, **_kw: Any) -> None:
                    return None

            class InputAudioBuffer:
                async def append(self, **_kw: Any) -> None:
                    return None

            class Item:
                async def create(self, **_kw: Any) -> None:
                    return None

            class Conversation:
                item = Item()

            class Response:
                async def create(self, **_kw: Any) -> None:
                    return None

                async def cancel(self, **_kw: Any) -> None:
                    return None

            self.session = Session()
            self.input_audio_buffer = InputAudioBuffer()
            self.conversation = Conversation()
            self.response = Response()

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            return None

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> None:
            if self._mode == "raise_on_iter":
                raise fake_cce("abrupt close (simulated)")
            raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            attempt_counter["n"] += 1
            return FakeConn("raise_on_iter" if attempt_counter["n"] == 1 else "clean")

    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            self.realtime = FakeRealtime()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(rt_mod, "AsyncOpenAI", FakeClient)
    handler = rt_mod.OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    await handler.start_up()
    assert attempt_counter["n"] == 2
    assert handler.connection is None
    assert len([r for r in caplog.records if r.levelname == "WARNING" and "closed unexpectedly" in r.msg]) == 1


@pytest.mark.asyncio
async def test_slow_music_tool_does_not_block_later_control(monkeypatch: Any) -> None:
    """An independent control task must finish while song work is suspended."""
    handler = _build_handler(asyncio.get_running_loop())
    gate = asyncio.Event()
    completed: list[str] = []

    async def dispatch(name: str, _args: str, _deps: Any) -> dict[str, str]:
        if name == "make_song_and_dance":
            await gate.wait()
        completed.append(name)
        return {"status": "ok"}

    monkeypatch.setattr(rt_mod, "dispatch_tool_call", dispatch)

    class Item:
        async def create(self, **_kwargs: Any) -> None:
            return None

    class Response:
        async def create(self, **_kwargs: Any) -> None:
            return None

    handler.connection = type(
        "Conn", (), {"conversation": type("Conversation", (), {"item": Item()})(), "response": Response()}
    )()
    music = handler.schedule_tool_call("make_song_and_dance", "{}", "music-call")
    await asyncio.sleep(0)
    control = handler.schedule_tool_call("pause_music", "{}", "pause-call")
    await asyncio.wait_for(control, timeout=1)
    assert completed == ["pause_music"]
    gate.set()
    await asyncio.wait_for(music, timeout=1)
