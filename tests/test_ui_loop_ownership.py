"""Voice UI construction must not leak loops or replace the caller's loop."""

import asyncio
from unittest.mock import Mock

import pytest
from fastapi import FastAPI

from dj_reachy_remix.main import mount_voice_ui
from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler
from dj_reachy_remix.tools.core_tools import ToolDependencies


def test_sync_voice_ui_owns_construction_loops(
    monkeypatch: pytest.MonkeyPatch,
    event_loop_policy: asyncio.AbstractEventLoopPolicy,
) -> None:
    """Track only this construction's real loops and assert closure before cleanup."""
    try:
        previous = event_loop_policy.get_event_loop()
    except RuntimeError:
        previous = None
    created: list[asyncio.AbstractEventLoop] = []
    original = asyncio.DefaultEventLoopPolicy.new_event_loop

    def track(policy: asyncio.DefaultEventLoopPolicy) -> asyncio.AbstractEventLoop:
        loop = original(policy)
        created.append(loop)
        return loop

    monkeypatch.setattr(asyncio.DefaultEventLoopPolicy, "new_event_loop", track)
    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())
    handler = OpenaiRealtimeHandler(deps, gradio_mode=True)
    try:
        stream = mount_voice_ui(FastAPI(), handler)
        assert stream.webrtc_component is not None
        assert all(loop.is_closed() for loop in created), "UI construction leaked an owned loop"
        try:
            after = event_loop_policy.get_event_loop()
        except RuntimeError:
            after = None
        assert after is previous, "UI construction replaced the caller's ambient loop"
    finally:
        # Only the loops allocated inside this observed call belong to this test.
        for loop in created:
            if not loop.is_closed() and not loop.is_running():
                loop.close()
        event_loop_policy.set_event_loop(previous)
