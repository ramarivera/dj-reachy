"""Flush the backend protocol, not a legacy backend enum label."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dj_reachy.console import LocalStream


@pytest.mark.parametrize("method", ["clear_player", "clear_output_buffer"])
def test_local_stop_flushes_supported_audio_protocol(method: str) -> None:
    """A current or legacy audio object owns its buffer flush implementation."""
    clear = Mock()
    audio = SimpleNamespace(**{method: clear})
    stream = object.__new__(LocalStream)
    stream._robot = Mock()
    stream._robot.media = SimpleNamespace(backend=object(), audio=audio)
    stream.handler = Mock()
    old_queue: asyncio.Queue[object] = asyncio.Queue()
    old_queue.put_nowait("queued frame")
    stream.handler.output_queue = old_queue
    stream.clear_audio_queue()
    clear.assert_called_once_with()
    assert stream.handler.output_queue is not old_queue
    assert stream.handler.output_queue.empty()
