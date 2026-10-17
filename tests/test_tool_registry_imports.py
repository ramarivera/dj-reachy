"""Import order cannot remove the commands used by the real control UI."""

import os
import sys
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "first_module",
    ["dj_reachy_remix.profiles.dj_reachy.make_song_and_dance", "dj_reachy_remix.tools.core_tools"],
)
def test_music_controls_dispatch_after_either_import_order(first_module: str, tmp_path: Path) -> None:
    """Dispatch the actual registered stop tool after either public import path."""
    env = dict(os.environ, DJ_REACHY_REMIX_HOME=str(tmp_path))
    code = """
import asyncio
import importlib
import sys
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock
importlib.import_module(sys.argv[1])
from dj_reachy_remix.tools.core_tools import ToolDependencies, dispatch_tool_call, get_tool_specs
async def check():
    deps = ToolDependencies(reachy_mini=Mock(), movement_manager=Mock())
    queue = asyncio.Queue()
    queue.put_nowait((24000, np.zeros(240, dtype=np.int16)))
    deps.openai_realtime_handler = SimpleNamespace(output_queue=queue)
    result = await dispatch_tool_call('stop_music', '{}', deps)
    assert result.get('status') == 'stopped', result
    assert queue.empty(), 'registered stop left PCM queued'
    names = {item['name'] for item in get_tool_specs()}
    assert {'make_song_and_dance', 'pause_music', 'resume_music', 'stop_music', 'play_saved_song'} <= names
asyncio.run(check())
"""
    result = subprocess.run(
        [sys.executable, "-c", code, first_module], env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
