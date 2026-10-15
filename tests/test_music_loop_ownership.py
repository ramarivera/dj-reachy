"""Exercise mixed async/synchronous music tests through pytest process teardown."""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path


_RUNNER_PROBE = """
import asyncio
import sys
import pytest

loops = []
original_factory = asyncio.DefaultEventLoopPolicy.new_event_loop

def record_loop(policy):
    loop = original_factory(policy)
    loops.append(loop)
    return loop

asyncio.DefaultEventLoopPolicy.new_event_loop = record_loop
exit_code = pytest.main(sys.argv[1:])
# Retain actual loops until teardown finishes, then inspect closure directly.
# This catches the leak even if GC doesn't happen while pytest handles warnings.
unclosed = [repr(loop) for loop in loops if not loop.is_closed()]
assert not unclosed, f"Test runners left event loops open after teardown: {unclosed}"
raise SystemExit(exit_code)
"""


def test_async_then_sync_music_closes_pytest_restore_loop(tmp_path: Path) -> None:
    """Fail on resource leaks at child-process shutdown, not only test assertions."""
    project = Path(__file__).resolve().parents[1]
    environment = {
        "PATH": os.defpath,
        "PYTHONPATH": str(project / "src"),
        "HF_HUB_OFFLINE": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }
    for name in (
        "HOME",
        "DJ_REACHY_HOME",
        "CODEX_HOME",
        "HERMES_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
        "HF_HOME",
        "TMPDIR",
    ):
        directory = tmp_path / name.lower()
        directory.mkdir(mode=0o700)
        environment[name] = str(directory)
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "tracemalloc=15",
            "-c",
            _RUNNER_PROBE,
            "-q",
            "tests/test_music_generation_edges.py::test_task_shutdown_closes_partial_http_body",
            "tests/test_music_generation_edges.py::test_already_cancelled_queued_generation_does_not_load_credentials",
            "-W",
            "error::ResourceWarning",
            "-W",
            "error::pytest.PytestUnraisableExceptionWarning",
            "-W",
            "error::pytest.PytestUnhandledThreadExceptionWarning",
        ],
        cwd=project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout
