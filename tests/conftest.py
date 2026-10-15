"""Pytest configuration for path setup."""

import sys
import asyncio
from pathlib import Path
from collections.abc import Iterator

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="session")
def event_loop_policy() -> Iterator[asyncio.AbstractEventLoopPolicy]:
    """Own the idle restore loop used by pytest-asyncio between async runners."""
    previous_policy = asyncio.get_event_loop_policy()
    policy = asyncio.DefaultEventLoopPolicy()
    restore_loop = policy.new_event_loop()
    try:
        # pytest-asyncio otherwise allocates an unowned loop while saving the
        # ambient policy. asyncio.run() can later detach it without closing it.
        policy.set_event_loop(restore_loop)
        asyncio.set_event_loop_policy(policy)
        yield policy
    finally:
        try:
            restore_loop.close()
        finally:
            asyncio.set_event_loop_policy(previous_policy)
