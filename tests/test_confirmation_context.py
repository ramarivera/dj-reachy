"""Offline regression: retain task context without replacing current user speech.

Synthetic wire fixtures, not a recording/replay of the failed live session. The
planner stub observes the real broker envelope and deliberately makes no tools.
The context-retention cases exercise labeled, delegation-local assistant history.
"""

from __future__ import annotations
import json
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

import dj_reachy.subscription_voice as subscription
from dj_reachy.openai_realtime import OpenaiRealtimeHandler
from dj_reachy.tools.core_tools import ToolDependencies


ORIGINAL = "Make a 10-second road-trip song about driving down the coast."
CANONICAL = "The user confirmed: " + ORIGINAL


class ConfirmationHarness:
    """Retain production queue, decoder, handler, scheduler and plan envelope."""

    def __init__(self) -> None:
        """Construct without starting media, a provider or a robot."""
        self.app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
        self.connection = subscription.SubscriptionRealtimeAdapter().connect()
        self.app.connection = self.connection
        self.requests: list[dict[str, Any]] = []

    async def broker(self, body: dict[str, Any], *, operation: str = "session") -> dict[str, Any]:
        """Observe plan serialization; reject any attempt to open a session."""
        assert operation == "plan", "This fixture permits no provider/session operation"
        self.requests.append(json.loads(json.dumps(body)))
        return {"calls": [], "output": [], "text": "Synthetic clarification; no tools executed."}

    async def event(self, value: dict[str, Any]) -> SimpleNamespace:
        """Exercise the actual consumer seam, never annotate context in the test."""
        self.connection._enqueue(SimpleNamespace(**value))
        event = await asyncio.wait_for(self.connection.__anext__(), timeout=1)
        assert isinstance(event, SimpleNamespace)
        await self.app._handle_event(event)
        return event

    async def user(self, text: str) -> None:
        """Provide the partial plus final form of one complete synthetic turn."""
        await self.event({"type": "input_transcript.added", "item": {"text": text}})
        await self.event({"type": "turn.done", "turn": {"role": "user", "transcript": text}})

    async def delegate(self, identity: str, text: str) -> SimpleNamespace:
        """Use the existing canonical delegation shape, without new schema."""
        return await self.event(
            {"type": "delegation.created", "item": {
                "id": identity, "type": "delegation", "target": "client",
                "content": [{"type": "input_text", "text": text}],
            }}
        )

    async def drain(self) -> None:
        """Wait only for bounded locally scheduled jobs."""
        await asyncio.wait_for(asyncio.gather(*list(self.app._subscription_tasks.values())), timeout=1)


@pytest_asyncio.fixture
async def confirmation(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ConfirmationHarness]:
    """Replace only the external broker; own and close all handler tasks."""
    harness = ConfirmationHarness()
    monkeypatch.setattr(subscription, "broker_request", harness.broker)
    try:
        yield harness
    finally:
        await harness.app.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_delegation", [False, True], ids=["first-delegation-after-confirm", "prior-clarification"])
async def test_confirmation_keeps_original_task_available_to_planner(
    confirmation: ConfirmationHarness, initial_delegation: bool
) -> None:
    """A complete canonical task must not disappear behind a fresh confirmation.

    This asserts information availability, not generation or model behavior.
    Current words must remain first; model content must not become a transcript.
    Both event orderings are fixture assumptions, not observed live evidence.
    """
    await confirmation.user(ORIGINAL)
    if initial_delegation:
        await confirmation.delegate("original-task", ORIGINAL)
        await confirmation.drain()
    await confirmation.event({"type": "turn.done", "turn": {
        "role": "assistant", "transcript": "Shall I make that song?",
    }})
    await confirmation.user("go ahead")
    event = await confirmation.delegate("confirmed-task", CANONICAL)
    await confirmation.drain()
    assert event.transcript_context == "go ahead"
    assert event.item["content"][0]["text"] == CANONICAL
    assert len(confirmation.requests) == (2 if initial_delegation else 1)
    request = confirmation.requests[-1]
    print("SYNTHETIC_PLANNER_INPUT=" + json.dumps(request, sort_keys=True))
    assert request["transcript"].startswith("go ahead"), "Current user words retain priority"
    assert request["transcript"] == "go ahead"
    from dj_reachy.voice_planner import _validate_history

    assert _validate_history(request["history"]) == request["history"]
    assert len(request["history"]) == 1
    context = request["history"][0]
    assert context["type"] == "message" and context["role"] == "assistant"
    assert "not a user transcript" in context["content"]
    assert "take precedence" in context["content"]
    available = json.dumps({"transcript": request["transcript"], "history": request["history"]})
    assert ORIGINAL in available, (
        "Original road-trip task and explicit 10-second duration were lost before the planner; "
        f"canonical content is still on the event, but broker input is {request!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["stop", "cancel"])
async def test_current_control_wins_over_canonical_song_and_duplicate(
    confirmation: ConfirmationHarness, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    """Context repair must not let a model task defeat current local controls."""
    import dj_reachy.openai_realtime as realtime

    dispatched: list[str] = []

    async def dispatch(name: str, arguments: str, deps: ToolDependencies) -> dict[str, str]:
        dispatched.append(name)
        return {"status": "completed"}

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    await confirmation.user(ORIGINAL)
    await confirmation.user(control)
    await confirmation.delegate("control", CANONICAL)
    await confirmation.drain()
    await confirmation.delegate("control", CANONICAL)
    await confirmation.drain()
    assert dispatched == ["stop_music"]
    assert confirmation.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("with_results", [False, True])
async def test_confirmation_context_survives_real_host_planner_encoding(
    confirmation: ConfirmationHarness, monkeypatch: pytest.MonkeyPatch, with_results: bool,
) -> None:
    """Pass the actual serialized history through host validation and SSE decoding."""
    import httpx

    import dj_reachy.voice_planner as planner
    from dj_reachy.subscription_broker import CodexCredential

    await confirmation.user(ORIGINAL)
    await confirmation.user("go ahead")
    await confirmation.delegate("host-wire", CANONICAL)
    await confirmation.drain()
    if with_results:
        confirmation.requests[-1]["history"].extend([
            {"type": "function_call", "call_id": "prior-control", "name": "pause_music", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "prior-control", "output": '{"status":"paused"}'},
        ])
    payloads: list[dict[str, Any]] = []

    def fixture_transport(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        completed = {"type": "response.completed", "response": {"status": "completed", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Synthetic wire fixture."}]},
        ]}}
        return httpx.Response(200, text="data: " + json.dumps(completed) + "\n\n")

    real_client = httpx.Client

    def fixture_client(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(fixture_transport), **kwargs)

    monkeypatch.setattr(httpx, "Client", fixture_client)
    result = planner.subscription_plan(
        confirmation.requests[-1], lambda: CodexCredential("fixture-access", "fixture-account"),
        endpoint="https://fixture.invalid/responses", model="fixture-planner",
    )
    assert result["calls"] == [] and result["text"] == "Synthetic wire fixture."
    assert len(payloads) == 1
    assert payloads[0]["tool_choice"] == "auto"
    assert payloads[0]["input"][0]["role"] == "assistant"
    assert ORIGINAL in payloads[0]["input"][0]["content"]
    assert payloads[0]["input"][1]["role"] == "user"
    assert payloads[0]["input"][1]["content"][0]["text"] == "go ahead"
    if with_results:
        assert [item["type"] for item in payloads[0]["input"][2:]] == ["function_call", "function_call_output"]
        assert payloads[0]["input"][2:]==confirmation.requests[-1]["history"][1:]
