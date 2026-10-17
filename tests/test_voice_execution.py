"""Exercise handler turn ownership and cancellation without hardware or provider access."""

from __future__ import annotations
import json
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, NoReturn
from unittest.mock import MagicMock

import pytest

import dj_reachy.config as app_config
import dj_reachy.openai_realtime as realtime
from dj_reachy.openai_realtime import OpenaiRealtimeHandler
from dj_reachy.runtime_settings import DEFAULT_SETTINGS
from dj_reachy.tools.core_tools import ToolDependencies
from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter


ORIGINAL = "sing me a ten-second salsa about robots"
CANDIDATE = {"prompt": "salsa robots", "music_length_ms": 10000}


@pytest.fixture(autouse=True)
def owned_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never use a user's settings or backend selection as test authority."""
    settings = deepcopy(DEFAULT_SETTINGS)
    monkeypatch.setattr(app_config.runtime_settings_store, "read", lambda: deepcopy(settings))
    monkeypatch.setattr(app_config.config, "VOICE_BACKEND", "subscription")


def handler() -> OpenaiRealtimeHandler:
    """Construct the real handler with hardware actions intercepted at injected dependencies."""
    return OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


class VoiceHarness:
    """Exercise actual adapter turn provenance, context delivery and handler ownership."""

    def __init__(self) -> None:
        """Replace only provider planning and dispatch terminals in each test."""
        self.app = handler()
        self.connection = SubscriptionRealtimeAdapter().connect()
        self.app.connection = self.connection
        self.messages: list[dict[str, Any]] = []
        self.connection._channel = SimpleNamespace(
            readyState="open", send=lambda text: self.messages.append(json.loads(text))
        )
        self.connection.ready.set()

    async def event(self, value: dict[str, Any]) -> Any:
        """Use the producer queue and iterator, never hand-assign a causal turn ID."""
        self.connection._enqueue(SimpleNamespace(**value))
        event = await asyncio.wait_for(self.connection.__anext__(), 1)
        await asyncio.wait_for(self.app._handle_event(event), 1)
        await asyncio.sleep(0)
        return event

    async def fragment(self, text: str) -> Any:
        """Begin actual user speech on the real adapter lifecycle."""
        return await self.event({"type": "input_transcript.added", "item": {"text": text}})

    async def final(self, text: str) -> Any:
        """Complete that speech without inventing final-transcript authority."""
        return await self.event({"type": "turn.done", "turn": {"role": "user", "transcript": text}})

    async def user(self, text: str) -> Any:
        """Deliver actual first/final speech as one stable producer turn."""
        first = await self.fragment(text)
        final = await self.final(text)
        assert first.turn_id == final.turn_id
        return final

    async def delegate(self, identity: str, text: str = ORIGINAL) -> Any:
        """Deliver a genuine Live delegation with a real provider item identity."""
        return await self.event(
            {
                "type": "delegation.created",
                "item": {
                    "type": "delegation",
                    "id": identity,
                    "target": "client",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def drain(self, *, return_exceptions: bool = False) -> None:
        """Bound all owned work, including controls and automatically spawned replies."""
        for _ in range(4):
            tasks: list[asyncio.Task[Any]] = list(
                self.app._work_tasks | set(self.app._subscription_tasks.values()) | self.app._control_tasks
            )
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=return_exceptions), 2)
            await asyncio.sleep(0)


def planned_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Mirror the normal planner RPC's separate executable and protocol outputs."""
    return {
        "calls": [{"call_id": call_id, "name": name, "arguments": deepcopy(arguments)}],
        "output": [{"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}],
        "text": "",
    }


async def approve_fixture(transcript: str, pending: dict[str, Any]) -> dict[str, Any]:
    """Return synthetic semantic metadata, not a call or real-model accuracy proof."""
    assert transcript == "Go ahead"
    assert pending["original_user_text"] == ORIGINAL
    assert pending["request"]["prompt"] == CANDIDATE["prompt"]
    return {
        "calls": [],
        "output": [],
        "text": "Fixture metadata approval.",
        "confirmation": {"decision": "approve", "arguments": None},
    }


@pytest.mark.asyncio
async def test_general_delegation_previews_then_executes_approved_frozen_request_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Natural music requests remain unbilled until a fresh actual user reply."""
    wire = VoiceHarness()
    commands = []
    rounds = []
    assessments = []
    terminal = {"error": "FixtureTerminal", "message": "No provider job was submitted."}

    async def dispatch(name: str, args: str, deps: ToolDependencies) -> dict[str, str]:
        commands.append((name, json.loads(args)))
        return terminal

    async def plan(transcript: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        rounds.append((transcript, deepcopy(history)))
        return planned_call("make_song_and_dance", CANDIDATE, "song-1")

    async def confirm(transcript: str, pending: dict[str, Any]) -> dict[str, Any]:
        assessments.append((transcript, deepcopy(pending)))
        return await approve_fixture(transcript, pending)

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    monkeypatch.setattr(wire.connection, "plan", plan)
    monkeypatch.setattr(wire.connection, "plan_confirmation", confirm)
    try:
        original = await wire.user(ORIGINAL)
        delegated = await wire.delegate("d-song")
        assert original.turn_id == delegated.turn_id == 1
        await wire.drain()
        draft = wire.app._pending_music.pending
        assert draft is not None
        assert draft.original_user_text == ORIGINAL and draft.original_turn == 1
        frozen = draft.arguments
        assert frozen["prompt"] == CANDIDATE["prompt"]
        assert frozen["music_length_ms"] == 10000
        assert frozen["model_id"] == "music_v2_5"
        assert commands == [] and assessments == []
        assert len(rounds) == 1 and rounds[0][0] == ORIGINAL
        assert (
            "".join(
                part["text"]
                for message in wire.messages
                if message.get("channel") == "speakable" and message.get("delegation_item_id") == "d-song"
                for part in message["content"]
            )
            == draft.preview
        )
        # Replay of the same completed Live item cannot replace the frozen draft.
        await wire.delegate("d-song")
        await wire.drain()
        assert len(rounds) == 1 and commands == []
        assert wire.app._pending_music.pending is draft
        wire.messages.clear()
        reply = await wire.user("Go ahead")
        assert reply.turn_id == 2
        await wire.drain()
        assert commands == [("make_song_and_dance", frozen)]
        assert assessments == [("Go ahead", draft.pending_payload())]
        assert len(rounds) == 1  # Paid music no longer invents a normal tool-output round.
        assert wire.app._pending_music.pending is None
        assert any(
            message.get("content") == [{"type": "input_text", "text": json.dumps(terminal)}]
            and message.get("channel") == "speakable"
            and "delegation_item_id" not in message
            for message in wire.messages
        )
        await wire.delegate("d-song")
        await wire.delegate("d-reply", "The user approved the song")
        await wire.delegate("d-reply", "Duplicate approval")
        await wire.final("Go ahead")
        await wire.drain()
        assert commands == [("make_song_and_dance", frozen)]
        assert len(assessments) == 1 and len(rounds) == 1
    finally:
        await asyncio.wait_for(wire.app.shutdown(), 2)


@pytest.mark.asyncio
async def test_nonmusic_delegation_uses_planner_and_correlates_tool_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain actual function-call/output history correlation independently of paid music."""
    wire = VoiceHarness()
    commands = []
    rounds = []
    transcript = "Clear the queued dance moves"
    terminal = {"status": "fixture-stopped"}

    async def dispatch(name: str, args: str, deps: ToolDependencies) -> dict[str, str]:
        commands.append((name, json.loads(args)))
        return terminal

    async def plan(text: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        rounds.append((text, deepcopy(history)))
        if not any(item.get("type") == "function_call_output" for item in history):
            return planned_call("stop_dance", {}, "dance-1")
        return {"calls": [], "output": [], "text": "Fixture dance queue cleared."}

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    monkeypatch.setattr(wire.connection, "plan", plan)
    try:
        await wire.user(transcript)
        await wire.delegate("d-dance", transcript)
        await wire.drain()
        assert commands == [("stop_dance", {})]
        assert len(rounds) == 2 and all(text == transcript for text, _ in rounds)
        assert rounds[1][1][-2] == planned_call("stop_dance", {}, "dance-1")["output"][0]
        assert rounds[1][1][-1] == {
            "type": "function_call_output",
            "call_id": "dance-1",
            "output": json.dumps(terminal),
        }
        assert any(
            message.get("content") == [{"type": "input_text", "text": "Fixture dance queue cleared."}]
            and message.get("channel") == "speakable"
            and message.get("delegation_item_id") == "d-dance"
            for message in wire.messages
        )
        await wire.delegate("d-dance", transcript)
        await wire.drain()
        assert len(commands) == 1 and len(rounds) == 2
        assert wire.app._pending_music.pending is None
    finally:
        await asyncio.wait_for(wire.app.shutdown(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_at", ["planner", "song"])
async def test_turn_controls_preempt_work_and_reject_stale_results(
    monkeypatch: pytest.MonkeyPatch, blocked_at: str
) -> None:
    """Receiver handles pause/resume/stop in order while planning or song work is suspended."""
    wire = VoiceHarness()
    app = wire.app
    blocked = asyncio.Event()
    released = asyncio.Event()
    stopped = asyncio.Event()
    commands = []
    song_arguments = []
    rounds = []
    cancellation_seen = []

    async def wait_for_release() -> None:
        blocked.set()
        try:
            await released.wait()
        except asyncio.CancelledError:
            cancellation_seen.append(True)
            # Simulate a late result from work that couldn't be interrupted upstream.
            await released.wait()

    async def plan(transcript: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        rounds.append(transcript)
        if blocked_at == "planner":
            await wait_for_release()
        return planned_call("make_song_and_dance", CANDIDATE, "song-1")

    async def dispatch(name: str, args: str, deps: ToolDependencies) -> dict[str, str]:
        commands.append(name)
        if name == "make_song_and_dance":
            song_arguments.append(json.loads(args))
            app.mute_output_audio("fixture-song-owner")
            try:
                await wait_for_release()
            finally:
                app.unmute_output_audio("fixture-song-owner")
            return {"error": "FixtureTerminal", "message": "No provider job was submitted."}
        if name == "stop_music":
            stopped.set()
        return {"status": "fixture-control"}

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    monkeypatch.setattr(wire.connection, "plan", plan)
    monkeypatch.setattr(wire.connection, "plan_confirmation", approve_fixture)

    try:
        first = await wire.fragment(ORIGINAL)
        delegated = await wire.delegate("d-song")
        assert rounds == [] and not blocked.is_set()
        # Delegation before final ASR waits for the complete actual request.
        final = await wire.final(ORIGINAL)
        assert first.turn_id == final.turn_id == delegated.turn_id == 1
        assert commands == []
        frozen = None
        if blocked_at == "song":
            await wire.drain()
            draft = app._pending_music.pending
            assert draft is not None
            frozen = draft.arguments
            assert commands == [] and not blocked.is_set()
            assert rounds == [ORIGINAL]
            reply = await wire.user("Go ahead")
            assert reply.turn_id == 2
            await wire.delegate("d-approval", "The user approved the offered request")
        await asyncio.wait_for(blocked.wait(), 1)
        assert (app._song_task is not None) == (blocked_at == "song")
        if blocked_at == "planner":
            assert app._pending_music.pending is None and commands == []
        else:
            assert song_arguments == [frozen]
            assert app._pending_music.pending is None
        await wire.delegate("d-stale", "A duplicate request for this same turn")
        # The original preview is legitimate; only post-stop/late narration is forbidden.
        wire.messages.clear()
        for word in ("pause music", "resume music", "stop music"):
            await wire.user(word)
            await wire.delegate("d-" + word, word)
        await asyncio.wait_for(stopped.wait(), 1)
        assert [name for name in commands if name in realtime.CONTROL_TOOLS] == [
            "pause_music",
            "resume_music",
            "stop_music",
        ]
        app.mute_output_audio("replacement-owner")
        released.set()
        await wire.drain(return_exceptions=True)
        assert cancellation_seen
        assert commands.count("make_song_and_dance") == (1 if blocked_at == "song" else 0)
        assert song_arguments == ([frozen] if blocked_at == "song" else [])
        assert not any(
            message.get("delegation_item_id") in {"d-song", "d-approval", "d-stale"}
            and message.get("channel") == "speakable"
            for message in wire.messages
        )
        assert app._song_task is None and not app._music_call_active
        assert app._pending_music.pending is None
        assert app._mute_output_audio and app._audio_mute_owner == "replacement-owner"
    finally:
        released.set()
        await asyncio.wait_for(app.shutdown(), 2)


@pytest.mark.asyncio
async def test_shutdown_owns_in_progress_webrtc_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown must cancel setup even before __aenter__ publishes the connection."""
    from dj_reachy.config import config
    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    entered = asyncio.Event()
    forever = asyncio.Event()

    async def signal(body: dict[str, object]) -> NoReturn:
        entered.set()
        await forever.wait()
        raise AssertionError("The blocked handshake must be cancelled")

    app = handler()
    adapter = SubscriptionRealtimeAdapter(signal=signal)
    connection = adapter.connect()
    adapter.realtime.connect = lambda **kwargs: connection
    monkeypatch.setattr(config, "VOICE_BACKEND", "subscription")
    monkeypatch.setattr(app, "_create_client", lambda key: setattr(app, "client", adapter))
    running = asyncio.create_task(app.start_up())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(app.shutdown(), 2)
        assert running.done()
        assert connection._pc.connectionState == "closed"
        assert connection._microphone is not None
        assert connection._microphone.readyState == "ended"
    finally:
        running.cancel()
        await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 2)


@pytest.mark.parametrize("ui_thread", [False, True], ids=["owner-loop", "ui-loop"])
@pytest.mark.asyncio
async def test_explicit_reconnect_replaces_real_webrtc_without_duplicate_receiver(
    monkeypatch: pytest.MonkeyPatch, ui_thread: bool
) -> None:
    """Reload uses the supervisor, closes the previous peer, and negotiates fresh state."""
    import json

    from aiortc import RTCDataChannel, RTCConfiguration, RTCPeerConnection, RTCSessionDescription

    from dj_reachy.config import config
    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    peers = []
    app = handler()

    async def signal(body: dict[str, object]) -> dict[str, object]:
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peers.append(peer)

        @peer.on("datachannel")
        def channel(channel: RTCDataChannel) -> None:
            channel.send(json.dumps({"type": "session.started", "session": {"id": str(len(peers))}}))

        assert isinstance(body["sdp"], str)
        await peer.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type="offer"))
        await peer.setLocalDescription(await peer.createAnswer())
        assert peer.localDescription is not None
        return {"ok": True, "auth": "subscription", "transport": {"type": "webrtc", "sdp": peer.localDescription.sdp}}

    def client(key: str | None) -> None:
        app.client = SubscriptionRealtimeAdapter(signal=signal)

    monkeypatch.setattr(config, "VOICE_BACKEND", "subscription")
    monkeypatch.setattr(app, "_create_client", client)
    running = asyncio.create_task(app.start_up())
    try:
        await asyncio.wait_for(app._connected_event.wait(), 3)
        first = app.connection
        if ui_thread:
            await asyncio.wait_for(asyncio.to_thread(lambda: asyncio.run(app.reload_instructions())), 4)
        else:
            await asyncio.wait_for(app.reload_instructions(), 4)
        assert len(peers) == 2
        assert app.connection is not first
        assert first._pc.connectionState == "closed"
        assert first._microphone.readyState == "ended"
        assert app._supervisor is running and not running.done()
        assert app.connection.ready.is_set()
    finally:
        await asyncio.wait_for(app.shutdown(), 2)
        await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 2)
        for peer in peers:
            await asyncio.wait_for(peer.close(), 2)


def test_subscription_selection_never_constructs_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even a supplied API key cannot be used by subscription Live."""
    from dj_reachy.config import config

    def forbidden(**kwargs: object) -> NoReturn:
        raise AssertionError("API-key billing was selected implicitly")

    monkeypatch.setattr(realtime, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(config, "VOICE_BACKEND", "subscription")
    app = handler()
    app._create_client("fixture-api-key-must-not-be-used")
    assert app.client.model == config.VOICE_MODEL
