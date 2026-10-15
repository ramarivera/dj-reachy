"""Exercise handler turn ownership and cancellation without hardware or provider access."""

from __future__ import annotations
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import dj_reachy.openai_realtime as realtime
from dj_reachy.openai_realtime import OpenaiRealtimeHandler
from dj_reachy.tools.core_tools import ToolDependencies


def handler():
    """Construct the real handler with hardware actions intercepted at injected dependencies."""
    return OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


def delegation(identity, transcript, turn):
    """Encode the pinned consumer event, which carries no task text in metadata."""
    return SimpleNamespace(
        type="delegation.created",
        item={"type": "delegation", "id": identity, "target": "client", "content": []},
        transcript_context=transcript,
        turn_id=turn,
    )


@pytest.mark.asyncio
async def test_general_delegation_uses_planner_and_correlates_tool_round(monkeypatch):
    """A natural singing request must reach app execution, not a regex rejection."""
    app = handler()
    commands = []
    contexts = []
    rounds = []

    async def dispatch(name, args, deps):
        commands.append((name, args))
        return {"status": "completed", "filename": "salsa.mp3"}

    async def plan(transcript, history):
        rounds.append((transcript, list(history)))
        if not history:
            call = {
                "type": "function_call",
                "call_id": "song-1",
                "name": "make_song_and_dance",
                "arguments": '{"prompt":"salsa robots","music_length_ms":10000}',
            }
            return {
                "calls": [
                    {
                        "call_id": "song-1",
                        "name": call["name"],
                        "arguments": {"prompt": "salsa robots", "music_length_ms": 10000},
                    }
                ],
                "output": [call],
                "text": "",
            }
        return {"calls": [], "output": [], "text": "The salsa finished."}

    async def context(text, channel, identity=None):
        contexts.append((text, channel, identity))

    async def close():
        pass

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    app.connection = SimpleNamespace(plan=plan, send_context=context, close=close)
    app.schedule_subscription_delegation(delegation("d-song", "sing me a ten-second salsa about robots", 1))
    await asyncio.gather(*list(app._subscription_tasks.values()))
    assert commands == [("make_song_and_dance", '{"prompt": "salsa robots", "music_length_ms": 10000}')]
    assert rounds[1][1][-1]["call_id"] == "song-1"
    assert rounds[1][1][-1]["output"] == '{"status": "completed", "filename": "salsa.mp3"}'
    assert contexts[-1] == ("The salsa finished.", "speakable", "d-song")
    # A duplicate completed delegation cannot generate another song.
    app.schedule_subscription_delegation(delegation("d-song", "sing me a ten-second salsa about robots", 1))
    await asyncio.sleep(0)
    assert len(commands) == 1
    await app.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_at", ["planner", "song"])
async def test_turn_controls_preempt_work_and_reject_stale_results(monkeypatch, blocked_at):
    """Receiver handles pause/resume/stop in order while planning or song work is suspended."""
    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    app = handler()
    connection = SubscriptionRealtimeAdapter().connect()
    app.connection = connection
    blocked = asyncio.Event()
    released = asyncio.Event()
    stopped = asyncio.Event()
    commands = []
    contexts = []
    cancellation_seen = []

    async def wait_for_release():
        blocked.set()
        try:
            await released.wait()
        except asyncio.CancelledError:
            cancellation_seen.append(True)
            # Simulate a late result from work that couldn't be interrupted upstream.
            await released.wait()

    async def plan(transcript, history):
        if blocked_at == "planner":
            await wait_for_release()
        return {
            "calls": [{"call_id": "song-1", "name": "make_song_and_dance", "arguments": {"prompt": transcript}}],
            "output": [
                {"type": "function_call", "call_id": "song-1", "name": "make_song_and_dance", "arguments": "{}"}
            ],
        }

    async def dispatch(name, args, deps):
        commands.append(name)
        if name == "make_song_and_dance":
            await wait_for_release()
        if name == "stop_music":
            stopped.set()
        return {"status": "stopped" if name == "stop_music" else "completed"}

    async def context(text, channel, identity=None):
        contexts.append((text, channel, identity))

    monkeypatch.setattr(realtime, "dispatch_tool_call", dispatch)
    connection.plan = plan
    connection.send_context = context

    async def event(value):
        connection._events.put_nowait(SimpleNamespace(**value))
        await app._handle_event(await connection.__anext__())
        await asyncio.sleep(0)

    try:
        await event({"type": "input_transcript.added", "item": {"text": "sing me a song"}})
        await event({"type": "delegation.created", "item": {"type": "delegation", "target": "client", "id": "d-song"}})
        await asyncio.wait_for(blocked.wait(), 1)
        # The final transcript can arrive after delegation. It must not create a second request.
        await event({"type": "turn.done", "turn": {"role": "user", "transcript": "sing me a song"}})
        await event(
            {"type": "delegation.created", "item": {"type": "delegation", "target": "client", "id": "d-stale"}}
        )
        for word in ("pause music", "resume music", "stop music"):
            await event({"type": "input_transcript.added", "item": {"text": word}})
            await event({"type": "turn.done", "turn": {"role": "user", "transcript": word}})
            await event(
                {"type": "delegation.created", "item": {"type": "delegation", "target": "client", "id": "d-" + word}}
            )
        await asyncio.wait_for(stopped.wait(), 1)
        assert [name for name in commands if name in realtime.CONTROL_TOOLS] == [
            "pause_music",
            "resume_music",
            "stop_music",
        ]
        released.set()
        await asyncio.gather(*list(app._subscription_tasks.values()), return_exceptions=True)
        assert cancellation_seen
        assert commands.count("make_song_and_dance") == (1 if blocked_at == "song" else 0)
        assert not any(identity == "d-song" and channel == "speakable" for _, channel, identity in contexts)
        assert app._song_task is None
    finally:
        released.set()
        await app.shutdown()


@pytest.mark.asyncio
async def test_shutdown_owns_in_progress_webrtc_handshake(monkeypatch):
    """Shutdown must cancel setup even before __aenter__ publishes the connection."""
    from dj_reachy.config import config
    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    entered = asyncio.Event()
    forever = asyncio.Event()

    async def signal(body):
        entered.set()
        await forever.wait()

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
        assert connection._microphone.readyState == "ended"
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_explicit_reconnect_replaces_real_webrtc_without_duplicate_receiver(monkeypatch):
    """Reload uses the supervisor, closes the previous peer, and negotiates fresh state."""
    import json

    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

    from dj_reachy.config import config
    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    peers = []
    app = handler()

    async def signal(body):
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peers.append(peer)

        @peer.on("datachannel")
        def channel(channel):
            channel.send(json.dumps({"type": "session.started", "session": {"id": str(len(peers))}}))

        await peer.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type="offer"))
        await peer.setLocalDescription(await peer.createAnswer())
        return {"ok": True, "auth": "subscription", "transport": {"type": "webrtc", "sdp": peer.localDescription.sdp}}

    def client(key):
        app.client = SubscriptionRealtimeAdapter(signal=signal)

    monkeypatch.setattr(config, "VOICE_BACKEND", "subscription")
    monkeypatch.setattr(app, "_create_client", client)
    running = asyncio.create_task(app.start_up())
    try:
        await asyncio.wait_for(app._connected_event.wait(), 3)
        first = app.connection
        await app.reload_instructions()
        assert len(peers) == 2
        assert app.connection is not first
        assert first._pc.connectionState == "closed"
        assert first._microphone.readyState == "ended"
        assert app._supervisor is running and not running.done()
        assert app.connection.ready.is_set()
    finally:
        await app.shutdown()
        await asyncio.gather(running, return_exceptions=True)
        for peer in peers:
            await peer.close()


def test_subscription_selection_never_constructs_api_client(monkeypatch):
    """Even a supplied API key cannot be used by subscription Live."""
    from dj_reachy.config import config

    def forbidden(**kwargs):
        raise AssertionError("API-key billing was selected implicitly")

    monkeypatch.setattr(realtime, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(config, "VOICE_BACKEND", "subscription")
    app = handler()
    app._create_client("fixture-api-key-must-not-be-used")
    assert app.client.model == config.VOICE_MODEL
