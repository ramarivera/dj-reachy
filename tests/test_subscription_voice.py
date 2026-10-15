from __future__ import annotations

import pytest

from dj_reachy.subscription_voice import fast_control, build_context_messages, require_subscription_broker


def test_subscription_context_chunks_utf8_at_500_bytes() -> None:
    """Keep each correlated append within the actual consumer byte limit."""
    messages = build_context_messages("🌶️ música " * 100, channel="commentary", delegation_id="dlg-1")
    assert messages
    assert all(len(message["content"][0]["text"].encode("utf-8")) <= 500 for message in messages)
    assert all(message["type"] == "delegation.context.append" for message in messages)
    assert all(message["delegation_item_id"] == "dlg-1" for message in messages)


def test_fast_controls_do_not_classify_general_requests() -> None:
    """Restrict the fast path to complete control commands."""
    assert fast_control("Please pause the music now") == "pause_music"
    assert fast_control("resume music") == "resume_music"
    assert fast_control("stop") == "stop_music"
    assert fast_control("sing me a song about a pause in the rain") is None


def test_subscription_backend_fails_closed_without_scoped_broker(monkeypatch) -> None:
    """Missing scoped signaling fails before any billing path can be chosen."""
    monkeypatch.delenv("DJ_REACHY_SUBSCRIPTION_BROKER_URL", raising=False)
    monkeypatch.delenv("DJ_REACHY_SUBSCRIPTION_BROKER_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="broker URL"):
        require_subscription_broker()


@pytest.mark.asyncio
async def test_remote_pcm_uses_real_av_frames_without_padding() -> None:
    """Deliver resampled media, not an error hidden behind a fake peer."""
    import base64
    import asyncio
    from fractions import Fraction

    import av
    import numpy as np
    from aiortc.mediastreams import MediaStreamError

    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    class Track:
        def __init__(self) -> None:
            self.count = 0

        async def recv(self):
            if self.count == 4:
                raise MediaStreamError
            frame = av.AudioFrame.from_ndarray(np.full((1, 960), 12000, dtype=np.int16), format="s16", layout="mono")
            frame.sample_rate = 48000
            frame.time_base = Fraction(1, 48000)
            frame.pts = self.count * 960
            self.count += 1
            return frame

    connection = SubscriptionRealtimeAdapter().connect()
    await connection._read_remote_audio(Track(), av, MediaStreamError)
    event = await asyncio.wait_for(connection.__anext__(), 1)
    assert event.type == "response.audio.delta"
    pcm = np.frombuffer(base64.b64decode(event.delta), dtype=np.int16)
    assert 400 <= pcm.size <= 480
    assert abs(float(pcm.mean()) - 12000) < 100


@pytest.mark.asyncio
async def test_real_webrtc_ready_context_pcm_and_cleanup() -> None:
    """Negotiate actual loopback peers, send pending context and both media directions."""
    import json
    import base64
    import asyncio

    import numpy as np
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

    from dj_reachy.voice_media import MicrophoneTrack
    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    received_context = asyncio.Queue()
    received_pcm = asyncio.Queue()
    outgoing = MicrophoneTrack()
    outgoing.feed(np.full(24000, 8000, dtype=np.int16).tobytes())
    peer.addTrack(outgoing)
    readers = []

    @peer.on("track")
    def track(track):
        async def read():
            while True:
                frame = await track.recv()
                if np.abs(frame.to_ndarray()).max() > 2000:
                    await received_pcm.put(frame)

        readers.append(asyncio.create_task(read()))

    @peer.on("datachannel")
    def channel(channel):
        @channel.on("message")
        def message(text):
            received_context.put_nowait(json.loads(text))

        channel.send(json.dumps({"type": "session.started", "session": {"id": "fixture"}}))

    async def signal(body):
        assert body["billing"] == "subscription"
        assert "greeting" not in body
        await peer.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type="offer"))
        await peer.setLocalDescription(await peer.createAnswer())
        return {"ok": True, "auth": "subscription", "transport": {"type": "webrtc", "sdp": peer.localDescription.sdp}}

    connection = SubscriptionRealtimeAdapter(signal=signal).connect()
    try:
        await connection.send_context("pending 🌶", "commentary", "d-1")
        async with connection:
            assert connection.ready.is_set()
            message = await asyncio.wait_for(received_context.get(), 5)
            assert message["delegation_item_id"] == "d-1"
            assert message["content"][0]["text"] == "pending 🌶"
            await connection.input_audio_buffer.append(
                audio=base64.b64encode(np.full(24000, 10000, dtype=np.int16).tobytes()).decode()
            )
            await asyncio.wait_for(received_pcm.get(), 5)

            async def remote_audio():
                async for event in connection:
                    if event.type == "response.audio.delta":
                        return np.frombuffer(base64.b64decode(event.delta), dtype=np.int16)

            pcm = await asyncio.wait_for(remote_audio(), 5)
            assert np.abs(pcm).max() > 2000
        assert connection._pc.connectionState == "closed"
        assert connection._microphone.readyState == "ended"
    finally:
        await connection.close()
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        outgoing.stop()
        await peer.close()


@pytest.mark.asyncio
async def test_slow_audio_consumer_cannot_discard_stop_transcript() -> None:
    """Bound media backlog separately from control events."""
    from types import SimpleNamespace

    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    connection = SubscriptionRealtimeAdapter().connect()
    connection._enqueue(SimpleNamespace(type="turn.done", turn={"role": "user", "transcript": "stop music"}))
    for _ in range(300):
        connection._enqueue(SimpleNamespace(type="response.audio.delta", delta="AAA="))
    event = await connection.__anext__()
    assert event.type == "turn.done"
    assert event.turn["transcript"] == "stop music"
    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        {"ok": True, "auth": "api", "transport": {"type": "webrtc", "sdp": "v=0"}},
        {"ok": True, "auth": "subscription", "transport": {"type": "webrtc", "sdp": "not-sdp"}},
    ],
)
async def test_enter_failure_closes_real_peer_and_media(answer) -> None:
    """Reject wrong billing and malformed SDP before starting any audio or fallback."""
    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    posts = []

    async def signal(body):
        posts.append(body)
        return answer

    connection = SubscriptionRealtimeAdapter(signal=signal).connect()
    with pytest.raises(RuntimeError):
        await connection.__aenter__()
    assert len(posts) == 1
    assert connection._pc.connectionState == "closed"
    assert connection._microphone.readyState == "ended"
    assert connection._closed


def test_utf8_context_roundtrips_exactly() -> None:
    """A scalar split must neither corrupt nor omit non-ASCII text."""
    text = "唱首歌 🌶️ " * 150
    messages = build_context_messages(text, channel="speakable", delegation_id="d-utf8")
    assert "".join(message["content"][0]["text"] for message in messages) == text.strip()
    assert all(len(message["content"][0]["text"].encode()) <= 500 for message in messages)
    assert all(message["channel"] == "speakable" for message in messages)


@pytest.mark.asyncio
async def test_context_waits_for_consumer_ready_not_only_channel_open() -> None:
    """An open SCTP channel alone doesn't mean the consumer has started its session."""
    import json
    import asyncio

    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

    from dj_reachy.subscription_voice import SubscriptionRealtimeAdapter

    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    channel_ready = asyncio.Event()
    received = asyncio.Event()
    remote_channel = []

    @peer.on("datachannel")
    def channel(channel):
        remote_channel.append(channel)
        channel_ready.set()

        @channel.on("message")
        def message(raw):
            if json.loads(raw)["type"] != "session.close":
                received.set()

    async def signal(body):
        await peer.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type="offer"))
        await peer.setLocalDescription(await peer.createAnswer())
        return {"ok": True, "auth": "subscription", "transport": {"type": "webrtc", "sdp": peer.localDescription.sdp}}

    connection = SubscriptionRealtimeAdapter(signal=signal).connect()
    await connection.send_context("Wait until ready", "commentary")
    entering = asyncio.create_task(connection.__aenter__())
    try:
        await asyncio.wait_for(channel_ready.wait(), 2)
        await asyncio.sleep(0.05)
        assert not received.is_set()
        remote_channel[0].send(json.dumps({"type": "session.started", "session": {"id": "fixture"}}))
        await asyncio.wait_for(entering, 2)
        await asyncio.wait_for(received.wait(), 2)
    finally:
        entering.cancel()
        await asyncio.gather(entering, return_exceptions=True)
        await connection.close()
        await peer.close()
