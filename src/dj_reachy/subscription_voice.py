"""Subscription-only ChatGPT Live transport, with host-side credential isolation.

Consumer dialect and media are adapted from pinned Conduit sources. See
``static/conduit-voice-NOTICE.txt``. This is not the public Realtime tool protocol.
"""

from __future__ import annotations
import os
import re
import json
import base64
import asyncio
from types import SimpleNamespace
from typing import Any, Callable, Awaitable
from collections import deque
from urllib.parse import urlparse

import av
import httpx
import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.stats import RTCOutboundRtpStreamStats
from aiortc.mediastreams import MediaStreamError

from dj_reachy.voice_media import MicrophoneTrack
from dj_reachy.audio_diagnostics import AudioDiagnostics


BROKER_URL_ENV = "DJ_REACHY_SUBSCRIPTION_BROKER_URL"
BROKER_TOKEN_ENV = "DJ_REACHY_SUBSCRIPTION_BROKER_TOKEN"
LIVE_ENDPOINT = "https://chatgpt.com/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas"
DEFAULT_MODEL = "gpt-live-1-codex"
DEFAULT_VOICE = "cove"
CHANNEL = "oai-events"
MAX_CONTEXT_BYTES = 500
MAX_BODY = 256 * 1024
Signal = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def require_subscription_broker() -> tuple[str, str]:
    """Validate the session endpoint and scoped credential, never provider OAuth."""
    url = os.environ.get(BROKER_URL_ENV, "").strip()
    token = os.environ.get(BROKER_TOKEN_ENV, "").strip()
    parsed = urlparse(url)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(c.isspace() for c in url)
    ):
        raise RuntimeError(f"Subscription voice needs an authenticated broker URL in {BROKER_URL_ENV}.")
    if parsed.scheme != "https" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("Subscription broker URL must use HTTPS unless it is loopback.")
    if not token or not token.isascii() or any(c.isspace() for c in token):
        raise RuntimeError(
            f"Subscription voice needs a scoped broker token in {BROKER_TOKEN_ENV}; no API fallback is used."
        )
    return url, token


async def broker_request(body: dict[str, Any], *, operation: str = "session") -> dict[str, Any]:
    """Call the narrow host interface without redirects, proxies, or unbounded bodies."""
    url, token = require_subscription_broker()
    if operation == "plan":
        url = url.rstrip("/").rsplit("/", 1)[0] + "/plan"
    elif operation != "session":
        raise ValueError("Unknown broker operation")
    data = json.dumps(body).encode()
    if len(data) > MAX_BODY:
        raise ValueError("Broker request exceeds size limit")
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=40, follow_redirects=False) as client:
            async with client.stream(
                "POST",
                url,
                content=data,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            ) as response:
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > MAX_BODY:
                        raise RuntimeError("Subscription broker response exceeds size limit")
                if not 200 <= response.status_code < 300:
                    try:
                        rejected = json.loads(raw)
                    except ValueError:
                        rejected = {}
                    detail = rejected.get("error") if isinstance(rejected, dict) else None
                    detail = (
                        detail.replace(token, "[redacted]")
                        if isinstance(detail, str) and len(detail) <= 2000
                        else "No safe diagnosis returned"
                    )
                    raise RuntimeError(
                        f"Subscription broker refused {operation} (HTTP {response.status_code}): {detail}. No API fallback was used."
                    )
                result = json.loads(raw)
    except (httpx.HTTPError, ValueError):
        raise RuntimeError("Subscription broker connection failed; no API fallback was used.") from None
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("auth") != "subscription":
        raise RuntimeError("Broker did not confirm subscription billing; no API fallback was used.")
    return result


def _utf8_chunks(text: str, limit: int = MAX_CONTEXT_BYTES) -> list[str]:
    if limit < 4:
        raise ValueError("UTF-8 chunk limit must accommodate a Unicode scalar")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for char in text.strip():
        n = len(char.encode("utf-8"))
        if size + n > limit:
            chunks.append("".join(current))
            current = []
            size = 0
        current.append(char)
        size += n
    if current:
        chunks.append("".join(current))
    return chunks


def build_context_messages(text: str, *, channel: str, delegation_id: str | None = None) -> list[dict[str, Any]]:
    """Encode exact Codex context appends, each bounded by 500 UTF-8 bytes."""
    if channel not in ("speakable", "commentary"):
        raise ValueError("context channel must be speakable or commentary")
    messages = []
    for chunk in _utf8_chunks(text):
        message: dict[str, Any] = {
            "type": "delegation.context.append" if delegation_id else "session.context.append",
            "channel": channel,
            "content": [{"type": "input_text", "text": chunk}],
        }
        if delegation_id:
            message["delegation_item_id"] = delegation_id
        messages.append(message)
    return messages


def fast_control(text: str) -> str | None:
    """Recognize only complete immediate control commands, never historical substrings."""
    match = re.fullmatch(
        r"(?:please\s+)?(stop|cancel|pause|resume|unpause)(?:\s+(?:the\s+)?(?:music|song|track|generation))?"
        r"(?:\s+(?:now|please))?[.!?]*",
        text.strip(),
        re.IGNORECASE,
    )
    if match is None:
        return None
    return {
        "stop": "stop_music",
        "cancel": "stop_music",
        "pause": "pause_music",
        "resume": "resume_music",
        "unpause": "resume_music",
    }[match[1].lower()]


class SubscriptionRealtimeAdapter:
    """Replaceable consumer transport. Each connection owns fresh media and turn state."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        instructions: str = "",
        signal: Signal | None = None,
        ready_timeout: float = 20,
        diagnostics: AudioDiagnostics | None = None,
    ):
        """Inject signaling for loopback qualification; production uses the scoped broker."""
        self.model = model or DEFAULT_MODEL
        self.voice = voice or DEFAULT_VOICE
        self.instructions = instructions
        self.signal = signal or broker_request
        self.ready_timeout = ready_timeout
        self.diagnostics = diagnostics
        self.realtime = SimpleNamespace(connect=self.connect)

    def connect(self, **_kwargs: Any) -> SubscriptionRealtimeConnection:
        """Create a new session rather than reusing closed peer state."""
        return SubscriptionRealtimeConnection(self)


class SubscriptionRealtimeConnection:
    """WebRTC media and consumer events with bounded pending sends and deterministic close."""

    def __init__(self, adapter: SubscriptionRealtimeAdapter):
        """Initialize without reading credentials or opening a socket."""
        self.adapter = adapter
        self.diagnostics = adapter.diagnostics or AudioDiagnostics()
        self._owns_diagnostics = adapter.diagnostics is None
        self._events: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
        self._audio_events: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self._available = asyncio.Event()
        self._pending: deque[str] = deque()
        self._audio_readers: list[asyncio.Task[None]] = []
        self._stats_task: asyncio.Task[None] | None = None
        self._closed = False
        self._user_transcript = ""
        self._current_input = ""
        self._turn_id = 0
        self._consumed_turn = -1
        self._awaiting_final = False
        self._pc: Any = None
        self._channel: Any = None
        self._microphone: MicrophoneTrack | None = None
        self.ready = asyncio.Event()
        self.input_audio_buffer = SimpleNamespace(append=self._append_audio)

    def _enqueue(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if isinstance(kind, str):
            self.diagnostics.provider_event(kind)
        if getattr(event, "type", None) == "response.audio.delta":
            if self._audio_events.full():
                self._audio_events.get_nowait()
            self._audio_events.put_nowait(event)
        else:
            if self._events.full():
                # Excess control traffic is a protocol failure, never authority to run stale commands.
                while not self._events.empty():
                    self._events.get_nowait()
                event = SimpleNamespace(type="session.closed", reason="event backlog exceeded")
            self._events.put_nowait(event)
        self._available.set()

    async def __aenter__(self) -> SubscriptionRealtimeConnection:
        """Perform exactly one signaling POST and wait for consumer session.started."""
        self.diagnostics.start()
        self.diagnostics.lifecycle("connection", "started")
        try:
            self._pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            self._microphone = MicrophoneTrack(diagnostics=self.diagnostics)
            self._pc.addTrack(self._microphone)
            self._channel = self._pc.createDataChannel(CHANNEL)

            def on_open() -> None:
                while self.ready.is_set() and self._pending:
                    self._channel.send(self._pending.popleft())

            def on_message(message: Any) -> None:
                if not isinstance(message, str) or len(message.encode()) > MAX_BODY:
                    return
                try:
                    obj = json.loads(message)
                except ValueError:
                    return
                if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
                    return
                if obj["type"] in ("session.started", "session.updated"):
                    self.ready.set()
                    on_open()
                self._enqueue(SimpleNamespace(**obj))

            def on_track(track: Any) -> None:
                if track.kind == "audio":
                    self._audio_readers.append(
                        asyncio.create_task(self._read_remote_audio(track, av, MediaStreamError))
                    )

            def on_close() -> None:
                if not self._closed:
                    self._enqueue(SimpleNamespace(type="session.closed"))

            def on_state() -> None:
                if not self._closed and self._pc.connectionState == "failed":
                    self._enqueue(SimpleNamespace(type="session.closed"))

            self._channel.on("open", on_open)
            self._channel.on("message", on_message)
            self._channel.on("close", on_close)
            self._pc.on("track", on_track)
            self._pc.on("connectionstatechange", on_state)

            await self._pc.setLocalDescription(await self._pc.createOffer())
            body = await self.adapter.signal(
                {
                    "sdp": self._pc.localDescription.sdp,
                    "billing": "subscription",
                    "model": self.adapter.model,
                    "voice": self.adapter.voice,
                    "briefing": self.adapter.instructions,
                }
            )
            if body.get("ok") is not True or body.get("auth") != "subscription":
                raise RuntimeError("Broker did not confirm subscription billing; no API fallback was used.")
            transport = body.get("transport")
            if (
                not isinstance(transport, dict)
                or transport.get("type") != "webrtc"
                or not isinstance(transport.get("sdp"), str)
                or not transport["sdp"].startswith("v=0")
                or len(transport["sdp"].encode()) > 65536
            ):
                raise RuntimeError("Subscription broker returned an invalid WebRTC answer.")
            await self._pc.setRemoteDescription(RTCSessionDescription(sdp=transport["sdp"], type="answer"))
            self._stats_task = asyncio.create_task(self._monitor_stats(), name="voice-audio-diagnostics-stats")
            await asyncio.wait_for(self.ready.wait(), self.adapter.ready_timeout)
            self.diagnostics.lifecycle("connection", "completed")
            return self
        except BaseException:
            self.diagnostics.lifecycle("connection", "failed")
            await self.close()
            raise

    async def _collect_stats(self) -> None:
        """Observe real aiortc outbound audio stats without retaining its report."""
        pc: RTCPeerConnection | None = self._pc
        if pc is None:
            return
        try:
            report = await pc.getStats()
            packets = 0
            size = 0
            for stat in report.values():
                if isinstance(stat, RTCOutboundRtpStreamStats) and stat.kind == "audio":
                    packets += stat.packetsSent
                    size += stat.bytesSent
            self.diagnostics.webrtc(
                packets_sent=packets,
                bytes_sent=size,
                connection_state=pc.connectionState,
                ice_state=pc.iceConnectionState,
                signaling_state=pc.signalingState,
                channel_state=self._channel.readyState if self._channel is not None else "unknown",
            )
        except Exception:
            self.diagnostics.stats_failed()

    async def _monitor_stats(self) -> None:
        """Own a low-frequency observer, never another provider connection."""
        while not self._closed:
            await self._collect_stats()
            await asyncio.sleep(5)

    async def _read_remote_audio(self, track: Any, av: Any, media_stream_error: type[Exception]) -> None:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
        try:
            while not self._closed:
                frame = await track.recv()
                for converted in resampler.resample(frame):
                    pcm = bytes(converted.planes[0])[: converted.samples * 2]
                    if pcm:
                        self._enqueue(
                            SimpleNamespace(type="response.audio.delta", delta=base64.b64encode(pcm).decode("ascii"))
                        )
        except media_stream_error:
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            self._enqueue(
                SimpleNamespace(
                    type="error", error={"code": "webrtc_audio", "message": "Subscription audio stream ended."}
                )
            )

    async def _append_audio(self, *, audio: str) -> None:
        if self._microphone is not None and not self._closed:
            pcm = base64.b64decode(audio, validate=True)
            self.diagnostics.audio("append", np.frombuffer(pcm, dtype=np.int16), 24000)
            self._microphone.feed(pcm)
            self.diagnostics.flow("append", input_bytes=len(pcm), backlog_bytes=len(self._microphone.backlog))
        else:
            self.diagnostics.drop("connection_closed" if self._closed else "no_microphone")

    async def send_event(self, message: dict[str, Any]) -> None:
        """Queue pre-open sends, failing explicitly rather than dropping context."""
        if self._closed:
            raise ConnectionError("Subscription session is closed")
        text = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        if self.ready.is_set() and self._channel is not None and self._channel.readyState == "open":
            self._channel.send(text)
        elif len(self._pending) < 64:
            self._pending.append(text)
        else:
            raise BufferError("Subscription pending context queue is full")

    async def send_context(self, text: str, channel: str, delegation_id: str | None = None) -> None:
        """Send quiet or speakable context using the actual consumer dialect."""
        for message in build_context_messages(text, channel=channel, delegation_id=delegation_id):
            await self.send_event(message)

    async def plan(self, transcript: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Ask only the trusted host's subscription-backed bounded app planner."""
        return await broker_request(
            {"billing": "subscription", "transcript": transcript, "history": history}, operation="plan"
        )

    async def close(self) -> None:
        """End the consumer session, all reader tasks, media and queued input."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._pc is not None:
                try:
                    await asyncio.wait_for(self._collect_stats(), timeout=0.25)
                except asyncio.TimeoutError:
                    self.diagnostics.stats_failed()
            if self._channel is not None and self._channel.readyState == "open":
                self._channel.send(json.dumps({"type": "session.close"}))
        finally:
            try:
                self.ready.clear()
                if self._stats_task is not None:
                    self._stats_task.cancel()
                    await asyncio.gather(self._stats_task, return_exceptions=True)
                for task in self._audio_readers:
                    task.cancel()
                await asyncio.gather(*self._audio_readers, return_exceptions=True)
                if self._microphone is not None:
                    self._microphone.stop()
                if self._pc is not None:
                    await self._pc.close()
            finally:
                self._pending.clear()
                for queue in (self._events, self._audio_events):
                    while not queue.empty():
                        queue.get_nowait()
                self._enqueue(None)
                self.diagnostics.lifecycle("connection", "closed")
                if self._owns_diagnostics:
                    self.diagnostics.close()

    async def __aexit__(self, *_args: Any) -> None:
        """Close on normal exit as well as exceptions."""
        await self.close()

    def __aiter__(self) -> SubscriptionRealtimeConnection:
        """Iterate decoded consumer events and resampled PCM."""
        return self

    async def __anext__(self) -> Any:
        """Attach only the current unconsumed user turn to a delegation."""
        while self._events.empty() and self._audio_events.empty():
            if self._closed:
                raise StopAsyncIteration
            self._available.clear()
            await self._available.wait()
        event = self._events.get_nowait() if not self._events.empty() else self._audio_events.get_nowait()
        if event is None:
            raise StopAsyncIteration
        kind = event.type
        if kind == "session.closed":
            raise ConnectionError("Subscription Live session closed.")
        if kind == "input_transcript.added":
            if not self._current_input:
                self._turn_id += 1
                self._awaiting_final = True
            self._current_input = (self._current_input + self._transcript(event))[-6000:]
            self._user_transcript = self._current_input
        elif kind == "turn.done":
            turn = getattr(event, "turn", {})
            if isinstance(turn, dict) and turn.get("role") == "user" and isinstance(turn.get("transcript"), str):
                if not self._awaiting_final:
                    self._turn_id += 1
                self._user_transcript = turn["transcript"][-6000:]
                self._current_input = ""
                self._awaiting_final = False
        elif kind == "delegation.created":
            event.transcript_context = self._user_transcript if self._consumed_turn != self._turn_id else ""
            self._consumed_turn = self._turn_id
            self._current_input = ""
        event.turn_id = self._turn_id
        return event

    @staticmethod
    def _transcript(event: Any) -> str:
        item = getattr(event, "item", {})
        return str(item.get("text", "")) if isinstance(item, dict) else ""


async def qualify_subscription() -> dict[str, Any]:
    """One headless consumer handshake, no recording, speaker, greeting or retry."""
    async with SubscriptionRealtimeAdapter(
        instructions="Wait silently for the user. Do not greet or delegate."
    ).connect() as connection:
        return {
            "ok": connection.ready.is_set(),
            "auth": "subscription",
            "handshakes": 1,
            "recording": False,
            "playback": False,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="One silent subscription Live handshake; may consume account entitlement."
    )
    parser.add_argument("--qualify", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(asyncio.run(qualify_subscription())))
