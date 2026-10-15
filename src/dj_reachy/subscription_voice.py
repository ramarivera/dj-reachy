"""ChatGPT subscription Live adapter using an explicitly configured broker.

The broker owns Codex credentials and confirms subscription billing before it
returns WebRTC SDP. This client never falls back to API-key Realtime.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from collections import deque
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import requests

BROKER_URL_ENV = "DJ_REACHY_SUBSCRIPTION_BROKER_URL"
BROKER_TOKEN_ENV = "DJ_REACHY_SUBSCRIPTION_BROKER_TOKEN"
LIVE_ENDPOINT = "https://chatgpt.com/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas"
DEFAULT_MODEL = "gpt-live-1-codex"
DEFAULT_VOICE = "cove"
CHANNEL = "oai-events"
MAX_CONTEXT_BYTES = 500


def require_subscription_broker() -> tuple[str, str]:
    """Return a validated broker endpoint and scoped broker credential."""
    url = os.environ.get(BROKER_URL_ENV, "").strip()
    token = os.environ.get(BROKER_TOKEN_ENV, "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError(f"Subscription voice needs an authenticated broker URL in {BROKER_URL_ENV}.")
    if parsed.scheme != "https" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("Subscription broker URL must use HTTPS unless it is loopback.")
    if not token or any(char in token for char in "\r\n"):
        raise RuntimeError(f"Subscription voice needs a scoped broker token in {BROKER_TOKEN_ENV}; no API fallback is used.")
    return url, token


def _utf8_chunks(text: str, limit: int = MAX_CONTEXT_BYTES) -> list[str]:
    text = text.strip()
    chunks: list[str] = []
    current = ""
    size = 0
    for char in text:
        n = len(char.encode("utf-8"))
        if size + n > limit and current:
            chunks.append(current)
            current = ""
            size = 0
        current += char
        size += n
    if current:
        chunks.append(current)
    return chunks


def build_context_messages(text: str, *, channel: str, delegation_id: str | None = None) -> list[dict[str, Any]]:
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


def plan_delegation(event_text: str, transcript: str) -> tuple[str, dict[str, Any]] | None:
    """Map a delegation to one bounded app command using transcript context."""
    text = (event_text or transcript).strip()
    if not text:
        text = transcript.strip()
    normalized = text.lower()
    if re.search(r"\b(pause|hold)\b.*\b(music|song|track)\b|\b(pause|hold)\b", normalized):
        return "pause_music", {}
    if re.search(r"\b(resume|continue|unpause)\b.*\b(music|song|track)\b|\b(resume|unpause)\b", normalized):
        return "resume_music", {}
    if re.search(r"\b(stop|cancel)\b.*\b(music|song|track|generation)\b|\bstop\b", normalized):
        return "stop_music", {}
    if re.search(r"\b(make|create|generate|compose)\b.*\b(song|music|track)\b", normalized):
        return "make_song_and_dance", {"prompt": text[:2000], "music_length_ms": 30000}
    if re.search(r"\b(dance|move|perform)\b", normalized):
        return "dance", {}
    return None


class SubscriptionRealtimeAdapter:
    """A broker-signaled WebRTC client compatible with this app handler's realtime ports."""

    def __init__(self, *, model: str = DEFAULT_MODEL, voice: str = DEFAULT_VOICE, instructions: str = ""):
        self.model = model or DEFAULT_MODEL
        self.voice = voice or DEFAULT_VOICE
        self.instructions = instructions
        self.realtime = SimpleNamespace(connect=self.connect)

    def connect(self, **_kwargs: Any) -> "SubscriptionRealtimeConnection":
        return SubscriptionRealtimeConnection(self)


class SubscriptionRealtimeConnection:
    def __init__(self, adapter: SubscriptionRealtimeAdapter):
        self.adapter = adapter
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self._mic_chunks: deque[bytes] = deque()
        self._mic_size = 0
        self._audio_readers: list[asyncio.Task[None]] = []
        self._closed = False
        self._user_transcript = ""
        self._remote_track: Any = None
        self._microphone: Any = None
        self._pc: Any = None
        self._channel: Any = None
        self.session = SimpleNamespace(update=self._session_update)
        self.input_audio_buffer = SimpleNamespace(append=self._append_audio)
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=self._conversation_item_create))
        self.response = SimpleNamespace(create=self._response_create, cancel=self._response_cancel)

    async def __aenter__(self) -> "SubscriptionRealtimeConnection":
        broker_url, broker_token = require_subscription_broker()
        try:
            import av
            from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
            from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
        except ImportError as exc:
            raise RuntimeError("Subscription Live requires the installed aiortc and av runtime.") from exc

        owner = self

        class Microphone(AudioStreamTrack):
            def __init__(self) -> None:
                super().__init__()
                self.resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
                self.backlog = bytearray()
                self.timestamp = 0
                self.started = time.monotonic()

            async def recv(self) -> Any:
                self.timestamp += 960
                await asyncio.sleep(max(0.0, self.started + self.timestamp / 48000 - time.monotonic()))
                while len(self.backlog) < 1920 and owner._mic_chunks:
                    block = owner._mic_chunks.popleft()
                    owner._mic_size -= len(block)
                    samples = len(block) // 2
                    if samples:
                        frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
                        frame.planes[0].update(block[: samples * 2])
                        frame.sample_rate = 24000
                        self.backlog.extend(b"".join(bytes(out.planes[0])[: out.samples * 2] for out in self.resampler.resample(frame)))
                raw = bytes(self.backlog[:1920])
                del self.backlog[: len(raw)]
                frame = av.AudioFrame(format="s16", layout="mono", samples=960)
                frame.planes[0].update(raw.ljust(1920, b"\0"))
                frame.pts = self.timestamp
                frame.sample_rate = 48000
                frame.time_base = __import__("fractions").Fraction(1, 48000)
                return frame

        self._pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._microphone = Microphone()
        self._pc.addTrack(self._microphone)
        self._channel = self._pc.createDataChannel(CHANNEL)

        @self._channel.on("message")
        def on_message(message: Any) -> None:
            if isinstance(message, str):
                try:
                    obj = json.loads(message)
                    if isinstance(obj, dict) and isinstance(obj.get("type"), str):
                        self._events.put_nowait(SimpleNamespace(**obj))
                except (json.JSONDecodeError, TypeError):
                    return

        @self._pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio":
                self._audio_readers.append(asyncio.create_task(self._read_remote_audio(track, av, MediaStreamError)))
        @self._channel.on("close")
        def on_channel_close() -> None:
            self._events.put_nowait(SimpleNamespace(type="session.closed"))

        await self._pc.setLocalDescription(await self._pc.createOffer())
        offer = self._pc.localDescription.sdp
        try:
            response = await asyncio.to_thread(
                requests.post,
                broker_url,
                json={"sdp": offer, "billing": "subscription", "voice": self.adapter.voice,
                      "briefing": self.adapter.instructions},
                headers={"Authorization": f"Bearer {broker_token}", "Content-Type": "application/json"},
                timeout=35,
            )
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(f"Subscription broker refused the session (HTTP {response.status_code}); no API fallback was used.")
            body = response.json()
            if not isinstance(body, dict) or body.get("ok") is not True or body.get("auth") != "subscription":
                raise RuntimeError("Broker did not confirm subscription billing; no API fallback was used.")
            transport = body.get("transport")
            if not isinstance(transport, dict) or transport.get("type") != "webrtc" or not str(transport.get("sdp", "")).startswith("v=0"):
                raise RuntimeError("Subscription broker returned an invalid WebRTC answer.")
            await self._pc.setRemoteDescription(RTCSessionDescription(sdp=transport["sdp"], type="answer"))
            return self
        except (requests.RequestException, ValueError) as exc:
            await self.close()
            raise RuntimeError(f"Subscription broker connection failed ({type(exc).__name__}); no API fallback was used.") from None
        except Exception:
            await self.close()
            raise

    async def _read_remote_audio(self, track: Any, av: Any, media_stream_error: type) -> None:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
        try:
            while not self._closed:
                frame = await track.recv()
                for converted in resampler.resample(frame):
                    pcm = b"".join(bytes(converted.planes[0])[: converted.samples * 2])
                    if pcm:
                        event = SimpleNamespace(type="response.audio.delta", delta=base64.b64encode(pcm).decode("ascii"))
                        await self._events.put(event)
        except media_stream_error:
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._events.put(SimpleNamespace(type="error", error=SimpleNamespace(code="webrtc_audio", message="Subscription audio stream ended.")))

    async def _session_update(self, **_kwargs: Any) -> None:
        return None

    async def _conversation_item_create(self, *, item: dict[str, Any]) -> None:
        content = item.get("content", [])
        text = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        await self.send_context(text, "commentary")

    async def _response_create(self, *, response: dict[str, Any]) -> None:
        await self.send_event({"type": "response.create", "response": response})

    async def _response_cancel(self, **_kwargs: Any) -> None:
        await self.send_event({"type": "response.cancel"})

    async def _append_audio(self, *, audio: str) -> None:
        pcm = base64.b64decode(audio)
        self._mic_chunks.append(pcm)
        self._mic_size += len(pcm)
        max_bytes = 24000 * 2 * 2
        while self._mic_size > max_bytes and self._mic_chunks:
            self._mic_size -= len(self._mic_chunks.popleft())

    async def send_event(self, message: dict[str, Any]) -> None:
        if self._channel is None or self._channel.readyState != "open":
            return
        self._channel.send(json.dumps(message, ensure_ascii=False, separators=(",", ":")))

    async def send_context(self, text: str, channel: str, delegation_id: str | None = None) -> None:
        for message in build_context_messages(text, channel=channel, delegation_id=delegation_id):
            await self.send_event(message)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self._audio_readers:
            task.cancel()
        await asyncio.gather(*self._audio_readers, return_exceptions=True)
        if self._pc is not None:
            await self._pc.close()
        await self._events.put(None)

    async def __aexit__(self, *_args: Any) -> None:
        await self.close()
    def __aiter__(self) -> "SubscriptionRealtimeConnection":
        return self

    async def __anext__(self) -> Any:
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        event_type = getattr(event, "type", None)
        if event_type == "session.closed":
            raise ConnectionError("Subscription Live session closed.")
        if event_type == "input_transcript.added":
            text = self._transcript(event)
            self._user_transcript = (self._user_transcript + " " + text).strip()[-6000:]
        if event_type == "turn.done":
            turn = getattr(event, "turn", {})
            role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", None)
            transcript = turn.get("transcript") if isinstance(turn, dict) else getattr(turn, "transcript", "")
            if role == "user" and isinstance(transcript, str):
                self._user_transcript = transcript[-6000:]
        if event_type == "delegation.created":
            event.transcript_context = self._user_transcript
        return event

    @staticmethod
    def _transcript(event: Any) -> str:
        item = getattr(event, "item", {})
        if isinstance(item, dict):
            return str(item.get("text") or "")
        return str(getattr(event, "text", "") or "")
