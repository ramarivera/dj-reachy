"""Narrow authenticated trusted-host broker, without Hermes or API-key fallback.

Consumer signaling is adapted from Conduit plugin_api.py, MIT. See the pinned
sources in static/conduit-voice-NOTICE.txt. Provider credentials never leave here.
"""

from __future__ import annotations
import os
import re
import ssl
import hmac
import json
import time
import base64
import argparse
import threading
from typing import Any, Callable
from pathlib import Path
from dataclasses import field, dataclass
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import httpx

from dj_reachy.subscription_voice import MAX_BODY, DEFAULT_MODEL, DEFAULT_VOICE, LIVE_ENDPOINT


DEFAULT_PLANNER_MODEL = "gpt-6.1-sol"


LIVE_INSTRUCTIONS = (
    "You are DJ Reachy. Converse naturally in the user's language. "
    "Delegate requests to make or sing songs, play saved songs, pause, resume, stop music, dance, or answer camera questions to the client. "
    "Client delegation is the only way to execute these supported app actions; you do not have the app tools directly. "
    "After the user confirms a pending supported action, immediately create a client delegation carrying that action and its current constraints. "
    "A spoken promise does not execute an app action. Do not substitute acknowledgment, narration or another generic question for delegation. "
    "Natural confirmation does not require any particular word. "
    "First describe the requested track and duration and obtain explicit user confirmation before any paid music generation. "
    "Never generate another track automatically. Current refusals, stop and cancellation requests take priority. "
    "Say that generation has started only after the client reports it actually started. "
    "The camera is available through the client; never claim to see its view before the image result. "
    "Song generation is asynchronous: continue conversing while it runs. "
    "Announce that the song is ready only after the explicit generation-completed notice arrives. "
    "The client has only bounded app tools, no shell, files, browsing or general agent. "
    "Ask for clarification for unsupported tasks. Never claim a command succeeded before its result arrives. "
    "Commentary is quiet state, not a new user request. Speakable context is a result to tell the user. "
    "Follow the startup behavior in the conversation instructions below. "
    "If no startup behavior is specified, wait for the user."
)


class BrokerError(RuntimeError):
    """Safe, nonsecret error carrying the local HTTP status."""

    def __init__(self, status: int, message: str):
        """Keep upstream bodies and credential values out of errors."""
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class CodexCredential:
    """Subscription OAuth plus account identity, kept on the trusted host."""

    access_token: str = field(repr=False)
    account_id: str = field(repr=False)

    def headers(self) -> dict[str, str]:
        """Reject malformed credentials rather than consulting an API key."""
        if (
            not self.access_token
            or self.access_token.startswith("sk-")
            or not self.account_id
            or not (self.access_token + self.account_id).isascii()
            or any(c.isspace() for c in self.access_token + self.account_id)
        ):
            raise BrokerError(503, "A valid ChatGPT Codex OAuth sign-in is required; no API fallback was used.")
        return {"Authorization": f"Bearer {self.access_token}", "ChatGPT-Account-Id": self.account_id}


def codex_file_resolver(path: Path) -> Callable[[], CodexCredential]:
    """Build an explicit local Codex file resolver. Never auto-discover or refresh auth."""

    def resolve() -> CodexCredential:
        try:
            if path.stat().st_size > 65536:
                raise ValueError
            body = json.loads(path.read_text())
            if not isinstance(body, dict) or body.get("auth_mode") not in (None, "chatgpt"):
                raise ValueError
            tokens = body["tokens"]
            access = tokens["access_token"]
            account = tokens["account_id"]
            if not isinstance(access, str) or not isinstance(account, str):
                raise ValueError
            # A stale local sign-in fails closed. Run Codex login on the host to renew it.
            if access.count(".") == 2:
                payload = access.split(".")[1]
                claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
                if isinstance(claims.get("exp"), (int, float)) and claims["exp"] <= time.time() + 30:
                    raise ValueError
            credential = CodexCredential(access, account)
            credential.headers()
            return credential
        except (OSError, ValueError, KeyError, TypeError, BrokerError):
            raise BrokerError(
                503,
                "Explicit Codex auth file is missing, expired or not a ChatGPT sign-in. Renew it on the trusted host; no API fallback was used.",
            ) from None

    return resolve


def _text(body: dict[str, Any], key: str, default: str, maximum: int) -> str:
    value = body.get(key, default)
    if not isinstance(value, str) or len(value.encode()) > maximum:
        raise BrokerError(400, f"Invalid {key}")
    return value


class BrokerService:
    """Resolve host credentials and exchange only approved session/planner requests."""

    def __init__(
        self,
        resolver: Callable[[], CodexCredential],
        *,
        live_endpoint: str = LIVE_ENDPOINT,
        responses_endpoint: str = "https://chatgpt.com/backend-api/codex/responses",
        planner_model: str = DEFAULT_PLANNER_MODEL,
    ):
        """Accept a pluggable resolver; endpoint overrides are for loopback fixtures."""
        self.resolver = resolver
        self.live_endpoint = live_endpoint
        self.responses_endpoint = responses_endpoint
        self.planner_model = planner_model

    def session(self, body: dict[str, Any]) -> dict[str, Any]:
        """Perform one consumer SDP exchange with the exact pinned Conduit session shape."""
        if set(body) - {"billing", "sdp", "model", "voice", "briefing"} or body.get("billing") != "subscription":
            raise BrokerError(400, "Only explicit subscription session requests are accepted")
        sdp = _text(body, "sdp", "", 65536)
        if not sdp.startswith("v=0"):
            raise BrokerError(400, "Invalid SDP offer")
        model = _text(body, "model", DEFAULT_MODEL, 80)
        voice = _text(body, "voice", DEFAULT_VOICE, 32)
        if model != DEFAULT_MODEL or re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", voice) is None:
            raise BrokerError(400, "Unsupported subscription Live model or voice")
        briefing = _text(body, "briefing", "", 16000)
        payload = {
            "sdp": sdp,
            "session": {
                "model": model,
                "instructions": LIVE_INSTRUCTIONS + "\n\n" + briefing,
                "audio": {"output": {"voice": voice}},
                "delegation": {"type": "client"},
            },
        }
        headers = {**self.resolver().headers(), "OpenAI-Alpha": "quicksilver=v2"}
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=30) as client:
                with client.stream("POST", self.live_endpoint, headers=headers, json=payload) as response:
                    if not 200 <= response.status_code < 300:
                        raise BrokerError(
                            502,
                            f"Subscription Live rejected the session (HTTP {response.status_code}); check account voice entitlement. No API fallback was used.",
                        )
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        raw.extend(chunk)
                        if len(raw) > 65536:
                            raise BrokerError(502, "Invalid upstream SDP answer")
                    answer = raw.decode("utf-8")
                    call_id = response.headers.get("Location", "").rstrip("/").rsplit("/", 1)[-1]
            if not answer.startswith("v=0") or re.fullmatch(r"rtc_[A-Za-z0-9_-]+|[0-9a-fA-F-]{36}", call_id) is None:
                raise BrokerError(502, "Invalid upstream SDP answer or session identity")
        except (httpx.HTTPError, UnicodeError):
            raise BrokerError(502, "Subscription Live transport failed; no API fallback was used.") from None
        return {
            "ok": True,
            "auth": "subscription",
            "session": {"id": call_id},
            "transport": {"type": "webrtc", "sdp": answer},
            "briefing_applied": True,
            "greeting_applied": False,
            "model": model,
            "voice": voice,
        }

    def plan(self, body: dict[str, Any]) -> dict[str, Any]:
        """Execute only the bounded subscription Responses planning round."""
        from dj_reachy.voice_planner import subscription_plan

        return subscription_plan(body, self.resolver, endpoint=self.responses_endpoint, model=self.planner_model)


def create_server(
    service: BrokerService, token: str, *, host: str = "127.0.0.1", port: int = 8766, tls: ssl.SSLContext | None = None
) -> ThreadingHTTPServer:
    """Bind loopback by default; remote listeners require TLS and every request a scoped token."""
    if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("Broker token must have at least 32 non-whitespace ASCII characters")
    if host not in ("127.0.0.1", "localhost", "::1") and tls is None:
        raise ValueError("Remote broker listeners require TLS")
    slots = threading.BoundedSemaphore(4)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            # HTTP access logging can expose credentials, prompts and account error bodies.
            return

        def reply(self, status: int, value: dict[str, Any]) -> None:
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            self.connection.settimeout(10)
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
                self.reply(401, {"ok": False, "error": "Scoped broker authentication required"})
                return
            if self.path not in ("/session", "/plan"):
                self.reply(404, {"ok": False, "error": "Unknown broker operation"})
                return
            if not slots.acquire(blocking=False):
                self.reply(429, {"ok": False, "error": "Broker is busy"})
                return
            try:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise BrokerError(400, "Invalid body length") from None
                if not 0 < length <= MAX_BODY or self.headers.get("Transfer-Encoding"):
                    raise BrokerError(413, "Request body exceeds limit or has unsupported framing")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise BrokerError(400, "Incomplete request body")
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise BrokerError(400, "Request must be an object")
                result = service.session(body) if self.path == "/session" else service.plan(body)
                self.reply(200, result)
            except BrokerError as exc:
                self.reply(exc.status, {"ok": False, "error": str(exc)})
            except (ValueError, UnicodeError):
                self.reply(400, {"ok": False, "error": "Invalid request JSON"})
            except Exception:
                self.reply(502, {"ok": False, "error": "Subscription broker failed; no API fallback was used."})
            finally:
                slots.release()

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    if tls is not None:
        server.socket = tls.wrap_socket(server.socket, server_side=True)
    return server


def main() -> None:
    """Run the standalone host broker using explicit environment-only credential settings."""
    parser = argparse.ArgumentParser(description="Trusted-host subscription-only Live and app planner broker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    args = parser.parse_args()
    auth_file = os.environ.get("DJ_REACHY_CODEX_AUTH_FILE")
    if not auth_file:
        parser.error("Set DJ_REACHY_CODEX_AUTH_FILE to the host's explicit Codex ChatGPT auth file")
    tls = None
    if args.tls_cert or args.tls_key:
        if not args.tls_cert or not args.tls_key:
            parser.error("Both TLS certificate and private key are required")
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.minimum_version = ssl.TLSVersion.TLSv1_2
        tls.load_cert_chain(args.tls_cert, args.tls_key)
    service = BrokerService(
        codex_file_resolver(Path(auth_file)),
        planner_model=os.environ.get("DJ_REACHY_SUBSCRIPTION_PLANNER_MODEL", DEFAULT_PLANNER_MODEL),
    )
    server = create_server(
        service, os.environ.get("DJ_REACHY_SUBSCRIPTION_BROKER_TOKEN", ""), host=args.host, port=args.port, tls=tls
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
