"""Loopback broker contract checks. Credentials are explicit synthetic fixtures."""

from __future__ import annotations
import json
import threading
from typing import NoReturn
from pathlib import Path
from contextlib import contextmanager
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from collections.abc import Iterator

import httpx
import pytest


@contextmanager
def serving(server: ThreadingHTTPServer) -> Iterator[str]:
    """Own a loopback HTTP server for one test."""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_broker_auth_contains_credentials_and_preserves_consumer_contract() -> None:
    """Reject unauthenticated calls before resolving OAuth; forward exact session config."""
    from dj_reachy_remix.subscription_broker import BrokerService, CodexCredential, create_server

    observed = []
    resolved = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            observed.append((dict(self.headers), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(201)
            self.send_header("Location", "/backend-api/codex/realtime/calls/rtc_fixture")
            self.end_headers()
            self.wfile.write(b"v=0\r\nfixture-answer\r\n")

        def log_message(self, format: str, *args: object) -> None:
            pass

    def credential() -> CodexCredential:
        resolved.append(True)
        return CodexCredential("fixture-oauth-private", "fixture-account-private")

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
        service = BrokerService(credential, live_endpoint=upstream)
        token = "fixture-scoped-broker-token-32-characters"
        with serving(create_server(service, token, port=0)) as broker:
            with httpx.Client(trust_env=False) as client:
                rejected = client.post(broker + "/session", json={"billing": "subscription", "sdp": "v=0\r\n"})
                assert rejected.status_code == 401
                assert not resolved and not observed
                response = client.post(
                    broker + "/session",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "billing": "subscription",
                        "sdp": "v=0\r\no=fixture\r\n",
                        "model": "gpt-live-1-codex",
                        "voice": "cove",
                        "briefing": "Speak Spanish. Greet the user briefly, then listen. Delegate app commands.",
                    },
                )
                assert response.status_code == 200
                body = response.json()
                assert body["auth"] == "subscription"
                assert body["transport"]["sdp"].startswith("v=0")
                assert "fixture-oauth-private" not in response.text
                assert "fixture-account-private" not in response.text
                assert "fixture-scoped" not in response.text
                headers, request = observed[0]
                assert headers["Authorization"] == "Bearer fixture-oauth-private"
                assert headers["ChatGPT-Account-Id"] == "fixture-account-private"
                assert headers["OpenAI-Alpha"] == "quicksilver=v2"
                assert request["session"]["model"] == "gpt-live-1-codex"
                assert request["session"]["audio"] == {"output": {"voice": "cove"}}
                assert request["session"]["delegation"] == {"type": "client"}
                assert "tools" not in request["session"]
                assert "Speak Spanish." in request["session"]["instructions"]
                assert "Greet the user briefly, then listen." in request["session"]["instructions"]
                assert "Do not greet first." not in request["session"]["instructions"]
                # This checks wire delivery, not whether the voice model obeys it.
                assert (
                    "Delegate every new music request to the client immediately to prepare an unbilled draft"
                    in request["session"]["instructions"]
                )
                assert "A spoken promise does not execute an app action." in request["session"]["instructions"]
                assert "obtain explicit user confirmation before any paid music generation" in request["session"]["instructions"]
                assert "Never generate another track automatically." in request["session"]["instructions"]
                for malformed in ({"billing": "api", "sdp": "v=0"}, {"billing": "subscription", "sdp": "x" * 70000}):
                    result = client.post(
                        broker + "/session", headers={"Authorization": f"Bearer {token}"}, json=malformed
                    )
                    assert result.status_code == 400
                assert len(resolved) == len(observed) == 1


def test_remote_broker_requires_tls_and_strong_token() -> None:
    """Remote plaintext listeners and absent authentication are never allowed."""
    from dj_reachy_remix.subscription_broker import BrokerService, CodexCredential, create_server

    service = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account"))
    with pytest.raises(ValueError, match="TLS"):
        create_server(service, "x" * 32, host="0.0.0.0")
    with pytest.raises(ValueError, match="token"):
        create_server(service, "")


def test_subscription_planner_sse_tool_round_and_no_arbitrary_authority() -> None:
    """Use actual HTTP/SSE, retain tool IDs/results, reject unsafe model calls."""
    from dj_reachy_remix.subscription_broker import BrokerService, CodexCredential, create_server

    requests = []
    call: dict[str, object] = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_song",
        "name": "make_song_and_dance",
        "namespace": "dj_reachy_remix",
        "arguments": json.dumps({"prompt": "Sing a salsa about robots", "music_length_ms": 10000}),
    }

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((dict(self.headers), body))
            output: list[dict[str, object]] = (
                [call]
                if len(requests) == 1
                else [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "The local song finished."}],
                    }
                ]
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # Real providers duplicate the completed item in the final response.
            events: list[dict[str, object]] = [
                {"type": "response.output_item.done", "item": output[0]},
                {"type": "response.completed", "response": {"id": "r1", "status": "completed", "output": output}},
            ]
            for event in events:
                raw = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                for start in range(0, len(raw), 17):
                    self.wfile.write(raw[start : start + 17])
                    self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            pass

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
        service = BrokerService(
            lambda: CodexCredential("fixture-provider", "fixture-account"), responses_endpoint=upstream
        )
        token = "test-broker-token-not-a-provider-key-32"
        with serving(create_server(service, token, port=0)) as broker:
            with httpx.Client(trust_env=False) as client:
                headers = {"Authorization": "Bearer " + token}
                first = client.post(
                    broker + "/plan",
                    headers=headers,
                    json={
                        "billing": "subscription",
                        "transcript": "sing me a ten second salsa about robots",
                        "history": [],
                    },
                )
                assert first.status_code == 200, first.text
                step = first.json()
                assert step["calls"] == [
                    {
                        "call_id": "call_song",
                        "name": "make_song_and_dance",
                        "arguments": {"prompt": "Sing a salsa about robots", "music_length_ms": 10000},
                    }
                ]
                output = {"type": "function_call_output", "call_id": "call_song", "output": '{"status":"completed"}'}
                second = client.post(
                    broker + "/plan",
                    headers=headers,
                    json={
                        "billing": "subscription",
                        "transcript": "sing me a ten second salsa about robots",
                        "history": [*step["output"], output],
                    },
                )
                assert second.status_code == 200
                assert second.json()["calls"] == []
                assert second.json()["text"] == "The local song finished."
                for captured_headers, body in requests:
                    assert captured_headers["Authorization"] == "Bearer fixture-provider"
                    assert captured_headers["ChatGPT-Account-Id"] == "fixture-account"
                    assert body["store"] is False and body["stream"] is True
                    assert "max_output_tokens" not in body and "previous_response_id" not in body
                    assert body["tools"][0]["type"] == "namespace"
                    names = {tool["name"] for tool in body["tools"][0]["tools"]}
                    assert "make_song_and_dance" in names and "shell" not in names
                assert requests[1][1]["input"][-1] == output
                assert requests[1][1]["input"][-2]["call_id"] == "call_song"
                assert "fixture-provider" not in first.text + second.text


def test_app_calls_reject_unbounded_or_unknown_commands() -> None:
    """Treat model/tool payloads as untrusted, including tools in broker history."""
    from dj_reachy_remix.voice_planner import validate_app_call

    for name, args in (
        ("shell", {"command": "id"}),
        ("make_song_and_dance", {"prompt": "x", "music_length_ms": True}),
        ("dance", {"repeat": 999}),
        ("play_saved_song", {"filename": "../auth.json"}),
        ("pause_music", {"extra": "ignored?"}),
    ):
        with pytest.raises(ValueError):
            validate_app_call(name, args)


def test_explicit_codex_fixture_resolver_never_uses_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read only an injected fixture path, and reject an API-key or expired sign-in."""
    from dj_reachy_remix.subscription_broker import BrokerError, codex_file_resolver

    path = tmp_path / "fixture-auth.json"
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-api-never-read")
    resolve = codex_file_resolver(path)
    for body in (
        {"auth_mode": "apikey", "OPENAI_API_KEY": "fixture-api"},
        {"auth_mode": "chatgpt", "tokens": {"access_token": "sk-fixture", "account_id": "fixture"}},
    ):
        path.write_text(json.dumps(body))
        with pytest.raises(BrokerError, match="ChatGPT"):
            resolve()
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {"access_token": "fixture-oauth-private", "account_id": "fixture-account-private"},
            }
        )
    )
    credentials = resolve()
    assert credentials.headers()["Authorization"] == "Bearer fixture-oauth-private"
    assert "fixture-oauth-private" not in repr(credentials)
    assert "fixture-account-private" not in repr(credentials)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.failed", "response": {"error": {"message": "fixture-private-secret"}}},
        {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "incomplete"}},
    ],
)
def test_planner_never_executes_partial_or_failed_sse(event: dict[str, object]) -> None:
    """EOF and partial function arguments aren't evidence of a completed tool call."""
    from dj_reachy_remix.voice_planner import _sse_completed
    from dj_reachy_remix.subscription_broker import BrokerError

    response = httpx.Response(200, content=("data: " + json.dumps(event) + "\n\n").encode())
    with pytest.raises(BrokerError) as failure:
        _sse_completed(response)
    assert "fixture-private-secret" not in str(failure.value)


def test_non_ascii_bad_authorization_is_rejected_before_credentials() -> None:
    """Malformed untrusted headers get 401, never a traceback or credential lookup."""
    from dj_reachy_remix.subscription_broker import BrokerService, create_server

    def forbidden() -> NoReturn:
        raise AssertionError("Credentials must not be resolved")

    with serving(create_server(BrokerService(forbidden), "fixture-scoped-token-32-characters", port=0)) as broker:
        with httpx.Client(trust_env=False) as client:
            response = client.post(broker + "/session", headers=[(b"Authorization", b"Bearer \xff")], json={})
            assert response.status_code == 401


@pytest.mark.asyncio
async def test_broker_client_reports_actionable_entitlement_without_provider_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate the host's safe diagnosis while keeping OAuth and upstream bodies host-side."""
    from dj_reachy_remix.subscription_voice import broker_request
    from dj_reachy_remix.subscription_broker import BrokerService, CodexCredential, create_server

    class RejectedProvider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"fixture-provider-private-body")

        def log_message(self, format: str, *args: object) -> None:
            pass

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), RejectedProvider)) as upstream:
        service = BrokerService(
            lambda: CodexCredential("fixture-private-oauth", "fixture-private-account"), live_endpoint=upstream
        )
        token = "fixture-scoped-broker-token-32-characters"
        with serving(create_server(service, token, port=0)) as broker:
            monkeypatch.setenv("DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_URL", broker + "/session")
            monkeypatch.setenv("DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_TOKEN", token)
            with pytest.raises(RuntimeError, match="entitlement") as failure:
                await broker_request({"billing": "subscription", "sdp": "v=0\r\n"})
            assert "HTTP 403" in str(failure.value)
            assert "fixture-" not in str(failure.value)


@pytest.mark.parametrize("override", [None, "explicit-fixture-model"])
def test_broker_planner_defaults_to_supported_subscription_model(
    monkeypatch: pytest.MonkeyPatch, override: str | None
) -> None:
    """Keep default and CLI selection aligned while preserving explicit overrides."""
    import sys

    import dj_reachy_remix.subscription_broker as broker

    def unused_credential() -> broker.CodexCredential:
        raise AssertionError("Selector testing must not resolve credentials")

    assert broker.BrokerService(unused_credential).planner_model == "gpt-6.1-sol"
    explicit = broker.BrokerService(unused_credential, planner_model="explicit-fixture-model")
    assert explicit.planner_model == "explicit-fixture-model"
    observed: list[str] = []

    class CapturedServer(Exception):
        """Stop before any socket opens."""

    def capture_server(service: broker.BrokerService, token: str, **kwargs: object) -> NoReturn:
        observed.append(service.planner_model)
        raise CapturedServer

    monkeypatch.setenv("DJ_REACHY_REMIX_CODEX_AUTH_FILE", "/nonexistent-fixture-auth.json")
    if override is None:
        monkeypatch.delenv("DJ_REACHY_REMIX_SUBSCRIPTION_PLANNER_MODEL", raising=False)
    else:
        monkeypatch.setenv("DJ_REACHY_REMIX_SUBSCRIPTION_PLANNER_MODEL", override)
    monkeypatch.setattr(sys, "argv", ["subscription-broker"])
    monkeypatch.setattr(broker, "create_server", capture_server)
    with pytest.raises(CapturedServer):
        broker.main()
    assert observed == [override or "gpt-6.1-sol"]


def test_subscription_session_uses_app_owned_preparation_and_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire a draft-first contract after persona text without broadening authority."""
    from dj_reachy_remix.subscription_broker import BrokerService, CodexCredential

    observed: list[dict[str, object]] = []
    original_client = httpx.Client

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        observed.append(body)
        return httpx.Response(201, headers={"Location": "/calls/rtc_fixture"}, content=b"v=0\r\nfixture-answer\r\n")

    def client(**kwargs: object) -> httpx.Client:
        return original_client(transport=httpx.MockTransport(upstream))

    monkeypatch.setattr(httpx, "Client", client)
    briefing = "PERSONA-FIXTURE: greet briefly in Spanish. Before calling make_song_and_dance ask for confirmation."
    service = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account"))
    service.session({"billing": "subscription", "sdp": "v=0\r\nfixture-offer", "briefing": briefing})
    session = observed[0]["session"]
    assert isinstance(session, dict)
    instructions = session["instructions"]
    assert isinstance(instructions, str)
    preparation = "Delegate every new music request to the client immediately to prepare an unbilled draft"
    assert preparation in instructions
    assert instructions.index(briefing) < instructions.index(preparation)
    assert "A second Live delegation is not required for a reply to that pending draft" in instructions
    assert "Only the client can authorize paid generation, after actual user confirmation" in instructions
    assert "otherwise it must obtain explicit user confirmation before any paid music generation" in instructions
    assert "Do not treat silence, your own speech or an earlier approval as current confirmation" in instructions
    assert "Handle ordinary self-contained conversation directly" in instructions
    assert "Never generate another track automatically." in instructions
    assert "The client has only bounded app tools, no shell, files, browsing or general agent." in instructions
    assert session["delegation"] == {"type": "client"}
    assert "tools" not in session
    assert session["model"] == "gpt-live-1-codex"
    assert session["audio"] == {"output": {"voice": "cove"}}
