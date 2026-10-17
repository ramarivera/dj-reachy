"""Offline confirmation RPC contract; all credentials and transports are synthetic."""

from __future__ import annotations
import json
import socket
from typing import Any

import httpx
import pytest

from dj_reachy_remix import subscription_voice
from dj_reachy_remix.voice_planner import validate_confirmation_input
from dj_reachy_remix.subscription_broker import BrokerError, BrokerService, CodexCredential


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make accidental socket connections or DNS resolution a test failure."""
    def denied(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Real network forbidden in confirmation planner tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def pending_request() -> dict[str, Any]:
    """Return the frozen, expanded execution request rather than model app arguments."""
    return {
        "draft_id": "local-draft-1",
        "request": {
            "prompt": "Una canción tranquila", "music_length_ms": 30000,
            "model_id": "music_v2_5", "output_format": "mp3_48000_192",
            "positive_styles": ["jazz"], "negative_styles": [], "lyrics": "",
            "language": "es", "force_instrumental": False,
        },
        "preview": "Una canción de jazz de treinta segundos. ¿La genero?",
        "original_user_text": "Hazme una canción tranquila de jazz",
        "assistant_context": ["¿Quieres que genere esa canción?"],
    }


def decision_item(decision: str = "approve", arguments: Any = None, text: str = "De acuerdo.") -> dict[str, Any]:
    """Produce one complete metadata function, never an app execution call."""
    return {
        "type": "function_call", "call_id": "provider-call-1", "namespace": "dj_reachy_remix",
        "name": "music_confirmation_decision",
        "arguments": json.dumps({"decision": decision, "arguments": arguments, "text": text}),
    }


def install_provider(monkeypatch: pytest.MonkeyPatch, output: list[dict[str, Any]]) -> list[httpx.Request]:
    """Intercept the real synchronous SSE boundary with MockTransport."""
    observed: list[httpx.Request] = []
    original = httpx.Client

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        event = {"type": "response.completed", "response": {"status": "completed", "output": output}}
        return httpx.Response(200, content="data: " + json.dumps(event) + "\n\n")

    def client(**kwargs: Any) -> httpx.Client:
        assert kwargs == {"trust_env": False, "follow_redirects": False, "timeout": 30}
        return original(transport=httpx.MockTransport(upstream), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    return observed


def test_confirmation_policy_never_approves_unapplied_setting_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose honest clarification for controls outside the revision schema."""
    observed = install_provider(monkeypatch, [decision_item("none", text="Change instrumental mode in Music settings first.")])
    service = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account"))
    result = service.plan({"billing": "subscription", "transcript": "Go ahead, but use a different model and turn instrumental mode off", "history": [], "pending_confirmation": pending_request()})
    policy = json.loads(observed[0].content)["instructions"]
    assert "Do not approve a frozen request that conflicts with the user's requested changes" in policy
    assert "Other option changes require Music settings" in policy
    assert "For unrelated conversation return none with empty text" in policy
    assert result["calls"] == [] and result["confirmation"]["decision"] == "none"


def test_normal_music_planning_prepares_unbilled_draft_before_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Describe preparation truthfully at the same real SSE serialization boundary."""
    item = {
        "type": "function_call", "call_id": "fixture-preparation", "namespace": "dj_reachy_remix",
        "name": "make_song_and_dance", "arguments": json.dumps({"prompt": "Instrumental road-trip rock", "music_length_ms": 10000}),
    }
    observed = install_provider(monkeypatch, [item])
    service = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account"))
    result = service.plan({"billing": "subscription", "transcript": "Make a ten-second instrumental road-trip rock track", "history": []})
    payload = json.loads(observed[0].content)
    assert "Music calls prepare unbilled drafts" in payload["instructions"]
    assert "Do not ask for approval before preparation" in payload["instructions"]
    assert "Returning a music tool call never starts generation" in payload["instructions"]
    song = next(tool for tool in payload["tools"][0]["tools"] if tool["name"] == "make_song_and_dance")
    assert "unbilled" in song["description"] and "draft" in song["description"]
    assert result["calls"][0]["arguments"] == {"prompt": "Instrumental road-trip rock", "music_length_ms": 10000}
    assert len(observed) == 1


@pytest.mark.asyncio
async def test_actual_adapter_to_host_sse_confirmation_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the latest real reply, frozen data and OAuth routing across both JSON hops."""
    observed = install_provider(monkeypatch, [decision_item()])
    service = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account"))
    original_async = httpx.AsyncClient
    broker_bodies: list[dict[str, Any]] = []

    def broker(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://broker.invalid/plan"
        assert request.headers["Authorization"] == "Bearer fixture-scoped-token"
        body = json.loads(request.content)
        broker_bodies.append(body)
        return httpx.Response(200, json=service.plan(body))

    def async_client(**kwargs: Any) -> httpx.AsyncClient:
        return original_async(transport=httpx.MockTransport(broker), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    monkeypatch.setenv(subscription_voice.BROKER_URL_ENV, "https://broker.invalid/session")
    monkeypatch.setenv(subscription_voice.BROKER_TOKEN_ENV, "fixture-scoped-token")
    pending = pending_request()
    reply = "  Sí, adelante con esa canción.  "
    connection = subscription_voice.SubscriptionRealtimeAdapter().connect()
    try:
        result = await connection.plan_confirmation(reply, pending)
    finally:
        await connection.close()
    assert broker_bodies == [{"billing": "subscription", "transcript": reply, "history": [], "pending_confirmation": pending}]
    assert result == {
        "ok": True, "auth": "subscription", "calls": [], "output": [], "text": "De acuerdo.",
        "confirmation": {"decision": "approve", "arguments": None},
    }
    assert len(observed) == 1
    request = observed[0]
    assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
    assert request.headers["Authorization"] == "Bearer fixture-oauth"
    assert request.headers["ChatGPT-Account-Id"] == "fixture-account"
    assert request.headers["Accept"] == "text/event-stream"
    assert request.headers["OpenAI-Beta"] == "responses=experimental"
    payload = json.loads(request.content)
    assert payload["model"] == "gpt-6.1-sol"
    assert payload["stream"] is True and payload["store"] is False
    assert payload["parallel_tool_calls"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["input"][-1] == {"type": "message", "role": "user", "content": [{"type": "input_text", "text": reply}]}
    assert json.loads(payload["input"][0]["content"].split("\n", 1)[1]) == pending
    tools = payload["tools"]
    assert tools[0]["name"] == "dj_reachy_remix"
    assert [tool["name"] for tool in tools[0]["tools"]] == ["music_confirmation_decision"]
    parameters = tools[0]["tools"][0]["parameters"]
    assert set(parameters["properties"]) == set(parameters["required"]) == {"decision", "arguments", "text"}
    assert parameters["additionalProperties"] is False
    instructions = payload["instructions"].lower()
    for policy in ("multilingual", "latest actual user reply", "not approval", "yes but", "refusal", "ambiguous", "general chat", "frozen", "revise", "decline", "none"):
        assert policy in instructions
    assert "input_image" not in json.dumps(payload)


@pytest.mark.parametrize("output", [
    [], [decision_item(), decision_item()],
    [{**decision_item(), "name": "make_song_and_dance"}],
    [{**decision_item(), "namespace": "other"}],
    [{**decision_item(), "call_id": ""}],
    [{**decision_item(), "call_id": " "}],
    [{**decision_item(), "call_id": "x" * 201}],
    [{**decision_item(), "call_id": "non-ascii-é"}],
    [{**decision_item(), "arguments": '{"decision":"none","decision":"approve","arguments":null,"text":""}'}],
    [{**decision_item(), "arguments": '{"decision":"approve","arguments":null,"text":"","extra":true}'}],
    [{**decision_item(), "arguments": "{}"}],
    [{**decision_item(), "arguments": "not JSON"}],
    [{**decision_item(), "arguments": {"decision": "approve", "arguments": None, "text": ""}}],
    [{**decision_item(), "arguments": " " * 524289 + decision_item()["arguments"]}],
    [{**decision_item(), "arguments": "[" * 50000 + "]" * 50000}],
    [{**decision_item(), "status": "in_progress"}],
    [decision_item("revise", {"prompt": "x"})],
    [decision_item("unknown")], [decision_item("approve", {"prompt": "changed"})],
    [decision_item("decline", {})], [decision_item("none", {})],
    [decision_item("revise", {"prompt": "x", "music_length_ms": True})],
    [decision_item("revise", {"prompt": "x", "music_length_ms": 2999})],
    [decision_item("revise", {"prompt": "x", "music_length_ms": 600001})],
    [decision_item("revise", {"prompt": "x", "music_length_ms": 3000, "model_id": "music_v1"})],
    [decision_item("revise", {"prompt": "x" * 2001, "music_length_ms": 3000})],
    [decision_item("revise", {"prompt": "x" + " " * 2000, "music_length_ms": 3000})],
    [decision_item("revise", {"prompt": "\ud800", "music_length_ms": 3000})],
    [decision_item(text="x" * 2001)], [decision_item(text="\ud800")],
    [decision_item(), {"type": "computer_call"}],
    [decision_item(), {"type": "image_generation_call"}],
])
def test_malformed_confirmation_output_fails_closed(monkeypatch: pytest.MonkeyPatch, output: list[dict[str, Any]]) -> None:
    """No malformed metadata or unexpected executable output can become consent."""
    install_provider(monkeypatch, output)
    service = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account"))
    with pytest.raises(BrokerError) as error:
        service.plan({"billing": "subscription", "transcript": "proceed", "history": [], "pending_confirmation": pending_request()})
    assert error.value.status == 502


@pytest.mark.parametrize("excess", [0, 1])
def test_full_request_json_byte_boundary(excess: int) -> None:
    """The frozen execution JSON is bounded independently of preview/context text."""
    pending = pending_request()
    request = pending["request"]
    request.update({"positive_styles": ["é" * 200] * 50, "lyrics": "\n".join(["é" * 199] * 30), "prompt": "x"})
    base = len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    request["prompt"] = "x" * (1 + 32768 - base + excess)
    assert len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) == 32768 + excess
    if excess:
        with pytest.raises(BrokerError) as error:
            validate_confirmation_input("yes", pending)
        assert error.value.status == 400
    else:
        assert validate_confirmation_input("yes", pending) == pending


def test_total_pending_json_is_bounded() -> None:
    """Individually bounded fields cannot bypass the aggregate confirmation budget."""
    pending = pending_request()
    request = pending["request"]
    request.update({"positive_styles": ["é" * 200] * 50, "lyrics": "\n".join(["é" * 199] * 30), "prompt": "x"})
    base = len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    request["prompt"] = "x" * (1 + 32768 - base)
    pending.update({"preview": "\x00" * 8192, "original_user_text": "u" * 12000, "assistant_context": ["c" * 8192]})
    assert len(json.dumps(pending, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > 65536
    with pytest.raises(BrokerError) as error:
        validate_confirmation_input("yes", pending)
    assert error.value.status == 400


@pytest.mark.parametrize(("decision", "arguments"), [
    ("approve", None), ("decline", None), ("none", None),
    ("revise", {"prompt": "Más rápido", "music_length_ms": 3000}),
    ("revise", {"prompt": "Más lento", "music_length_ms": None}),
])
def test_valid_decisions_return_only_bounded_metadata(monkeypatch: pytest.MonkeyPatch, decision: str, arguments: Any) -> None:
    """Revision is validated proposal data, not an executable app-tool call."""
    install_provider(monkeypatch, [decision_item(decision, arguments, "é" * 2000)])
    result = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account")).plan({
        "billing": "subscription", "transcript": "yes but make it faster", "history": [], "pending_confirmation": pending_request(),
    })
    expected = {"prompt": arguments["prompt"]} if arguments is not None and arguments.get("music_length_ms") is None else arguments
    assert result["confirmation"] == {"decision": decision, "arguments": expected}
    assert result["calls"] == [] and result["output"] == [] and result["text"] == "é" * 2000


@pytest.mark.parametrize(("field", "value"), [
    ("draft_id", ""), ("draft_id", "a b"), ("draft_id", "é"), ("draft_id", "x" * 201),
    ("preview", "é" * 4097), ("preview", "\ud800"),
    ("original_user_text", "é" * 6001), ("original_user_text", "\udfff"),
    ("assistant_context", [""] * 5), ("assistant_context", ["é" * 2049, "é" * 2048]),
    ("assistant_context", ["\ud800"]), ("assistant_context", "not a list"),
    ("request", {"prompt": "not expanded", "music_length_ms": 3000}),
    ("request", None),
])
def test_invalid_pending_rejected_before_resolver(field: str, value: Any) -> None:
    """Malformed input never reaches credentials, providers or app execution."""
    pending = pending_request()
    pending[field] = value

    def forbidden() -> CodexCredential:
        pytest.fail("resolver called for invalid pending confirmation")

    with pytest.raises(BrokerError) as error:
        BrokerService(forbidden).plan({"billing": "subscription", "transcript": "yes", "history": [], "pending_confirmation": pending})
    assert error.value.status == 400


@pytest.mark.parametrize("mutation", ["extra", "missing", "secret", "image", "bad_duration", "nan", "surrogate", "tuple", "integer_key", "oversize"])
def test_invalid_request_json_is_not_normalized_into_validity(mutation: str) -> None:
    """Reject non-JSON trees, unknown fields and full-request bounds before resolver."""
    pending = pending_request()
    request = pending["request"]
    if mutation == "extra":
        pending["provider_id"] = "do-not-repair"
    elif mutation == "missing":
        del pending["preview"]
    elif mutation == "secret":
        request["api_key"] = "fixture-not-a-real-secret"
    elif mutation == "image":
        request["input_image"] = "data:image/jpeg;base64,fixture"
    elif mutation == "bad_duration":
        request["music_length_ms"] = True
    elif mutation == "nan":
        request["composition_plan"] = {"chunks": [{"duration_ms": float("nan")}]}
        del request["prompt"]
    elif mutation == "surrogate":
        request["prompt"] = "\ud800"
    elif mutation == "tuple":
        request["composition_plan"] = {"chunks": [{"arbitrary_data": (1, 2)}]}
        del request["prompt"]
    elif mutation == "integer_key":
        request["composition_plan"] = {"chunks": [{1: "not JSON"}]}
        del request["prompt"]
    else:
        request["lyrics"] = "é" * 6000
        request["positive_styles"] = ["é" * 200] * 50
    with pytest.raises(BrokerError) as error:
        validate_confirmation_input("yes", pending)
    assert error.value.status == 400


@pytest.mark.parametrize("transcript", ["", " ", "é" * 8001, "\ud800", None])
def test_invalid_current_reply_rejected_before_resolver(transcript: Any) -> None:
    """Historical data cannot substitute for a genuine bounded current reply."""
    with pytest.raises(BrokerError) as error:
        BrokerService(lambda: pytest.fail("resolver called")).plan({
            "billing": "subscription", "transcript": transcript, "history": [], "pending_confirmation": pending_request(),
        })
    assert error.value.status == 400


def test_confirmation_never_accepts_app_or_camera_history() -> None:
    """Even valid normal planner history is disallowed in metadata mode."""
    with pytest.raises(BrokerError) as error:
        BrokerService(lambda: pytest.fail("resolver called")).plan({
            "billing": "subscription", "transcript": "yes", "history": [{"type": "message", "role": "assistant", "content": "yes"}],
            "pending_confirmation": pending_request(),
        })
    assert error.value.status == 400


def test_unicode_decision_bounds_allow_escaped_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid Unicode scalar limits must not depend on the provider's JSON escaping."""
    text = "🎵" * 2000
    arguments = {"prompt": text, "music_length_ms": 3000}
    install_provider(monkeypatch, [decision_item("revise", arguments, text)])
    result = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account")).plan({
        "billing": "subscription", "transcript": "Make it musical", "history": [], "pending_confirmation": pending_request(),
    })
    assert result["confirmation"] == {"decision": "revise", "arguments": arguments}
    assert result["text"] == text


def test_domain_pending_payload_matches_planner_contract() -> None:
    """The real pure draft producer exports expanded options accepted by the host."""
    from dj_reachy_remix.runtime_settings import DEFAULT_SETTINGS
    from dj_reachy_remix.music_confirmation import PendingMusicConfirmation

    manager = PendingMusicConfirmation(clock=lambda: 1.0)
    draft = manager.prepare({"prompt": "Calm jazz"}, json.loads(json.dumps(DEFAULT_SETTINGS)),
                            original_turn=1, original_user_text="Make some calm jazz")
    pending = draft.pending_payload(assistant_context=["Shall I generate this track?"])
    assert pending["request"] == draft.arguments
    assert pending["request"]["music_length_ms"] == 30000
    assert pending["request"]["model_id"] == "music_v2_5"
    assert validate_confirmation_input("go ahead", pending) == pending


def test_exact_utf8_boundaries_snapshot_without_mutation() -> None:
    """Full boundary data is preserved, and caller mutations cannot change the snapshot."""
    pending = pending_request()
    pending.update({"draft_id": "x" * 200, "preview": "é" * 4096, "original_user_text": "é" * 6000,
                    "assistant_context": ["é" * 1024] * 4})
    snapshot = validate_confirmation_input("é" * 8000, pending)
    assert snapshot == pending
    pending["request"]["positive_styles"].append("mutation")
    assert snapshot["request"]["positive_styles"] == ["jazz"]


@pytest.mark.asyncio
async def test_adapter_rejects_invalid_pending_before_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid local confirmation data must not consult broker settings or send HTTP."""
    async def forbidden(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Broker must not be reached")

    monkeypatch.setattr(subscription_voice, "broker_request", forbidden)
    connection = subscription_voice.SubscriptionRealtimeAdapter().connect()
    try:
        with pytest.raises(BrokerError):
            await connection.plan_confirmation("yes", {"draft_id": "incomplete"})
    finally:
        await connection.close()


@pytest.mark.parametrize("transcript", ["current words", " " , "é" * 8000])
def test_normal_planning_registry_and_history_are_unchanged(monkeypatch: pytest.MonkeyPatch, transcript: str) -> None:
    """Normal planning keeps the existing app tools and correlated historical output."""
    observed = install_provider(monkeypatch, [{
        "type": "function_call", "call_id": "new-call", "namespace": "dj_reachy_remix", "name": "pause_music", "arguments": "{}",
    }])
    history = [
        {"type": "function_call", "call_id": "old-call", "name": "stop_dance", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "old-call", "output": '{"status":"stopped"}'},
    ]
    result = BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account")).plan({
        "billing": "subscription", "transcript": transcript, "history": history,
    })
    assert result["calls"] == [{"call_id": "new-call", "name": "pause_music", "arguments": {}}]
    assert "confirmation" not in result
    payload = json.loads(observed[0].content)
    assert payload["input"][0]["content"][0]["text"] == transcript
    assert payload["input"][1:] == history
    assert payload["tool_choice"] == "auto"
    names = {spec["name"] for spec in payload["tools"][0]["tools"]}
    assert {"camera", "make_song_and_dance", "dance", "pause_music"} <= names
    assert "music_confirmation_decision" not in names


@pytest.mark.parametrize("history", [
    [{"type": "reasoning"}] * 65,
    [{"type": "message", "role": "assistant", "content": "x" * 131073}],
    [{"type": "function_call_output", "call_id": "missing-call", "output": "{}"}],
    [{"type": "function_call", "call_id": "pending-call", "name": "pause_music", "arguments": "{}"}],
])
def test_normal_history_bounds_precede_resolver(history: list[dict[str, Any]]) -> None:
    """Existing normal history size/count/correlation limits are still enforced."""
    with pytest.raises(BrokerError) as error:
        BrokerService(lambda: pytest.fail("resolver called")).plan({
            "billing": "subscription", "transcript": "current words", "history": history,
        })
    assert error.value.status == 400


@pytest.mark.parametrize("encoded", ["x" * 80001, "not-base64!", "ZmFrZS1ub3QtanBlZw=="])
def test_normal_camera_bounds_precede_resolver(encoded: str) -> None:
    """Metadata support does not relax normal camera size/base64/JPEG bounds."""
    history = [
        {"type": "function_call", "call_id": "photo-1", "name": "camera", "arguments": '{"question":"What is here?"}'},
        {"type": "function_call_output", "call_id": "photo-1", "output": json.dumps({"b64_im": encoded})},
    ]
    with pytest.raises(BrokerError) as error:
        BrokerService(lambda: pytest.fail("resolver called")).plan({
            "billing": "subscription", "transcript": "What is here?", "history": history,
        })
    assert error.value.status == 400


@pytest.mark.parametrize("raw", [
    "data: {}\n\n", 'data: {"type":"response.failed"}\n\n',
    'data: {"type":"response.incomplete"}\n\n', 'data: {"type":"error"}\n\n',
    "data: not-json\n\n", "x" * 524289,
])
def test_confirmation_reuses_hard_sse_fail_closed_limits(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """No partial, malformed, failed or oversized SSE stream can become a decision."""
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=raw)), **kwargs,
    ))
    with pytest.raises(BrokerError) as error:
        BrokerService(lambda: CodexCredential("fixture-oauth", "fixture-account")).plan({
            "billing": "subscription", "transcript": "yes", "history": [], "pending_confirmation": pending_request(),
        })
    assert error.value.status == 502
