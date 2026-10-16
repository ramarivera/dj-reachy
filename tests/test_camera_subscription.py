"""Camera requests stay explicit and use typed image input, never base64 text."""

import json
from pathlib import Path

import pytest

from dj_reachy.voice_planner import app_tool_specs, validate_app_call


def test_camera_is_exposed_in_both_dj_registries() -> None:
    """A camera question must be an actual approved command, not persona text."""
    tools = app_tool_specs()[0]["tools"]
    assert any(tool["name"] == "camera" for tool in tools)
    assert validate_app_call("camera", {"question": "What is in front of you?"}) == (
        "camera", {"question": "What is in front of you?"}
    )
    profile = Path(__file__).resolve().parents[1] / "src/dj_reachy/profiles/dj_reachy/tools.txt"
    assert "camera" in profile.read_text().splitlines()


def test_live_frontend_knows_camera_and_background_generation() -> None:
    """The Live frontend must delegate supported vision instead of refusing it."""
    from dj_reachy.subscription_broker import LIVE_INSTRUCTIONS

    text = LIVE_INSTRUCTIONS.lower()
    assert "camera" in text and "delegate" in text
    assert "continue" in text and "generat" in text


def test_subscription_camera_result_is_a_typed_image_not_base64_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inspect the exact Responses payload using a synthetic JPEG and mock SSE."""
    import base64

    import cv2
    import httpx
    import numpy as np

    from dj_reachy.voice_planner import subscription_plan
    from dj_reachy.subscription_broker import CodexCredential

    ok, data = cv2.imencode(".jpg", np.full((16, 16, 3), 90, dtype=np.uint8))
    assert ok
    encoded = base64.b64encode(data).decode()
    history = [
        {"type": "function_call", "call_id": "photo-1", "name": "camera", "arguments": json.dumps({"question": "What do you see?"})},
        {"type": "function_call_output", "call_id": "photo-1", "output": json.dumps({"b64_im": encoded})},
    ]
    observed = []

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(json.loads(request.content))
        result = {"type": "response.completed", "response": {"status": "completed", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "A synthetic test image."}]}
        ]}}
        return httpx.Response(200, content="data: " + json.dumps(result) + "\n\n")

    original_client = httpx.Client

    def fixture_client(*, trust_env: bool, follow_redirects: bool, timeout: int) -> httpx.Client:
        return original_client(transport=httpx.MockTransport(upstream), trust_env=trust_env, follow_redirects=follow_redirects, timeout=timeout)

    monkeypatch.setattr(httpx, "Client", fixture_client)
    result = subscription_plan(
        {"billing": "subscription", "transcript": "What do you see?", "history": history},
        lambda: CodexCredential("fixture-only", "fixture-account"),
        endpoint="https://fixture.invalid/responses", model="fixture-model",
    )
    assert result["text"] == "A synthetic test image."
    inputs = observed[0]["input"]
    images = [part for item in inputs for part in item.get("content", []) if part.get("type") == "input_image"]
    assert images == [{"type": "input_image", "image_url": "data:image/jpeg;base64," + encoded, "detail": "auto"}]
    output = next(item for item in inputs if item.get("type") == "function_call_output")
    assert json.loads(output["output"]) == {"status": "image_captured"}
    assert encoded not in output["output"]
    assert json.loads(history[1]["output"])["b64_im"] == encoded
    assert observed[0]["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_camera_encodes_a_bounded_jpeg_without_persisting_frames() -> None:
    """Even a large synthetic frame fits the existing broker history allowance."""
    import base64
    from unittest.mock import MagicMock

    import cv2
    import numpy as np

    from dj_reachy.tools.camera import Camera
    from dj_reachy.tools.core_tools import ToolDependencies

    worker = MagicMock()
    worker.get_latest_frame.return_value = np.random.default_rng(4).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_worker=worker)
    result = await Camera()(deps, question="Describe this synthetic test image")
    assert 0 < len(result["b64_im"]) <= 80000
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(result["b64_im"]), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 480


@pytest.mark.asyncio
async def test_camera_bytes_are_not_sent_as_live_frontend_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image bytes go to the vision planner only, not spoken/quiet text context."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from dj_reachy.openai_realtime import OpenaiRealtimeHandler
    from dj_reachy.tools.core_tools import ToolDependencies

    app = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    call = {"type": "function_call", "call_id": "cam", "name": "camera", "arguments": json.dumps({"question": "What do you see?"})}
    connection = SimpleNamespace(send_context=AsyncMock(), plan=AsyncMock(side_effect=[
        {"calls": [{"call_id": "cam", "name": "camera", "arguments": {"question": "What do you see?"}}], "output": [call]},
        {"calls": [], "output": [], "text": "A synthetic test image."},
    ]))
    app.connection = connection
    monkeypatch.setattr(app, "_dispatch", AsyncMock(return_value={"b64_im": "fixture-only-image-bytes"}))
    try:
        await app._run_subscription_delegation("vision", "What do you see?")
        assert connection.plan.await_count == 2
        contexts = [c.args[0] for c in connection.send_context.call_args_list]
        assert not any("fixture-only-image-bytes" in text for text in contexts)
        assert contexts[-1] == "A synthetic test image."
        assert "fixture-only-image-bytes" in connection.plan.call_args.args[1][-1]["output"]
    finally:
        app.diagnostics.close()

