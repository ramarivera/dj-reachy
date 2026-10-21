"""Offline notice ordering, not a prediction of Live's spoken feedback behavior.

Exercise the real song tool, handler and subscription context serializer. Only
provider composition, decoder playback and the data-channel sink are fakes;
no SDK object, connection, microphone or model session is started.
"""

import json
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from collections.abc import Iterator

import pytest

from dj_reachy_remix.openai_realtime import OpenaiRealtimeHandler, voice_config
from dj_reachy_remix.tools.core_tools import ToolDependencies
from dj_reachy_remix.profiles.dj_reachy import make_song_and_dance as music
from dj_reachy_remix.subscription_voice import SubscriptionRealtimeAdapter, SubscriptionRealtimeConnection


@pytest.fixture
def notice_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[OpenaiRealtimeHandler, SubscriptionRealtimeConnection, list[dict[str, object]]]]:
    """Use a ready in-memory data channel without opening a peer or any socket."""
    monkeypatch.setattr(voice_config, "VOICE_BACKEND", "subscription")
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    app = OpenaiRealtimeHandler(deps)
    deps.openai_realtime_handler = app
    connection = SubscriptionRealtimeConnection(SubscriptionRealtimeAdapter(diagnostics=app.diagnostics))
    sent: list[dict[str, object]] = []
    connection._channel = SimpleNamespace(readyState="open", send=lambda text: sent.append(json.loads(text)))
    connection.ready.set()
    app.connection = connection
    try:
        yield app, connection, sent
    finally:
        app.diagnostics.close()


async def wait_for_notice(sent: list[dict[str, object]]) -> None:
    """Advance only the local scheduler; fail if the real notice was never sent."""
    for _ in range(50):
        if sent:
            return
        await asyncio.sleep(0)
    raise AssertionError("The generation-ready context never reached the in-memory channel")


async def deliver(connection: SubscriptionRealtimeConnection, app: OpenaiRealtimeHandler, event: dict[str, object]) -> None:
    """Carry the synthetic provider event through the real consumer iterator."""
    connection._enqueue(SimpleNamespace(**event))
    await app._handle_event(await connection.__anext__())


@pytest.mark.asyncio
async def test_generation_ready_precedes_playback_but_completed_result_follows_it(
    notice_actor: tuple[OpenaiRealtimeHandler, SubscriptionRealtimeConnection, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate the actual ready notice, speech drain and final tool result."""
    app, connection, sent = notice_actor
    order: list[str] = []
    drain_started, release_drain = asyncio.Event(), asyncio.Event()
    playing, finish_playback = asyncio.Event(), asyncio.Event()

    async def compose(*_args: object, **_kwargs: object) -> str:
        order.append("generated")
        return "offline-fixture.mp3"

    async def drain() -> bool:
        order.append("notice_drain")
        drain_started.set()
        await release_drain.wait()
        return True

    async def play(*_args: object, **_kwargs: object) -> bool:
        order.append("playing")
        playing.set()
        await finish_playback.wait()
        order.append("playback_completed")
        return False

    monkeypatch.setattr(music, "_compose_music", compose)
    monkeypatch.setattr(music, "_play_mp3", play)
    app._drain_queue = drain
    task = asyncio.create_task(music.MakeSongAndDance()(app.deps, prompt="Offline fixture", music_length_ms=3000))
    try:
        await wait_for_notice(sent)
        assert order == ["generated"] and not task.done()
        assert len(sent) == 1
        assert sent[0]["type"] == "session.context.append"
        assert sent[0]["channel"] == "speakable"
        assert "delegation_item_id" not in sent[0]
        content = sent[0]["content"]
        assert isinstance(content, list) and len(content) == 1
        assert content[0]["type"] == "input_text"
        text = content[0]["text"]
        assert isinstance(text, str) and text.strip()
        # Assert the executable phase order without pinning the policy wording.
        assert not playing.is_set() and not app._mute_output_audio

        await deliver(connection, app, {"type": "turn.done", "turn": {"role": "user", "transcript": "Thanks"}})
        assert app._music_ready_notice is not None and not app._music_ready_notice.is_set()
        await deliver(connection, app, {
            "type": "turn.done", "turn": {"role": "assistant", "transcript": "Your song is ready."},
        })
        await asyncio.wait_for(drain_started.wait(), 1)
        assert not playing.is_set() and not task.done()
        release_drain.set()
        await asyncio.wait_for(playing.wait(), 1)
        assert not task.done(), "A completed tool result must not precede playback completion"
        finish_playback.set()
        result = await asyncio.wait_for(task, 1)
        assert result == {"status": "completed", "song_path": "offline-fixture.mp3"}
        assert order == ["generated", "notice_drain", "playing", "playback_completed"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("unrelated", [
    pytest.param(
        {"type": "turn.done", "turn": {"id": "ordinary-conversation", "role": "assistant", "transcript": "Paris is the capital of France."}},
        id="unrelated-assistant-turn",
        # Known open bug, ramarivera/dj-reachy-remix#7: no provider contract links the
        # ready append to a particular assistant utterance yet.
        marks=pytest.mark.xfail(strict=True, reason="ramarivera/dj-reachy-remix#7: ready notice lacks a causal completion"),
    ),
    pytest.param(
        {"type": "response.done", "response": {"id": "earlier-unrelated-response", "status": "completed"}},
        id="unrelated-response-done",
    ),
])
async def test_unrelated_completion_does_not_release_generation_ready_notice(
    notice_actor: tuple[OpenaiRealtimeHandler, SubscriptionRealtimeConnection, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
    unrelated: dict[str, object],
) -> None:
    """An unrelated completion is not acknowledgment of the ready announcement.

    The assistant turn case is the consumer Live dialect. The response.done case
    probes the handler's additional accepted completion kind; it does not assert
    that subscription Live actually emits that public-Realtime event.
    """
    app, connection, sent = notice_actor
    compose = AsyncMock(return_value="offline-fixture.mp3")
    play = AsyncMock(return_value=False)
    drain = AsyncMock(return_value=True)
    monkeypatch.setattr(music, "_compose_music", compose)
    monkeypatch.setattr(music, "_play_mp3", play)
    app._drain_queue = drain
    task = asyncio.create_task(music.MakeSongAndDance()(app.deps, prompt="Offline fixture", music_length_ms=3000))
    try:
        await wait_for_notice(sent)
        compose.assert_awaited_once()
        play.assert_not_awaited()
        assert not task.done()
        await deliver(connection, app, unrelated)
        assert app._music_ready_notice is not None
        assert not app._music_ready_notice.is_set(), (
            "Unrelated completion released the song-ready speech gate; "
            "the announcement has no correlated completion contract"
        )
        drain.assert_not_awaited()
        play.assert_not_awaited()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
