# Subscription voice

The app's `OpenaiRealtimeHandler` supports two explicitly selected billing paths. `subscription` uses ChatGPT Live through the included trusted-host broker. `api_key` uses OpenAI's public Realtime API and an explicitly supplied API key. Neither path falls back to the other.

This implementation has offline protocol and real loopback WebRTC qualification. It doesn't establish live account entitlement, robot startup, microphone capture, speaker playback or motor behavior.

## Trusted host setup

Run the broker on a machine you trust with your Codex ChatGPT sign-in. It doesn't need Hermes. Use the existing application Python environment, which includes aiortc, AV and HTTPX through the declared dependencies. Don't install anything into a shared environment merely to start the broker.

The following commands use Nushell. Set values through your local secret mechanism, not shell history or settings JSON. `DJ_REACHY_CODEX_AUTH_FILE` is an explicit host-side path to an existing Codex file containing `tokens.access_token` and `tokens.account_id`, with `auth_mode` absent or `chatgpt`. The resolver never searches for credentials, accepts an API-key-only file, reads `OPENAI_API_KEY`, or refreshes tokens. Renew an expired sign-in with Codex on the trusted host. Applications can instead inject a `Callable[[], CodexCredential]` resolver into `BrokerService`.

```nu
$env.DJ_REACHY_CODEX_AUTH_FILE = '/explicit/host-only/path/to/codex-auth.json'
$env.DJ_REACHY_SUBSCRIPTION_BROKER_TOKEN = (input --suppress-output 'Scoped broker token, at least 32 random ASCII characters: ')
python -m dj_reachy.subscription_broker --host 127.0.0.1 --port 8766
```

The default listener is loopback. Every session and planning request requires that dedicated bearer token, including loopback requests. Don't reuse a provider credential as the broker token. A remote listener requires TLS directly:

```nu
python -m dj_reachy.subscription_broker --host 0.0.0.0 --port 8766 --tls-cert /host/cert.pem --tls-key /host/key.pem
```

Use a trusted certificate and firewall the listener to approved clients. The token authorizes subscription voice sessions and bounded app planning. It doesn't grant shell, filesystem, arbitrary upstream HTTP, credential retrieval or general agent execution. Rotate it on both sides to revoke clients. The broker accepts at most four concurrent operations. It doesn't follow redirects or use environment HTTP proxies. Bodies are capped at 256 KiB, SDP at 64 KiB, planner history at 128 KiB and SSE responses at 512 KiB.

On the app client, set only the broker URL and scoped broker token. Never copy the host's Codex auth file or OAuth/account credentials to the robot:

```nu
$env.DJ_REACHY_SUBSCRIPTION_BROKER_URL = 'https://trusted-host.example:8766/session'
$env.DJ_REACHY_SUBSCRIPTION_BROKER_TOKEN = (input --suppress-output 'Scoped broker token: ')
```

For a client on the broker host, `http://127.0.0.1:8766/session` is allowed. Non-loopback HTTP, URL credentials, query parameters, fragments and token whitespace are rejected. Environment variables are intentionally separate from persisted voice/music/motion settings.

## One silent headless qualification

Run this only when the account owner has authorized a session. It makes exactly one broker session POST, which makes exactly one consumer session POST. There are no retries, greeting requests, recording devices, speakers, motion commands or planner calls. WebRTC sends a synthetic silence track, receives media into memory and closes immediately after `session.started`.

```nu
python -m dj_reachy.subscription_voice --qualify
```

The equivalent async function is `await dj_reachy.subscription_voice.qualify_subscription()`. Success returns:

```json
{"ok": true, "auth": "subscription", "handshakes": 1, "recording": false, "playback": false}
```

A rejection is final for that attempt. Check the host sign-in and account entitlement before asking for another attempt. This command doesn't start the robot application.

## App configuration and reconnect

Use the existing app launch path and its runtime settings. The subscription voice group is:

```json
{"voice": {"backend": "subscription", "model": "gpt-live-1-codex", "voice": "cove", "language": "auto"}}
```

Public API-key Realtime must be selected explicitly, with its own compatible model and voice:

```json
{"voice": {"backend": "api_key", "model": "gpt-realtime", "voice": "cedar", "language": "auto"}}
```

Supply `OPENAI_API_KEY` through the existing key input/environment only for that explicit backend. The handler uses `voice.model`, not the inherited fixed `MODEL_NAME`. Subscription OAuth never reaches `AsyncOpenAI`. The subscription adapter never consults an API key, even if one exists in the app environment.

The UI/config lane owns persistence and conversational instructions. `get_session_instructions()` reads the app's user-owned `instructions.md`; instruction text and conversation language are independent of music lyrics, duration, styles and instrumental settings. The inherited built-in profile still contains English-only rules in this lane's base. Integration must replace those profile rules or supply user-owned instructions for the selected conversation language. This lane removed English-only startup/response injection from the handler itself.

After changing persona, model, voice or backend, call `await handler.reload_instructions()`. Calls from a UI event loop route to the supervisor's owning loop before touching WebRTC. `apply_personality()` retains its existing UI signature and uses the same reconnect path. Reconnect closes the old media and work, reloads settings/instructions, and negotiates one new session through the existing supervisor. It doesn't create a second receiver. Calling reload before the supervisor runs fails explicitly. Subscription connection loss doesn't automatically start another account session; reconnect is an explicit action. Public API Realtime retains bounded transport retries.

## Delegated execution

The Live session uses consumer client delegation, not Realtime function tools. The broker sends this consumer configuration with the SDP offer:

- Endpoint `https://chatgpt.com/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas`.
- Default model `gpt-live-1-codex`, voice `cove`, `delegation: {"type": "client"}`.
- Host-only `Authorization`, `ChatGPT-Account-Id`, and `OpenAI-Alpha: quicksilver=v2` headers.
- Audio on WebRTC tracks and consumer events on `oai-events`.

Readiness requires `session.started` or `session.updated`, not merely an open data channel or valid SDP. Pending context waits for readiness. Each `delegation.context.append` or `session.context.append` carries at most 500 UTF-8 bytes in `content: [{"type":"input_text","text":"..."}]`. Delegation replies use `delegation_item_id`. Commentary is quiet progress/state; speakable context is a result to tell the user. PCM isn't evidence of a public `response.done` event. `trigger_response_and_wait()` returns false for consumer context delivery instead of claiming speech completion.

The client associates `input_transcript.added`, user `turn.done`, and client `delegation.created` events with the current user turn. Completed delegation IDs and consumed turns aren't executed again. Model commentary and empty repeated delegations can't reauthorize old work. Full anchored stop/pause/resume commands bypass planning, and user `turn.done` can invoke them before a delegation arrives. An earlier pause isn't included in a later resume request.

General requests, including "sing me a ten-second salsa about robots", use the broker's subscription-backed Codex Responses planner at `https://chatgpt.com/backend-api/codex/responses`. `DJ_REACHY_SUBSCRIPTION_PLANNER_MODEL` on the host selects its model, default `gpt-5.3-codex`. There is no API-key planner fallback. Requests use `store: false`, `stream: true`, full per-request history and a `dj_reachy` function namespace. Only completed SSE responses can authorize commands. Tool call IDs and encrypted reasoning/history survive stateless tool rounds.

The fixed registry permits `make_song_and_dance`, `play_saved_song`, `pause_music`, `resume_music`, `stop_music`, `dance` and `stop_dance`. Arguments are validated at the broker and client. Song duration is 3000 to 600000 ms; unspecified duration uses the app music setting rather than a hardcoded 30 seconds. Replay needs an exact known filename. Dance repeat is bounded to four and move names are allowlisted. Each delegation has at most four planner rounds and six tool calls. Unsupported requests need clarification, not shell or file access.

App tools execute locally through the existing `dispatch_tool_call`/`ToolDependencies` route. The receiver never waits for a song to complete. Stop invalidates old planner/generation epochs, cancels local work, invokes the real stop tool and prevents late results from starting playback or speaking as a replacement session. New playback waits for the stop barrier. Assistant muting has an owner ID, so an old playback task can't unmute a newer one. Shutdown owns setup, reader, planner and tool tasks, including failure during `__aenter__`.

Canceling a local coroutine doesn't prove that an upstream paid generation stopped or that credits were refunded. Stop feedback explicitly leaves provider cancellation unconfirmed. The music lane owns decoder/generation cancellation internals; combined integration must exercise that implementation with this handler.

## Offline verification

The voice tests require no credentials and use explicit loopback fixtures. Use an owned sandbox for HOME, DJ_REACHY_HOME, TMPDIR, XDG roots, HF_HOME and caches, strip provider credentials, and set `HF_HUB_OFFLINE=1`, `GRADIO_ANALYTICS_ENABLED=False`, and `PYTHONPATH` to the current worktree's `src`.

```nu
python -m pytest -q tests/test_subscription_voice.py tests/test_subscription_broker.py tests/test_voice_execution.py tests/test_openai_realtime.py
```

Coverage includes actual aiortc negotiation, bidirectional PCM with real AV frames, readiness ordering, malformed SDP/billing cleanup, a real HTTP/SSE planner tool round, credential containment, sequential controls during blocked planner/song work, stale results, explicit reconnect and shutdown during setup. The lane artifact contains a separate executable smoke that runs the actual qualification CLI and then the real app handler through loopback broker/SSE/WebRTC and the registered `pause_music` tool, without mocking dispatch or opening hardware.

## Contract and entitlement caveat

The ChatGPT consumer endpoint and event dialect aren't a stable public third-party API contract. Matching Conduit's implementation doesn't guarantee that an account has ChatGPT Live access. Public Sign in with ChatGPT Responses support excludes audio and isn't proof of consumer Live entitlement. This adapter fails closed on billing mismatches, rejected signaling, incomplete SSE and invalid commands. Protocol changes require another source review and qualification.

Conduit source revisions, attribution and the complete MIT notice ship in [`src/dj_reachy/static/conduit-voice-NOTICE.txt`](../src/dj_reachy/static/conduit-voice-NOTICE.txt). The Codex Responses request shape was checked against [openai/codex revision 922510f4d7570b42838e015787ed53abfa8761a1](https://github.com/openai/codex/tree/922510f4d7570b42838e015787ed53abfa8761a1/codex-rs/codex-api/src), especially `common.rs` and `endpoint/responses.rs`. The upstream Apache 2.0 license is unchanged.
