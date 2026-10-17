# Subscription voice

The app's `OpenaiRealtimeHandler` supports two explicitly selected billing paths. `subscription` uses ChatGPT Live through the included trusted-host broker. `api_key` uses OpenAI's public Realtime API and an explicitly supplied API key. Neither path falls back to the other.

This implementation has offline protocol and real loopback WebRTC qualification. It doesn't establish live account entitlement, robot startup, microphone capture, speaker playback or motor behavior.

## Trusted host setup

Run the broker on a machine you trust with your Codex ChatGPT sign-in. It doesn't need Hermes. Use the existing application Python environment, which includes aiortc, AV and HTTPX through the declared dependencies. Don't install anything into a shared environment merely to start the broker.

The following commands use Nushell. Set values through your local secret mechanism, not shell history or settings JSON. `DJ_REACHY_REMIX_CODEX_AUTH_FILE` is an explicit host-side path to an existing Codex file containing `tokens.access_token` and `tokens.account_id`, with `auth_mode` absent or `chatgpt`. The resolver never searches for credentials, accepts an API-key-only file, reads `OPENAI_API_KEY`, or refreshes tokens. Renew an expired sign-in with Codex on the trusted host. Applications can instead inject a `Callable[[], CodexCredential]` resolver into `BrokerService`.

```nu
$env.DJ_REACHY_REMIX_CODEX_AUTH_FILE = '/explicit/host-only/path/to/codex-auth.json'
$env.DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_TOKEN = (input --suppress-output 'Scoped broker token, at least 32 random ASCII characters: ')
python -m dj_reachy_remix.subscription_broker --host 127.0.0.1 --port 8766
```

The default listener is loopback. Every session and planning request requires that dedicated bearer token, including loopback requests. Don't reuse a provider credential as the broker token. A remote listener requires TLS directly:

```nu
python -m dj_reachy_remix.subscription_broker --host 0.0.0.0 --port 8766 --tls-cert /host/cert.pem --tls-key /host/key.pem
```

Use a trusted certificate and firewall the listener to approved clients. The token authorizes subscription voice sessions and bounded app planning. It doesn't grant shell, filesystem, arbitrary upstream HTTP, credential retrieval or general agent execution. Rotate it on both sides to revoke clients. The broker accepts at most four concurrent operations. It doesn't follow redirects or use environment HTTP proxies. Bodies are capped at 256 KiB, SDP at 64 KiB, planner history at 128 KiB and SSE responses at 512 KiB.

On the app client, set only the broker URL and scoped broker token. Never copy the host's Codex auth file or OAuth/account credentials to the robot:

```nu
$env.DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_URL = 'https://trusted-host.example:8766/session'
$env.DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_TOKEN = (input --suppress-output 'Scoped broker token: ')
```

For a client on the broker host, `http://127.0.0.1:8766/session` is allowed. Non-loopback HTTP, URL credentials, query parameters, fragments and token whitespace are rejected. Environment variables are intentionally separate from persisted voice/music/motion settings.

## One silent headless qualification

Run this only when the account owner has authorized a session. It makes exactly one broker session POST, which makes exactly one consumer session POST. There are no retries, greeting requests, recording devices, speakers, motion commands or planner calls. WebRTC sends a synthetic silence track, receives media into memory and closes immediately after `session.started`.

```nu
python -m dj_reachy_remix.subscription_voice --qualify
```

The equivalent async function is `await dj_reachy_remix.subscription_voice.qualify_subscription()`. Success returns:

```json
{"ok": true, "auth": "subscription", "handshakes": 1, "recording": false, "playback": false}
```

A rejection is final for that attempt. Check the host sign-in and account entitlement before asking for another attempt. This command doesn't start the robot application.

## Required voice model

The requested product is GPT-Live-1.1 with ChatGPT subscription authentication, not an API-key Realtime substitute. OpenAI's [GPT-Live-1.1 system card](https://deploymentsafety.openai.com/gpt-live-1-1) identifies it as the default for Plus and higher tiers, including Codex.

The app requests `gpt-live-1-codex`. This is the default frameless selector in [current official Codex source](https://github.com/openai/codex/blob/aff92080f2b8d5f63a63b4ada7b92054287f237c/codex-rs/core/src/realtime_conversation.rs#L112-L113), not an invented `gpt-live-1.1-codex` selector. Official [ChatGPT-auth WebRTC tests](https://github.com/openai/codex/blob/aff92080f2b8d5f63a63b4ada7b92054287f237c/codex-rs/core/tests/suite/realtime_conversation.rs#L805-L992) also use the same subscription endpoint, `quicksilver=v2` header and client delegation shape as this adapter. The requested alias or a broker echo of it is not server attestation of the resolved model generation.

OpenAI's [public Live documentation](https://developers.openai.com/api/docs/guides/live) and its consumer Codex implementation have different authentication and event contracts. Do not replace consumer event names with public API names, or switch to project API-key billing, just because both support WebRTC.

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

`get_session_instructions()` reads the app's user-owned `instructions.md`. Instruction text and conversation language are independent of music lyrics, duration, styles and instrumental settings. The integrated DJ profile uses the configured conversation language and requires confirmation before a new paid generation; the handler does not inject English-only startup or response instructions.

After changing persona, model, voice or backend, call `await handler.reload_instructions()`. Calls from a UI event loop route to the supervisor's owning loop before touching WebRTC. `apply_personality()` retains its existing UI signature and uses the same reconnect path. Reconnect closes the old media and work, reloads settings/instructions, and negotiates one new session through the existing supervisor. It doesn't create a second receiver. Calling reload before the supervisor runs fails explicitly. Subscription connection loss doesn't automatically start another account session; reconnect is an explicit action. Public API Realtime retains bounded transport retries.

## Delegated execution

The Live session uses consumer client delegation, not Realtime function tools. The broker sends this consumer configuration with the SDP offer:

- Endpoint `https://chatgpt.com/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas`.
- Default model `gpt-live-1-codex`, voice `cove`, `delegation: {"type": "client"}`.
- Host-only `Authorization`, `ChatGPT-Account-Id`, and `OpenAI-Alpha: quicksilver=v2` headers.
- Audio on WebRTC tracks and consumer events on `oai-events`.

Readiness requires `session.started` or `session.updated`, not merely an open data channel or valid SDP. Pending context waits for readiness. Each `delegation.context.append` or `session.context.append` carries at most 500 UTF-8 bytes in `content: [{"type":"input_text","text":"..."}]`. Delegation replies use `delegation_item_id`. Commentary is quiet progress/state; speakable context is a result to tell the user. PCM isn't evidence of a public `response.done` event. `trigger_response_and_wait()` returns false for consumer context delivery instead of claiming speech completion.

The client associates `input_transcript.added`, user `turn.done`, and client `delegation.created` events with the current request. A canonical client delegation may carry bounded `content: [{"type":"input_text","text":"..."}]` without a separate transcript event. The handler accepts that protocol request and labels it as model-authored delegation content, not verbatim user speech. Actual fresh user speech takes precedence when present. Empty notices reserve neither an ID nor a user turn, so a populated same-ID notice can still run; accepted IDs are deduplicated even while work is in flight. Distinct content-only requests have separate correlations. Assistant output/commentary is never a delegation request or evidence of user approval. App permissions, credit confirmations and argument validation still apply.

Full anchored stop/pause/resume requests bypass planning, and user `turn.done` can invoke them before a delegation arrives. An earlier pause isn't included in a later resume request. Cancellation invalidates work before a delayed result can restart playback or speak through a replacement connection.


General requests, including "sing me a ten-second salsa about robots", use the broker's subscription-backed Codex Responses planner at `https://chatgpt.com/backend-api/codex/responses`. `DJ_REACHY_REMIX_SUBSCRIPTION_PLANNER_MODEL` on the host selects its model, default `gpt-6.1-sol`. There is no API-key planner fallback. Requests use `store: false`, `stream: true`, full per-request history and a `dj_reachy_remix` function namespace. Only completed SSE responses can authorize commands. Tool call IDs and encrypted reasoning/history survive stateless tool rounds.

The fixed registry permits `make_song_and_dance`, `play_saved_song`, `pause_music`, `resume_music`, `stop_music`, `dance` and `stop_dance`. Arguments are validated at the broker and client. Song duration is 3000 to 600000 ms; unspecified duration uses the app music setting rather than a hardcoded 30 seconds. Replay needs an exact known filename. Dance repeat is bounded to four and move names are allowlisted. Each delegation has at most four planner rounds and six tool calls. Unsupported requests need clarification, not shell or file access.

App tools execute locally through the existing `dispatch_tool_call`/`ToolDependencies` route. The receiver never waits for a song to complete. Stop invalidates old planner/generation epochs, cancels local work, invokes the real stop tool and prevents late results from starting playback or speaking as a replacement session. New playback waits for the stop barrier. Assistant muting has an owner ID, so an old playback task can't unmute a newer one. Shutdown owns setup, reader, planner and tool tasks, including failure during `__aenter__`.

Canceling a local coroutine doesn't prove that an upstream paid generation stopped or that credits were refunded. Stop feedback explicitly leaves provider cancellation unconfirmed. The music lane owns decoder/generation cancellation internals; combined integration must exercise that implementation with this handler.

## Offline verification

The voice tests require no credentials and use explicit loopback fixtures. Use an owned sandbox for HOME, DJ_REACHY_REMIX_HOME, TMPDIR, XDG roots, HF_HOME and caches, strip provider credentials, and set `HF_HUB_OFFLINE=1`, `GRADIO_ANALYTICS_ENABLED=False`, and `PYTHONPATH` to the current worktree's `src`.

```nu
python -m pytest -q tests/test_subscription_voice.py tests/test_subscription_broker.py tests/test_voice_execution.py tests/test_openai_realtime.py
```

Coverage includes actual aiortc negotiation, bidirectional PCM with real AV frames, readiness ordering, malformed SDP/billing cleanup, a real HTTP/SSE planner tool round, credential containment, sequential controls during blocked planner/song work, stale results, explicit reconnect and shutdown during setup. The lane artifact contains a separate executable smoke that runs the actual qualification CLI and then the real app handler through loopback broker/SSE/WebRTC and the registered `pause_music` tool, without mocking dispatch or opening hardware.

## Contract and entitlement caveat

The ChatGPT consumer endpoint and event dialect aren't a stable public third-party API contract. Matching Conduit's implementation doesn't guarantee that an account has ChatGPT Live access. Public Sign in with ChatGPT Responses support excludes audio and isn't proof of consumer Live entitlement. This adapter fails closed on billing mismatches, rejected signaling, incomplete SSE and invalid commands. Protocol changes require another source review and qualification.

Conduit source revisions, attribution and the complete MIT notice ship in [`src/dj_reachy_remix/static/conduit-voice-NOTICE.txt`](../src/dj_reachy_remix/static/conduit-voice-NOTICE.txt). The Codex Responses request shape was checked against [openai/codex revision 922510f4d7570b42838e015787ed53abfa8761a1](https://github.com/openai/codex/tree/922510f4d7570b42838e015787ed53abfa8761a1/codex-rs/codex-api/src), especially `common.rs` and `endpoint/responses.rs`. The upstream Apache 2.0 license is unchanged.


## Ordinary launches outside a test harness

The robot's application-home `.env` must provide `DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_URL` and `DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_TOKEN`. The main app loads this file before opening voice. Keep it private and use only the scoped broker client credential, never the account's OAuth material. Leave the installed package's legacy provider `.env` unchanged.

`deployment/run_broker.py` bridges a systemd `broker-token` credential into the ordinary broker CLI. `deployment/dj-reachy-remix-subscription-broker.service` runs the host-owned virtual environment and bridge with a loopback-only listener on port 18766. Install the bridge under the host broker directory and the service in the user's systemd directory. The existing private SSH forward carries robot loopback requests to that host listener. Do not run the exhausted test-budget broker and the ordinary broker on the same port.

The ordinary broker is idle until a client requests a session or plan. Starting it does not start robot capture or playback. It is not enabled at login by this setup. An application launch remains an explicit action. Keep test request caps in a separate disposable test broker and ledger; never reset a consumed ledger to make the ordinary app work.
