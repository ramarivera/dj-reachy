# DJ Reachy fork development

This repository is Ramiro's standalone derivative of https://huggingface.co/spaces/mzurki/dj-reachy. Preserve the upstream Apache 2.0 license, history, app entry point and local dance controller. Attribute any reused Conduit code with its MIT notice and source revision.

## Scope and acceptance

Implement responsive stop/pause/resume and cancellation during music, explicit Eleven Music model/duration/prompt-or-versioned-plan/style/instrumental controls, saved tracks, subscription-only ChatGPT Live behind a replaceable voice adapter, explicit API-key Realtime selection, and separate conversational/music/motion settings. Expose the controls through the existing app UI and launch path, not only unused classes.

Runtime completion and provider entitlement are separate. No robot installation, app stop/start, physical motion, microphone recording or speaker playback is authorized here. Only the parent integration owner may perform the separately authorized single subscription handshake and single 10-second music generation. Workers must make no authenticated provider requests or read real credentials.

## Engineering

- Write and run a focused failing behavior test before each implementation slice, then verify green. Test actual execution paths and concurrency, not only mocked helper returns.
- Keep timing-critical movement and motor bounds local. Preserve current app/provider IDs.
- Separate provider protocol/auth, task execution, music generation and playback. Subscription failure must never choose API-key billing.
- Keep music playback in independently owned cancellable tasks. A voice receiver must not await song completion. Pause/stop must clear queued audio and reset dance offsets. Stale jobs must not restart playback or unmute a replacement owner.
- Never report upstream generation cancellation or a credit refund solely because a local task was canceled.
- Validate provider requests before network I/O. Music v1 sections differ from v2/v2.5 chunks. Do not invent Music voice_id support or imply account access to persistent Vocals.
- Never log, copy into settings, commit or return provider keys or subscription credentials. No placeholder credential fallback in production. Tests use explicitly injected fixtures.
- Do not add runtime dependencies or change CI workflows without the parent's approval. Existing declared dependencies and stdlib are available; optional missing capabilities must fail clearly.
- Preserve hooks, tests and unrelated work. No destructive git, force pushes, disabled checks or fabricated verification results.

## Test isolation

All build outputs, package environments, provider fixtures and caches belong under `~/dev/agentic-scratchpads/dj-reachy/fork-build/`, never system /tmp. Run tests with owned HOME, XDG config/data/state/cache roots, TMPDIR and DJ_REACHY_HOME. No test may read real OAuth/key files, refresh real tokens or initialize a real robot/provider.

Use the interpreter and commands documented by the integration owner. Keep fixture traffic on loopback. Real SDK protocol models must retain their types, especially DaemonStatus; do not replace a model with a dict and call startup verified.
