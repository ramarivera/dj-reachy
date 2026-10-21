---
title: DJ Reachy Remix
emoji: 🎧
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Generate music and dance with Reachy Mini
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# DJ Reachy Remix

A standalone derivative of [mzurki/dj-reachy](https://huggingface.co/spaces/mzurki/dj-reachy). It preserves the upstream Apache 2.0 history and local audio-reactive dance engine, with a separate release identity so it can be installed alongside vanilla DJ Reachy.

The control panel separates conversation and personality, Eleven Music generation, and bounded local motion. Saved tracks can be listed, previewed, replayed without regeneration, and deleted. Pause, resume and stop use the same music tools as the voice handler.

## Separate fork identity

| Identity | Vanilla DJ Reachy | DJ Reachy Remix |
| --- | --- | --- |
| Python distribution / CLI | `dj-reachy` | `dj-reachy-remix` |
| Python import namespace | `dj_reachy` | `dj_reachy_remix` |
| Reachy app-manager slug | `dj_reachy` | `dj_reachy_remix` |

Remix stores settings, provider configuration, instructions and saved tracks under `~/.local/share/dj-reachy-remix`, or `$XDG_DATA_HOME/dj-reachy-remix`. `DJ_REACHY_REMIX_HOME` overrides that root. It deliberately ignores the old `DJ_REACHY_HOME` selector and ships no `dj_reachy` import or console compatibility alias. Default model-cache storage is separate too. Provider model IDs and the host-owned subscription authorization flow are unchanged.

Both apps can be installed, but the SDK app manager still gives one app control of the robot at a time. Their hardware and settings server are not intended to run concurrently. The GitHub repository is `ramarivera/dj-reachy-remix`, matching the app identifier; the old `ramarivera/dj-reachy` URL redirects.

Earlier development builds of this fork used vanilla's package identity. Do not remove an installed `dj-reachy` package assuming it is the fork. Install Remix separately, inspect the app-manager entries, and restore or update vanilla from its canonical source only when intended. Remix does not automatically copy or modify the old application's data. If you need data from an earlier fork build, explicitly copy only the settings, instructions and tracks you own into the Remix home, then configure its credentials separately.

## Install and configure

Python 3.10 or later and the Reachy Mini SDK are required. Music playback also requires the decoder used by the music implementation. From this checkout, install into your own environment with `pip install -e .`, or install the wheel built by your integration owner. Don't install into a shared development environment.

Configure without connecting to a robot, recording a microphone, or contacting a provider:

```nu
dj-reachy-remix --settings-only
```

Open <http://127.0.0.1:7860/>. The default voice backend is ChatGPT subscription Live. Missing subscription configuration never selects API-key billing. Subscription access and the consumer Live endpoint's account entitlement require a separate authorized qualification check. UI construction or a configured key doesn't prove access.

To explicitly select OpenAI API-key Realtime and reset the subscription model/voice defaults:

```nu
dj-reachy-remix --settings-only --voice-backend api_key
```

Save the OpenAI key under Provider credentials. Save an ElevenLabs key there for music generation. Keys aren't returned by the settings API or embedded in the Gradio config. They are separate from the nonsecret JSON settings.

## Subscription music confirmation

Subscription Live carries the spoken conversation. A genuine Live delegation starts unbilled preparation of a music draft, not generation. The app returns the complete request and duration for a preview, then sends only replies to that pending draft to the subscription planner for semantic confirmation, correction or refusal. A second Live delegation and a fixed approval phrase aren't required. Ordinary conversation without a pending draft doesn't trigger this extra planning.

Approval consumes the frozen request once. Corrections or changed music settings require another preview and a fresh reply. A draft lasts at most 90 seconds and four assessed reply turns across corrections. Refusal, stop and disconnect discard it; stale approval can't revive it. Silence and assistant narration never authorize music generation.

This flow uses additional subscription planner calls while a draft awaits a reply. It doesn't replace Live speech with a transcript-to-TTS pipeline or select API-key billing. Explicit local UI/API generation and the separately selected API-key Realtime backend keep their existing paths.

Offline tests can check state, transport and cancellation with synthetic fixtures. They don't establish real-model consent interpretation or reliable integrated robot operation; those need separately authorized live qualification.

## Run with an authorized robot

These commands connect to the robot and may record audio, play speakers and start motion services. Run them only when those actions are authorized.

```nu
# Browser microphone and Gradio conversation at /voice; controls remain at /
dj-reachy-remix --gradio

# Headless robot microphone/speaker loop; controls at /
dj-reachy-remix
```

The app-manager entry point uses the SDK's existing settings server. The standalone CLI starts its own loopback server. `--gradio` exposes the same shared control panel inside the voice page, not a second music executor. Stop remains usable while a long generation/replay HTTP request is running.

Conversation model, voice, language and user instructions apply on the next voice connection. Disconnect and reconnect browser voice after saving. In headless mode, restart the app to apply them to a connected session. A session that is already offline can use Reconnect voice in the control panel instead, unless its cleanup failed. Music defaults apply to subsequent generation requests. Motion settings update an already-created local dancer as well as future playback.

See [configuration and API usage](docs/configuration.md) for storage paths, versioned plan editing, billing selection and local endpoint examples.

## Verification boundaries

This UI implementation has isolated HTTP, real SDK status-model and app-construction regression tests. The settings-only CLI and responsive browser controls can be exercised without hardware or credentials. No live provider handshake, paid generation, microphone recording, speaker playback, robot launch or physical motion was performed by this lane. The parent integration owner must qualify the combined voice/music implementation and built wheel separately.

The subscription and music modules have independent protocol and entitlement requirements. A local cancellation doesn't prove that the provider canceled generation or refunded credits. `finetune_id` is an optional account-qualified Music control, not a TTS voice ID or a guarantee of persistent Vocals access. Unsupported implementations reject it before generation.

## Development

Run tests in an isolated app home with no real credentials. Set `PYTHONPATH` to this checkout's `src`, and put HOME, TMPDIR, XDG roots, HF_HOME, package caches and all outputs under an owned scratch directory. Set `HF_HUB_OFFLINE=1` and `GRADIO_ANALYTICS_ENABLED=False`. Never run the suite against your normal OAuth/key files.

```nu
pytest -q
ruff check src/dj_reachy_remix/runtime_settings.py src/dj_reachy_remix/config.py src/dj_reachy_remix/gradio_dj_ui.py src/dj_reachy_remix/console.py src/dj_reachy_remix/main.py src/dj_reachy_remix/prompts.py
mypy src/dj_reachy_remix/runtime_settings.py src/dj_reachy_remix/config.py src/dj_reachy_remix/gradio_dj_ui.py src/dj_reachy_remix/console.py src/dj_reachy_remix/main.py src/dj_reachy_remix/prompts.py
```

## Releases

Pushing a `v<version>` tag that matches `pyproject.toml` runs `.github/workflows/release.yml`. It builds the sdist and wheel, publishes them to PyPI as `dj-reachy-remix` through trusted publishing (GitHub OIDC, no stored token) from the `pypi` environment, and attaches them to a GitHub release.

## License

[Apache 2.0](LICENSE). Original DJ Reachy app by Max Zurkinden. Preserve the existing source attribution and any third-party notices when distributing the integrated fork.
