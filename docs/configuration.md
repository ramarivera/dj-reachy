# Configuration and local controls

## User-owned storage

`DJ_REACHY_REMIX_HOME` is the explicit storage root. Otherwise the app uses `$XDG_DATA_HOME/dj-reachy-remix`, falling back to `~/.local/share/dj-reachy-remix`. The root contains:

- `settings.json`, validated nonsecret `voice`, `music` and `motion` groups.
- `instructions.md`, user personality additions loaded on each new voice connection.
- `songs/`, saved MP3 tracks and their metadata.
- `.env`, only provider keys submitted through the credentials endpoint, written atomically with mode 0600.

Settings and instructions also use private same-directory temporary files, fsync and atomic replacement. Invalid settings leave the previous file and live config unchanged. Symlinked settings, persona, tracks and track sidecars aren't followed. The SDK's auto-generated instance path may point into site-packages; it never becomes a write root. A legacy instance `.env` can supply migration values, but the app doesn't overwrite it. Explicit process environment takes precedence over user `.env`, which takes precedence over legacy `.env`.

Choose a separate root in Nushell:

```nu
with-env {DJ_REACHY_REMIX_HOME: ($env.HOME | path join ".local" "share" "dj-reachy-remix") } {
  dj-reachy-remix --settings-only
}
```

`DJ_REACHY_REMIX_UI_HOST` defaults to `127.0.0.1`; `DJ_REACHY_REMIX_UI_PORT` defaults to `7860`. The SDK-managed app retains its SDK server address. The control API can modify settings, credentials and playback. Don't expose it to an untrusted network. Use a trusted local connection or an authenticated reverse proxy; the app doesn't provide user accounts.

## Conversation and billing

The default voice group is:

```json
{"backend":"subscription","model":"gpt-live-1-codex","voice":"cove","language":"auto"}
```

Select `api_key` explicitly in the UI or with `--voice-backend api_key`. A backend-only change resets incompatible model/voice defaults to `gpt-realtime` and `cedar`. API-key mode rejects subscription models and `cove`; it never inherits them accidentally. Selecting subscription resets to its own defaults and never consumes an OpenAI API key as a fallback.

Subscription broker URL and scoped token remain environment-only through `DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_URL` and `DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_TOKEN`. The voice implementation owns broker/credential-source setup and qualification. These values aren't settings JSON fields or public status values. The consumer ChatGPT Live contract and account access aren't established by this settings UI.

The voice model, conversation voice, conversation language, and personality instructions apply on the next connection. Save, disconnect browser voice, and reconnect it at `/voice`. Headless sessions reconnect through Reconnect voice in the control panel once they are offline; an app restart is needed only after a failed audio or voice cleanup. This is deliberate; saving a form doesn't automatically create a billable session. The API response for saved instructions includes `reconnect_required: true`.

User instructions add personality to the packaged DJ instructions; they don't replace the app's paid-generation confirmation and stop/pause policy. Conversation language and lyric language are separate. Conversation voice doesn't choose a singer.

## Music

The Music generation form exposes model, duration in seconds, styles and exclusions, original lyrics, lyric language, prompt-only instrumental mode, compatible MP3 output formats, and an optional qualified `finetune_id`.

Choose exactly one request mode:

- Prompt, with a nonempty song description.
- Versioned composition plan JSON. Music v1 uses `sections`. Music v2 and v2.5 use `chunks`. Use the schema for the selected provider model. The total duration must match the requested duration. The editor doesn't convert between model schemas.

The form disables force-instrumental in plan mode. The server uses the music module's offline `build_music_payload` validation before invoking `make_song_and_dance`. Model-specific provider validation remains owned by the music implementation. Saving music defaults doesn't generate anything. The Generate button requires a separate confirmation that the new track may spend credits.

Only playback-compatible MP3 formats are exposed. Raw PCM, codecs that the MP3 decoder can't consume, and unsupported singer controls aren't offered. A Music `finetune_id` isn't a TTS `voice_id`. Leave it blank unless the account and selected model have qualified access. If the music implementation doesn't support that field, generation returns an actionable local validation error rather than silently ignoring it.

Pause, resume, stop, generation and replay call the existing registered music tools with `ToolDependencies`. The receiver isn't replaced by a second UI music engine. Headless controls cross to LocalStream's owning loop. Browser controls locate FastRTC's active component handler copy, not the disconnected template. Stop stays on a separate request and doesn't wait for a long generation request to finish.

A stop response describes local work only. The provider may continue a paid job. Replay uses the saved filename and doesn't request regeneration. Preview audio never autoplays. Deletion requires confirmation in the browser and refuses track/metadata symlinks.

## Motion

The motion group has `style` with values `groove`, `bounce`, `smooth`, or `classic`, and numeric `intensity` between `0` and `0.25`. Booleans, NaN, values outside that range and nested objects are invalid. Saving updates runtime config and the already-created dancer through the existing local setter functions. Timing, motor limits and offsets stay in the local motion implementation.

## Local HTTP API

The same endpoints exist in the app-manager, standalone headless and Gradio modes. The settings-only CLI exposes them without constructing a robot or starting voice. Settings writes merge supported fields and validate the complete result before replacement. Unknown groups, unknown fields, credentials in settings, malformed JSON and wrong/nested types return 400.

```nu
# Nonsecret settings only. Does not start a provider session.
curl --fail-with-body --request PUT http://127.0.0.1:7860/api/dj-reachy-remix/settings --header 'Content-Type: application/json' --data '{"voice":{"backend":"api_key"},"music":{"duration_ms":10000,"language":"Japanese"},"motion":{"style":"smooth","intensity":0.12}}'

curl --fail-with-body http://127.0.0.1:7860/api/dj-reachy-remix/settings
curl --fail-with-body http://127.0.0.1:7860/api/dj-reachy-remix/status
```

| Method and path | Behavior |
| --- | --- |
| `GET /` | Wheel-packaged control panel |
| `GET /voice` | Browser conversation, only when launched with `--gradio` |
| `GET/PUT /api/dj-reachy-remix/settings` | Read or atomically update the three nonsecret groups |
| `GET/PUT /api/dj-reachy-remix/instructions` | Read or save `{"instructions":"..."}`; reconnect required |
| `POST /api/dj-reachy-remix/credentials` | Persist `{"provider":"openai" or "elevenlabs","key":"..."}`; never returns the value |
| `GET /api/dj-reachy-remix/status` | Selected backend/billing, configured-key booleans and explicitly unverified provider access |
| `GET /songs` | Regular saved-track filenames, titles and sizes |
| `GET/DELETE /songs/{filename}` | Read or delete one owned regular MP3 track |
| `POST /api/dj-reachy-remix/music/replay` | `{"filename":"exact-name.mp3"}`, no regeneration |
| `POST /api/dj-reachy-remix/music/pause`, `/resume`, `/stop` | Empty object, existing tool call |
| `POST /api/dj-reachy-remix/music/generate` | Existing tool parameters, with `music_length_ms`, and exactly one prompt or composition_plan |

Playback returns 409 if no audio session is available. The settings-only mode never silently creates a provider connection to satisfy a playback request. Status says `unverified` even when a key is configured; it isn't a provider health probe.

## Recent changes

The UI lane replaced the inherited key-only settings page and incomplete Gradio callbacks with one shared responsive control panel. It fixed local FastAPI Request annotation resolution, SDK DaemonStatus attribute access, installed-package path ownership, explicit backend defaults, and private atomic credential persistence. A loopback-only tool smoke decoded synthetic MP3 into the handler queue and exercised pause, resume, stop, replay and deletion. An isolated installed-wheel consumer served the packaged HTML, JavaScript and CSS, and its actual CLI served settings-only mode. These checks don't establish provider entitlement, speaker playback, robot startup or physical motion.
