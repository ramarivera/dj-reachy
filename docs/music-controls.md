# Music controls

Music uses ElevenLabs `POST /v1/music`, independently of conversational speech. The default music model is `music_v2_5`. Selecting a conversational voice never selects a singer, changes Music billing, or supplies a Music credential.

## Run the existing app

Use the existing entry point when the integration owner has authorized app and hardware startup:

```nu
$env.DJ_REACHY_REMIX_HOME = ($env.HOME | path join ".local/share/dj-reachy-remix")
dj-reachy-remix --gradio --no-camera
```

This command starts the existing app, not an offline music simulator. It wasn't run against a robot in this implementation lane. `mpg123` must be installed for playback. The installed OpenAI dependency already supplies `httpx`; no package installation or dependency change is needed for cancellable generation.

The runtime settings groups remain `voice`, `music`, and `motion`. Music reads these existing config attributes at each new generation: `MUSIC_MODEL_ID`, `ELEVEN_MUSIC_LENGTH_MS`, `MUSIC_OUTPUT_FORMAT`, `MUSIC_POSITIVE_STYLES`, `MUSIC_NEGATIVE_STYLES`, `MUSIC_LYRICS`, `MUSIC_LANGUAGE`, `MUSIC_FORCE_INSTRUMENTAL`, and `MUSIC_FINETUNE_ID`. Explicit tool arguments override their corresponding defaults, including empty style lists and `force_instrumental=false`.

Provide the ElevenLabs key through the app's existing credential input or `ELEVENLABS_API_KEY`. It isn't saved in track metadata or runtime settings. Missing credentials fail rather than selecting another provider. Only the integration owner may perform the separately authorized provider qualification calls.

## Generate and control a track

The existing `make_song_and_dance` tool accepts this prompt-mode argument object:

```json
{
  "prompt": "An original upbeat salsa song with warm percussion",
  "music_length_ms": 10000,
  "model_id": "music_v2_5",
  "positive_styles": ["bright brass", "danceable rhythm"],
  "negative_styles": ["distorted guitar"],
  "lyrics": "Bailamos juntos esta noche",
  "language": "Spanish",
  "force_instrumental": false,
  "output_format": "mp3_48000_192"
}
```

Duration must be integer milliseconds from 3000 through 600000. Prompt and composition plan are mutually exclusive. `positive_styles`, `negative_styles`, `lyrics`, and `language` aren't top-level REST fields. The tool compiles them into prompt directions or version-specific plan fields. Styles accept at most 50 nonempty strings. The app's lyrics input accepts at most 30 lines of 200 characters. Language accepts a single line of at most 100 characters. Musical directions are requests to the model, not measured guarantees of language, instrumentation, or tempo.

`force_instrumental=true` is prompt-only and cannot accompany original lyrics. A plan must not use this flag. If the saved default is true, override it with false for plan mode.

Playback supports `auto` and the documented MP3 formats: `mp3_48000_128`, `mp3_48000_192`, `mp3_48000_240`, `mp3_48000_320`, `mp3_22050_32`, `mp3_24000_48`, `mp3_44100_32`, `mp3_44100_64`, `mp3_44100_96`, `mp3_44100_128`, and `mp3_44100_192`. `output_format` goes in the HTTP query string. `auto` resolves to the documented model default, MP3 44.1 kHz/128 kbps for v1 or 48 kHz/192 kbps for v2/v2.5. PCM, Opus, µ-law, and A-law are rejected locally because the existing decoder doesn't support those carriers. They aren't mislabeled as `.mp3`.

Controls use empty argument objects:

- `pause_music` stops PCM delivery, flushes queued/backend audio, fades the local dancer to zero, and releases music's conversation mute. The decoder's consumer position stays held. Speech can continue while music is paused.
- `resume_music` continues at the held decoder position and restores music's own mute. Frames resume at real-time cadence rather than catching up with a burst. Repeated pause/resume calls don't flush audio again.
- `stop_music` closes local generation or stops playback. Playback stop returns after decoder termination/reaping, pipe draining, buffer flushing, and the bounded motion fade. An uncooperative decoder receives SIGKILL after the one-second terminate grace period.

Audio is delivered in 960-sample frames at 24 kHz, nominally 40 ms. Pause holds future decoder consumption; already delivered frames and backend audio that was flushed aren't reconstructed. This is frame-boundary pause, not sample-accurate seeking in an audio device. The existing motion fade takes about 500 ms and runs outside the voice event loop.

Only one generation or replay may own music at a time. Another music job returns `busy`, including during pause and cleanup. Canceling the owning asyncio task also performs cleanup. Repeated shutdown cancellation cannot release admission while an older motion fade is still running. Music never takes the voice receiver's lock or waits for a conversational response.

Generation uses cancellable asynchronous HTTP, not a blocking request thread. Stop/cancellation closes a stalled header or body read, discards its local result, and prevents a late saved track or playback. Work already canceled while queued doesn't read a key or contact the provider. **Closing local HTTP doesn't prove that ElevenLabs stopped computation or refunded credits.** Successful generation remains saved when subsequent playback is stopped.

## Use a versioned plan

For v2/v2.5, pass `composition_plan` instead of `prompt`:

```json
{
  "music_length_ms": 10000,
  "model_id": "music_v2_5",
  "force_instrumental": false,
  "composition_plan": {
    "chunks": [{
      "text": "[Verse]\nBailamos juntos esta noche",
      "duration_ms": 10000,
      "positive_styles": ["salsa", "warm percussion", "bright brass"],
      "negative_styles": ["distorted guitar"],
      "context_adherence": "high"
    }]
  }
}
```

Plans require 1 through 30 chunks/sections, each 3000 through 120000 ms, with total duration equal to the requested `music_length_ms` and at most 600000 ms. V1 requires `positive_global_styles`, `negative_global_styles`, and `sections`. Each v1 section requires `section_name`, `duration_ms`, `positive_local_styles`, `negative_local_styles`, and `lines`. V2/v2.5 require `chunks` with `text`, `duration_ms`, and `positive_styles`. Nested types, ranges, style limits, lyrics limits, section names, supported fields, and conditioning enums are validated before HTTP. The caller's plan isn't mutated. Language directions go in style fields rather than becoming sung lyric lines. Original lyrics append to the first generation section/chunk, so its final lyrics limit must still hold.

Optional `finetune_id` forwards an existing Music finetune identifier. It doesn't upload or train anything, grant account access, or promise compatibility with a particular model. Music finetunes are permission-gated. Persistent Vocal Library identity isn't the TTS `voice_id`, and the reviewed compose schema doesn't establish a Vocal Library-to-API mapping. This app doesn't claim to supply one.

## Saved tracks and integration helpers

Tracks live in `config.SONGS_DIR`, under the configured user app root rather than site-packages. New names contain a timestamp, UUID, and sanitized title. Audio and normalized JSON metadata are staged with exclusive temporary files and fsynced. Metadata publishes first, then the atomic audio rename makes the pair visible to library readers. Normal write/rename/cancellation failures remove partial pairs. A process/power failure between the two renames can leave orphan metadata, which isn't listed as a song; this isn't a filesystem-wide two-file transaction.

Metadata contains the normalized request, title, UUID, creation timestamp, selected/requested output format, provider name, and a validated `song-id` header if supplied. It doesn't contain credentials, raw response headers, or provider error bodies.

Use `play_saved_song` with `{"filename": "the-exact-existing-name.mp3"}`. It never regenerates. Existing regular MP3 files without sidecars remain replayable. Replay opens a no-follow regular-file descriptor before spawning mpg123 so a filename replaced with a symlink cannot redirect the decoder.

The owned music module exposes these helpers for the settings/UI integration lane:

- `resolve_saved_song(filename) -> Path` validates an exact regular MP3 and optional regular sidecar.
- `load_song_metadata(filename) -> dict` reads a no-follow JSON sidecar, or returns an empty dict for legacy audio.
- `list_saved_songs() -> list[dict]` returns `filename`, `title`, `size_bytes`, and `created_at_ms`, newest first. Unsafe/malformed entries are omitted.
- `delete_saved_song(filename) -> bool` removes only a validated exact audio/sidecar pair.

Traversal, symlinks, non-MP3 paths, FIFOs, and other special files are rejected. The UI lane owns its list/download/delete handlers and must call the shared helpers rather than concatenate unchecked filenames. Legacy aggregate `metadata.json` isn't rewritten by these helpers.

## Offline verification

The shared interpreter is read-only at `../venv`. The lane's isolated runners and red/green logs are outside the repository in `../sol61-lanes/music/`. Run the full suite inside a loopback-only network namespace, with resource and unraisable exceptions treated as errors:

```nu
../venv/bin/python ../sol61-lanes/music/strict_run.py full-strict unshare --user --map-root-user --net sh -c 'ip link set dev lo up && exec "$@"' dj-test ../venv/bin/python -m pytest -q tests -W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning -W error::pytest.PytestUnhandledThreadExceptionWarning
../venv/bin/python ../sol61-lanes/music/run.py music-smoke python ../sol61-lanes/music/smoke_music.py
```

`strict_run.py` assigns owned HOME, DJ_REACHY_REMIX_HOME, CODEX_HOME, HERMES_HOME, all XDG roots including RUNTIME_DIR, TMPDIR, HF_HOME and caches. It strips inherited credentials, disables Hugging Face networking and Gradio analytics, and records the exact command and process exit code through pytest shutdown. Add `-X tracemalloc=15` before `-m pytest` to trace resource allocations. Passing assertions followed by a teardown failure aren't a successful gate.

`tests/conftest.py` owns an isolated event-loop policy and its idle restore loop. Pytest-asyncio's runner otherwise implicitly creates that idle loop while saving the ambient policy; a later synchronous `asyncio.run()` can detach it. The fixture closes only the restore loop it created, then restores the previous policy. The subprocess regression checks every recorded real loop's `is_closed()` state after pytest teardown, independently of GC timing.

The separate smoke constructs the real `OpenaiRealtimeHandler` without starting a provider connection, dispatches the registered tools against loopback HTTP, decodes real MP3 with mpg123, and intercepts only audio output and motion callbacks. It exercises generation, pause, resume, stop, replay, deletion and stalled HTTP cancellation. No robot, microphone or speaker is initialized.

The integrated registry refreshes missing concrete tools at its public spec and dispatch interfaces, preserving existing instances. Both music-first and registry-first imports are covered by subprocess regressions. Callers do not need a special import order.

Official request references: [compose](https://elevenlabs.io/docs/api-reference/music/compose) and [composition plans](https://elevenlabs.io/docs/eleven-api/guides/how-to/music/composition-plans). Provider entitlement, live provider behavior, combined UI wiring, SDK/app-manager startup, and physical audio/motion remain the integration owner's separate checks.
