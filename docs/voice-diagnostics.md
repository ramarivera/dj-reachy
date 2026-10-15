# Voice diagnostics

Voice sessions use Python's standard logging stack to emit structured JSONL diagnostics. No new logging dependency is required. These diagnostics locate missing input without retaining speech.

## Storage and privacy

The file is `logs/audio-trace.jsonl` under `DJ_REACHY_HOME`. With no override, the home is `dj-reachy` under `XDG_DATA_HOME`, or `~/.local/share/dj-reachy`. On the qualified robot installation this is `/home/pollen/.local/share/dj-reachy/logs/audio-trace.jsonl`.

Each file is limited to 1 MiB, with three backups. The log directory is mode `0700` and files are mode `0600`. Diagnostic records also go to stdout. File errors must not stop voice processing. Handlers belong to the voice lifetime and close at shutdown.

Only aggregate signal and flow metrics are retained. Diagnostic files do not capture raw audio samples, transcripts, prompts, instructions, SDP, addresses, authorization headers, credentials, tool arguments or provider response/error bodies. Unknown event and tool names become fixed `other` categories. Existing application logs are separate; this privacy guarantee applies to the diagnostic logger, not every third-party logger.

## Follow one run

Every record has a `run_id` and monotonic `elapsed_s`. Routine observations are throttled to five seconds per boundary. The first observation and coalesced format/silence changes are visible sooner. Counters continue updating between emitted records.

The native audio path records these boundaries in order:

1. `capture`: SDK frame dtype, shape, sample rate and per-channel peak/RMS/nonzero counts. `capture_configuration` identifies the known backend class and SDK-reported channel count/rate without serializing device configuration.
2. `selected`: mono audio after the handler's existing channel-zero selection.
3. `resampled`: audio converted to 24 kHz.
4. `encoded`: PCM16 audio before base64 encoding.
5. `append` and `feed`: PCM entering the subscription microphone track, byte totals, backlog and trimming.
6. `track`: paced 48 kHz WebRTC frames, including nonzero counts and silence-padding underruns.
7. `webrtc_stats`: actual outbound audio RTP packet/byte totals and connection, ICE, signaling and data-channel states.

The Gradio path shares handler and subscription diagnostics, but does not pass through the native SDK capture loop.

`input_dropped` records fixed reasons such as no connection, failed append, missing microphone, ended track, trimmed backlog or conversion failure. Empty capture polls are counted too; they do not by themselves mean the microphone failed.

Provider records count received and handled event categories, without event content. Lifecycle records count connection, delegation, planner and allowlisted music-tool starts, completions, failures and cancellations. Tool failures handled as results can leave a completed delegation with a failed tool count; inspect both.

At shutdown, `run_summary` includes cumulative counts, per-boundary nonzero totals, the last observed boundary, the furthest reached boundary, the furthest boundary with nonzero audio, and final sampled RTP totals. A paced track can keep sending silence after useful input disappears, so reaching `track` is not enough.

## Interpret the evidence

- Nonzero raw capture with silent `selected` audio isolates channel selection. The handler currently keeps channel zero; this diagnostic change deliberately does not switch or mix channels.
- `resampled` without `encoded`, plus `conversion_failed`, isolates conversion before append. Non-24-kHz int16 input has a separately reproduced float64 conversion problem; the inspected SDK capture uses float32. Logging that exception does not repair the input contract.
- Nonzero encoded input with no feed/track progress points to the local append or microphone-track boundary.
- Growing underrun totals indicate that the track is padding missing input. Some padding at startup is expected.
- Nonzero track frames and outbound RTP progress establish local send activity, not provider receipt, speech recognition, or a successful conversation.
- Provider input-transcript/turn categories without delegation or tool progress move diagnosis beyond capture and transport.

Energy is not proof of speech. Background noise is nonzero too. Do not invent a VAD or microphone-enable protocol change from the absence of a reply.

## Hardware test conditions

Preserve the user's existing speaker level unless a different level is explicitly needed and approved. A raw ALSA percentage is not a perceptual loudness percentage. On the inspected robot, the PCM mixer spans 0 to 60 with a linear -60 to 0 dB scale: setting 20% selects -48 dB, while the existing 88% setting reads -7 dB. Lowering to 20% made the test output nearly inaudible. Do not repeat that test setup or use the volume endpoint merely to check a value; its setter plays a sound.

Check timing as well as energy. Native SDK capture can wait synchronously for a GStreamer sample, so it must not block the asyncio loop that owns WebRTC and controls. In a device-free on-robot probe, the actual SDK with a synthetic GStreamer source delivered the same 186 frames over three seconds in both modes; direct capture allowed 187 heartbeat ticks, while offloaded capture allowed 557. This isolates a scheduling cost, not the complete cause of the failed voice conversation. A separate observed live stream fell below real-time pacing despite nonzero samples.

Native capture and playback each have one owned worker and bounded handoff, rather than running synchronous SDK calls on the voice loop. Flush invalidates queued PCM immediately, fences any already-running push, then clears the native buffer before accepting replacement audio. Cancellation does not pretend to terminate a native thread. Normal shutdown drains both directions before either media stop runs.

A permanently stalled native operation cannot safely return through the app's unconditional media-close cleanup. If native teardown fails to quiesce within two seconds, the standalone app logs a critical error and terminates its own process with status 70 instead of claiming clean shutdown or closing resources under a live native call. Run this native path in its app process, not embedded in another long-lived Python host. A supervisor may restart it only under the user's session policy; termination is not proof of a neutral-pose reset or provider cancellation.

## Unattended bounded tests

Repeated tests within the established per-test caps do not need another approval or presence confirmation. Prefer locally synthesized speech injected through the installed media/voice path, with output captured silently and movement targets intercepted. Do not make the user repeat test phrases. Keep room microphone, physical speaker, camera and physical motion disabled for this synthetic qualification.

Give every trial a fresh private sandbox and durable request ledger. Per trial, allow at most one subscription session of 120 seconds, four planning calls and one 10-second music generation. Reject API-key fallback and additional generation requests. Synthetic ASR, model replies and real tool results must be observed, never fabricated. Keep synthetic-path success separate from physical-device audibility. This permission does not authorize public exposure or account-credential transfer.

## Sustained speech-motion qualification

Do not qualify progressive stutter with a short no-motion audio probe. Exercise the real `HeadWobbler`, `MovementManager`, inbound Opus decoder worker, adapter audio reader, handler queue, outbound Opus encoder and native media calls together. Use synthetic source/sink endpoints and intercepted robot targets when physical testing is not authorized. Supply inbound frames on an anchored 50 Hz clock, not a sleep-after-processing loop that silently reduces the workload as the app slows. Compare early and late cadence and queue depth over at least 60 seconds.

The speech-motion analysis only consumes its latest 320-sample frame. Keeping ten seconds of samples made `islice` traverse up to 160,000 values every hop. The deterministic regression counts values traversed, avoiding a CPU-dependent timing assertion, and compares motion output against the legacy window across sample rates and chunk boundaries.

On the same robot, a 60-second device-free duplex test with the old window produced 2,434 outbound packets and 2,640 playback pushes while its playback queue grew to 273 frames at the last sample. Changing only the analysis-window bound produced 3,000 outbound packets and 3,000 playback pushes with zero playback backlog at every sample. This reproduces a progressive processing failure and verifies its offline repair. It does not establish physical audibility, the full severity of a provider-connected failure, or successful spoken music control. Keep those acceptance gates separate.

## Qualification limits

Offline tests exercise synthetic stereo frames through the real native capture loop, handler selection/resampling/encoding, subscription append and microphone track. Loopback WebRTC tests check actual RTP statistics and cleanup. Privacy, rotation, rate limiting and owned-resource closure are tested separately.

The first failed physical session lacked these boundary logs. A later instrumented run established nonzero capture on both channels, nonzero selected/encoded/track audio and outbound RTP progress, with late generated output but no usable response reported by the user. Do not present that as a successful physical conversation or a proven model-version problem. A new microphone/speaker test or subscription session requires fresh authorization and a bounded test window; do not automatically reconnect to gather more evidence.
