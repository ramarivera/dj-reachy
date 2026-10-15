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

## Qualification limits

Offline tests exercise synthetic stereo frames through the real native capture loop, handler selection/resampling/encoding, subscription append and microphone track. Loopback WebRTC tests check actual RTP statistics and cleanup. Privacy, rotation, rate limiting and owned-resource closure are tested separately.

The failed physical session did not have these boundary logs, so its cause remains unproven. Installing this revision does not establish that live speech works. A new microphone/speaker test or subscription session requires fresh authorization and a bounded test window; do not automatically reconnect to gather more evidence.
