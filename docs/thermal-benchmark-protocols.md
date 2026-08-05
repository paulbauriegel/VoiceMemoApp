# Android thermal benchmark protocols

These protocols measure VoiceMemo performance and the Pixel thermal response
without changing Android's thermal controls. They are intentionally separate
from the controlled backend suite: use them after the current investigation,
with one named workload and one exact audio artifact at a time.

## Continuous temperature capture

Start the host-side sampler immediately before the measured workload:

```sh
scripts/sample-android-thermal.py \
  --serial DEVICE_SERIAL \
  --output build/reports/thermal/RUN_ID.ndjson \
  --interval-seconds 1 \
  --label RUN_ID \
  --metadata audio_sha256=AUDIO_SHA256 \
  --metadata backend=BACKEND
```

Stop it with `Ctrl-C`, or add `--duration-seconds SECONDS`. The first NDJSON
record describes the session; later records contain UTC and monotonic times,
Android thermal status, every sensor from the live `Current temperatures from
HAL` section, and the battery snapshot. A neighboring `.summary.json` records
sensor extrema and the first time each thermal status was observed. Existing
outputs are never overwritten.

Sensor names and meanings are vendor-specific. On the Pixel 7 Pro, useful
names currently include `BIG`, `MID`, `LITTLE`, `G3D`, `TPU`, `VIRTUAL-SKIN`,
and `battery`. Preserve every sensor in the raw data, but predeclare which
sensor answers each question before comparing runs:

- `VIRTUAL-SKIN`: primary phone/body heating measure and throttling context.
- `BIG`, `MID`, `LITTLE`: CPU-cluster temperature response.
- `G3D` or `TPU`: accelerator response when the selected backend uses it.
- `battery`: slow-moving secondary measure, not a substitute for SoC or skin.
- Android thermal status: authoritative throttling-state label.

Use the live HAL section rather than `Cached temperatures`; cached and live
values can differ substantially. Keep the raw NDJSON with the benchmark report
so later analysis does not depend on a renamed or reinterpreted sensor.

## Controls shared by every protocol

Record these before each series and keep them fixed unless they are the tested
variable:

- device model, build fingerprint, Android version, app commit/APK hash;
- backend, model hash, worker/thread configuration, and language setting;
- exact audio bytes and SHA-256, duration, sample rate, channel count, VAD
  fraction, output-token count, and retry/truncation warnings;
- case on/off, phone orientation and supporting surface, ambient temperature,
  airflow or external cooling, screen state/brightness, radios, and power mode;
- charging method and battery level. Prefer a fixed battery-level band and do
  not mix USB-powered, battery-powered, and wireless-charging runs;
- start temperatures, thermal status, thermal headroom when available, and the
  time since the preceding workload.

Do not use external heat sources, disable thermal protection, override thermal
status, or place the phone in a freezer. Abort on Android's `CRITICAL`,
`EMERGENCY`, or `SHUTDOWN` status, a device warning, instability, or an
unexpected cooling-device response. An aborted or missing state is a valid
censored result, not a reason to force the phone hotter.

For comparative work, use at least three valid repeats per cell and randomize
or counterbalance run order. Do not compare a resumed transcription with a
fresh full run. Preserve failed and interrupted runs with their reason rather
than silently rerunning them.

## Protocol A: effect of the audio sample on temperature

Purpose: determine whether audio duration or audio-dependent decoding work
changes heat generation and sustained performance.

1. Define the hypothesis before selecting samples:
   - To test duration, use one source recording cut into increasing durations.
   - To test content, use byte-distinct samples with matched duration, sample
     rate, channels, and approximately matched VAD-active fraction.
   - Include silence only as an explicit VAD-control cell.
2. Pin one backend/model/configuration for the complete series. Precompute and
   record the audio hashes and provenance.
3. Begin every run at thermal status `NONE`. Also require a stable primary skin
   baseline: collect at least 180 seconds before starting and accept the run
   only when the final 120-second `VIRTUAL-SKIN` slope is within
   ±0.1 °C/minute. If this is impractical, predeclare a wider threshold and use
   it for every run.
4. Start temperature capture, wait 10 seconds for a pre-workload baseline, run
   exactly one fresh full transcription, then continue capture for at least
   120 seconds of recovery.
5. Repeat each sample at least three times in randomized order. If one sample
   causes a retry or materially different token count, retain the run but
   analyze it separately rather than attributing the extra work to waveform
   temperature alone.

Primary measures:

- peak and end `VIRTUAL-SKIN` delta from the 10-second pre-workload median;
- temperature area above baseline in degree-seconds during transcription;
- maximum CPU/GPU/TPU sensor delta and time to peak;
- transcription wall time, inference time, RTF, tokens, VAD-active duration,
  and joules if energy collection is later added;
- recovery time to within 0.5 °C of the pre-workload skin median.

Report temperature delta per minute of input audio in addition to the absolute
delta, but do not use that normalized value alone: heating and cooling are not
linear over long workloads.

## Protocol B: performance at observed throttling states

Purpose: quantify performance loss at Android's real `NONE`, `LIGHT`,
`MODERATE`, and, only if reached naturally and safely, `SEVERE` states.

1. Use a short, fixed measurement unit that completes quickly enough to remain
   mostly within one state, such as one pinned 28-second Whisper window. Verify
   identical input bytes, model/runtime fingerprint, and decoding settings.
2. From a stable `NONE` baseline, use the actual transcription workload as the
   heat source. Do not inject or override a thermal status.
3. When the sampler first observes a target state, begin the next measurement
   unit and tag its start/end timestamps. Record every status transition during
   the unit; classify a unit as single-state only if it stayed in that state.
4. Collect at least three single-state units per reachable state. Keep
   transition-spanning units as separate evidence rather than assigning them
   to either endpoint.
5. End the heating phase after the predeclared maximum duration, after enough
   state cells are collected, or immediately on the abort conditions above.

For each state report median/range of inference time, RTF, CPU time, PSS, token
count, and the primary sensor temperatures at start/end. Compare each state to
the same-session `NONE` units. Thermal status is system-wide, so the associated
temperature is context, not a universal threshold transferable to another
device or Android build.

## Protocol C: time to throttle and recovery

Purpose: measure how quickly a fixed sustained workload enters each throttling
state and how quickly the device recovers afterward.

1. Use the stable 180-second baseline rule from Protocol A and start at
   `NONE`. Remove external cooling unless cooling is the explicit independent
   variable.
2. Start sampling, record 10 seconds of baseline, then continuously repeat the
   same pinned inference unit without idle gaps. Set a maximum heating duration
   in advance; 30 minutes is a reasonable first cap for exploratory runs.
3. Record elapsed time from workload start to the first observation of
   `LIGHT`, `MODERATE`, and `SEVERE`, plus the temperature and completed-work
   count at each transition. A state not reached before the cap is
   right-censored at the cap.
4. Stop the workload at the cap or abort condition, but continue sampling
   during passive recovery until status returns to `NONE` and skin temperature
   is within 0.5 °C of baseline, subject to a predeclared recovery timeout.
5. Repeat at least three times from an independently stable baseline. Do not
   treat consecutive heating passes separated by a fixed short wait as
   independent unless the stability criterion is satisfied.

Report a status timeline, time-to-first-state, time spent in each state,
throughput/RTF over fixed time bins, temperature slopes, peak temperatures,
and recovery time. Plotting RTF and sensor temperatures on the same monotonic
time axis makes the onset and magnitude of throttling directly visible.

## Minimum result manifest

Each result directory should contain:

- the raw thermal NDJSON and generated summary;
- the content-free VoiceMemo benchmark JSON and raw backend timings;
- an experiment manifest containing all shared controls and audio/model hashes;
- explicit workload start/end monotonic timestamps or an alignment event;
- protocol ID/version, run/cell/repeat IDs, validity decision, and exclusion or
  abort reason;
- derived tables/plots generated from preserved raw inputs.

Do not publish memo identifiers, transcript text, audio, or app-private database
copies as part of the thermal report.
