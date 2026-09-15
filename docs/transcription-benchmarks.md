# Transcription benchmarks

Run the read-only Parakeet comparison against the existing Whisper CPU
baseline for an existing memo:

```sh
scripts/run-parakeet-whisper-benchmark.sh --memo-id MEMO_ID
```

The harness runs Whisper CPU, Parakeet CPU, and Parakeet GPU against the same
audio, reporting load/inference time, Parakeet WER versus Whisper, and
Parakeet GPU-versus-CPU WER. It installs both APKs with `adb install -r` and
does not create, replace, or delete recordings or transcript rows.

Qwen3-ASR is kept benchmark-only rather than bundled into the production APK.
Provide the saved Whisper baseline as a UTF-8 text file:

```sh
scripts/run-qwen-whisper-benchmark.sh \
  --memo-id MEMO_ID \
  --whisper-reference whisper-reference.txt \
  --backends cpu,gpu
```

The runner fetches and verifies the pinned
`litert-community/Qwen3-ASR-0.6B` five-second INT8 graph and Qwen tokenizer,
updates only the test APK in place, stages the model in app cache for the run,
and removes the temporary device copies afterward. Reports are written under
`build/reports/qwen-benchmark/`. Qwen long-form decoding uses 50%-overlapped
five-second windows and a fuzzy word-boundary merger because this graph does
not expose token timestamps.

whisper.cpp large-v3-turbo Q5_0, Q8_0, and FP16 are also available as benchmark-only
Vulkan/CPU models. They are not compiled into normal APKs, do not appear in
the app's model selector, and the selected GGML model is staged in app-private
storage only for an explicit run:

```sh
scripts/fetch-whisper-cpp-benchmark.sh all fp16
scripts/run-whisper-cpp-benchmark.sh run \
  --memo-id MEMO_ID \
  --model fp16 \
  --backend vulkan \
  --threads 4
```

The runner defaults to Q5_0 and Vulkan and rejects silent CPU fallback. Pass
`--model q8_0` for the 8-bit comparison, `--model fp16` for the unquantized
FP16 comparison, or `--backend cpu` only for an explicit CPU comparison.

The native source is pinned to whisper.cpp `v1.9.1` commit
`f049fff95a089aa9969deb009cdd4892b3e74916`. Q5_0 is
`ggml-large-v3-turbo-q5_0.bin`, 574,041,195 bytes, SHA-256
`394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`;
Q8_0 is `ggml-large-v3-turbo-q8_0.bin`, 874,188,075 bytes, SHA-256
`317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1`;
FP16 is `ggml-large-v3-turbo.bin`, 1,624,555,275 bytes, SHA-256
`1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`.
The runner builds whisper.cpp with `GGML_VULKAN=ON` under
`-PincludeWhisperCppBenchmark=true`, installs both APKs
in place, verifies the staged model again on-device, reads the selected memo
without persisting output, and removes the staged device model after a
successful run. Its JSON report uses the existing whole-memo timing, chunk,
memory, and transcript schema. Candidate-only reports measure performance and
output but do not establish WER unless compared with a separately saved
reference transcript.

For a controlled comparison of the six current reference configurations, run
the three-repeat suite:

```sh
scripts/run-whisper-benchmark-suite.py plan
scripts/run-whisper-benchmark-suite.py run \
  --serial DEVICE_SERIAL \
  --suite-id SUITE_ID
```

The suite runs whisper.cpp FP16 Vulkan, whisper.cpp Q8_0 Vulkan, LiteRT INT8
CPU/XNNPACK with 2 workers × 3 threads and 4 workers × 3 threads, LiteRT
FP16-storage OpenCL, and the LiteRT quantized-encoder/FLOAT32-decoder OpenCL
model. The old LiteRT INT8 1-worker × 3-thread row is deliberately excluded.
Every backend receives the same reference WAV and forced German language
token, each repeat starts at Android thermal status 0, and raw
text/tokens/timings plus host-recalculated WER evidence are stored under
`build/reports/whisper-benchmark-suite/SUITE_ID/`. If a run is interrupted,
repeat the command with `--resume`; existing reports and hashes are validated
before the next missing run starts.

## SKaiNET-audio frontend comparison

The SKaiNET-audio benchmark compares the current VoiceMemo WAV/mel frontend
with `sk.ainet.audio:audio-wav:0.2.1` and `audio-mel:0.2.1`. Both frontends use
the same bundled LiteRT Whisper model, two workers, three threads, forced
fixture language, 28-second windows, and two-second overlap:

```sh
scripts/run-skainet-audio-benchmark.py plan
scripts/run-skainet-audio-benchmark.py run --serial DEVICE_SERIAL
```

The five pinned fixtures run three times per frontend from fresh processes,
followed by one PCM/mel parity pass per fixture. Resume an interrupted suite
with the same output identifier:

```sh
scripts/run-skainet-audio-benchmark.py run \
  --serial DEVICE_SERIAL \
  --suite-id SUITE_ID \
  --resume
```

This runner builds with `-PisolatedBenchmarkPackage=true` and installs only
`com.example.voicememo.benchmark` plus its instrumentation package. The
isolated target has no enabled launcher. It never installs, uninstalls,
clears, or reads private files from `com.example.voicememo`; public package
metadata is captured before and after the run and must remain identical.
Staged WAVs and device reports are removed only from the benchmark package.

Reports are written under
`build/reports/skainet-audio-benchmark/SUITE_ID/`. They include all raw device
transcripts, per-file and corpus RTF, frontend timing, process CPU time, peak
PSS sampling, thermal snapshots, and WER edit counts. WER is disagreement with
the pinned OpenAI Whisper large-v3-turbo FP32 baseline, not human-ground-truth
WER.

SKaiNET-audio 0.2.1 is published with Kotlin 2.4 metadata while VoiceMemo is
kept on Kotlin 2.2. The artifacts therefore remain runtime-only test
dependencies; a cached reflection bridge calls their public JVM methods. This
uses the published library without changing the production compiler or adding
SKaiNET classes to the production APK.

The default build uses two parallel Whisper interpreters with three
CPU/XNNPACK threads each. It does not create a GPU delegate. Every
transcription writes a content-free benchmark report to app-private storage
and emits the same JSON under the `VoiceMemoBench` logcat tag. The report
includes model-load wall/CPU time, transcription wall/CPU time, real-time
factor, effective lane overlap, PSS memory, thermal status, the requested and
actual backend, and any GPU fallback stage/reason.

Read or monitor reports without clearing app data:

```sh
scripts/collect-transcription-benchmark.sh latest
scripts/collect-transcription-benchmark.sh history
scripts/collect-transcription-benchmark.sh watch
scripts/collect-transcription-benchmark.sh status
```

For an apples-to-apples cold-start CPU comparison, build and run the same memo
once with each configuration. Let the device return to thermal status `NONE`
before each run and keep the same power and screen state:

```sh
# Sequential baseline
./gradlew -PwhisperWorkerCount=1 -PwhisperThreadsPerWorker=4 assembleDebug

# Default parallel configuration
./gradlew -PwhisperWorkerCount=2 -PwhisperThreadsPerWorker=3 assembleDebug
```

## CPU benchmark matrix and UI runner

Only benchmark reports with `"fresh_full_run": true` have a whole-memo
real-time factor; resumed jobs are labeled so they are not compared as full
runs.

The interactive matrix runner automates the build/install/report-collection
parts for `2x2`, `2x3`, `2x4`, `1x4`, and `1x6` while leaving memo selection
and the Transcribe/Retranscribe tap under operator control:

```sh
scripts/run-cpu-benchmark-matrix.sh run
```

It requires the explicit `run` subcommand, performs exactly one transcription
per requested configuration, and never retries a rejected or timed-out run.
Before every in-place `adb install -r`, it checks foreground services,
notifications, and two consecutive matching read-only snapshots of the job
database, then repeats that full check immediately before installation. It
refuses to continue if recording or transcription is active or if the
snapshots disagree. It never uninstalls the app, clears app data, cancels work,
or deletes recordings.

Those checks temporarily copy the complete Room database—not recording audio—
into a user-only host temporary directory. The database can contain memo titles
and transcript text. Each snapshot is removed immediately after its query, and
the parent directory is removed on normal exit, `INT`, or `TERM`; a host crash
or `SIGKILL` can leave the temporary directory behind. This developer-only
tradeoff avoids adding a production database-query endpoint.

Use the exact same existing memo for each prompt. The first accepted report
pins the completed job's opaque memo ID in memory and checks it after every
later run using a read-only database snapshot. It also validates a content-free
fingerprint (audio duration, segment and VAD counts, device, SoC, and Android
version). The memo ID is not written into the benchmark report or summary.
Mismatches, resumed jobs, failed jobs, and runs that did not start at thermal
status `NONE` are rejected without an automatic rerun. Timestamped results are
written under `benchmark-results/`, including one JSON report per
configuration plus `summary.csv` and `summary.md`. Use `--output`, `--serial`,
`--configs`, or `--timeout-minutes` when needed; run
`scripts/run-cpu-benchmark-matrix.sh --help` for details.

For a repeatable end-to-end UI run, use the host-side UI Automator script:

```sh
scripts/run-transcription-ui-test.sh run \
  --memo-id EXISTING_MEMO_ID \
  --install-apk app/build/outputs/apk/debug/app-debug.apk
```

It selects Compose controls by stable resource IDs, starts a fresh
Transcribe/Transcribe again job, polls its durable Room state, collects the
content-free benchmark, and verifies that memo count, audio bytes, and edited
text did not change. Add `--reference-db` and `--reference-job` to compare every
persisted chunk token, language/truncation metadata, raw chunk text, and final
generated text against a read-only CPU reference. The report contains only
counts, hashes, and mismatch positions; it does not print transcript content.
Use `--save-result-db` on a CPU run to create that reference snapshot in a
private temporary directory; unlike the report, this opt-in snapshot contains
memo metadata and transcript text and should be deleted after comparison.
Like the matrix runner, it only installs with `adb install -r` and never
uninstalls, clears data, deletes recordings, or cancels active work.

The matrix is intentionally a cold-process comparison: every measured run
follows an in-place APK install and reports model-load time separately. For a
steady-state experiment, manually run a warm-up and measured transcription
before moving to the next installed configuration. The matrix tool never adds
hidden warm-ups or repeats a run on its own.

The Gradle defaults remain `2x3`. The device keeps the final benchmark build
installed after the matrix so the tool does not perform a hidden extra build
or install; explicitly rebuild and install the default configuration in place
when the comparison is finished.

For GPU model implementation, parity findings, and diagnostic artifacts, see [Whisper GPU diagnostics](whisper-gpu-diagnostics.md).

For continuous CPU/GPU/TPU/skin/battery temperature capture and the deferred
audio-sample, throttling-state, and time-to-throttle experiments, see
[Android thermal benchmark protocols](thermal-benchmark-protocols.md).
