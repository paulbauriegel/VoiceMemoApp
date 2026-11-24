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

## Speaker segmentation model (pyannote segmentation-3.0)

This benchmark checks whether pyannote segmentation-3.0 (MIT, speaker activity
per 17 ms frame over 10-second windows) gives the same output on the phone as
on the host, and how fast it runs there. It is a model benchmark only; nothing
is wired into transcription.

Convert the model first. The source is the sherpa-onnx ONNX export; its weights
are copied into a rebuilt network, and each step refuses to continue when the
output differs from ONNX Runtime:

```sh
build/speaker-spike/venv/bin/python scripts/export-pyannote-segmentation-litert.py extract
LITERT_EXPORT_VENV/bin/python scripts/export-pyannote-segmentation-litert.py convert
build/venvs/original-fp16/bin/python scripts/export-pyannote-segmentation-litert.py convert --backend keras
```

- `segmentation.tflite` (litert-torch, 19.7 MB): the four bidirectional LSTM
  layers are unrolled into about 75,000 operations.
- `segmentation-fused.tflite` (TensorFlow/Keras, 6.0 MB): fused sequence-LSTM
  operations. Two things are required for correct output. The converter's
  hard-coded `cell_clip=10` is patched to 0, because this model exceeds it.
  And the fused operations keep their hidden state between runs, so the caller
  must call `Interpreter.resetVariableTensors()` before every window.

Then run on the phone:

```sh
build/speaker-spike/venv/bin/python scripts/run-pyannote-segmentation-benchmark.py --serial DEVICE_SERIAL
```

The runner uses the isolated `com.example.voicememo.benchmark` package with the
same guarantees as the SKaiNET benchmark above, and stages only the five public
fixtures. It runs ONNX Runtime, the unrolled LiteRT model (`CompiledModel`) and
the fused LiteRT model (`Interpreter`) at 1, 2 and 4 threads from fresh
processes, then a parity pass on every fixture. Each run is compared with ONNX
Runtime on the host: maximum absolute difference of the log-probabilities and
the share of frames with the same most likely class (pass: at most 0.01 and at
least 99.9%). Reports go to
`build/reports/pyannote-segmentation-benchmark/SUITE_ID/`.

Result on a Pixel 7 Pro (2026-11-19, Android 17, 170-second fixture, 2.5 s hop,
median per 10-second window, no thermal throttling). All 21 runs matched the
host on 100% of frames; the largest numeric difference was 1.2e-3.

| Runtime | Load | 1 thread | 2 threads | 4 threads | 60 min recording, best |
| --- | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime 1.26 (5.7 MB) | 0.07 s | 107 ms | 62 ms | 146 ms | 1.5 min |
| LiteRT unrolled, `CompiledModel` | 41 s | 234 ms | 165 ms | 188 ms | 4.0 min |
| LiteRT fused, `Interpreter` | 0.03 s | 225 ms | 225 ms | 225 ms | 5.4 min |

ONNX Runtime with 2 threads is the fastest by 2.6x. Four threads are slower
because the work spills onto the small cores. The unrolled LiteRT model needs
41 seconds to load, and `CompiledModel` cannot load the fused model at all.

Diagnostics at 2 threads: without `resetVariableTensors()` the fused model
agrees with the host on only 86.9% of frames, which confirms the reset is
required on the device too. The unrolled model through `Interpreter` loads in
2 seconds instead of 41 but runs at 225 ms per window.

GPU (`--gpu-only`, unrolled model, cold load without program cache): 16-bit
precision takes 340 ms per window after a 105-second load and is wrong (94.7%
of frames agree, max difference 8.9); 32-bit precision is correct but takes
378 ms after a 101-second load. The fused model does not load on the GPU. The
GPU is therefore slower than every CPU variant for this model.

## Speaker-label engines (production classes)

`scripts/run-speaker-chain-benchmark.py` runs the app's own `PyannoteSegmenter`
and `CamPlusPlusEmbedder` (ONNX Runtime, bundled assets from
`scripts/fetch-speaker-assets.sh`, Kotlin `KaldiFbank`) in the isolated
benchmark package on the five public fixtures and compares them with ONNX
Runtime plus kaldi-native-fbank on the host.

Result on a Pixel 7 Pro (2026-11-19, no thermal throttling). All seven runs
matched the host: segmentation on 100% of frames, voiceprints with cosine
similarity 1.000000 on every window. Both models load in about 60 ms.

| Threads | Segmentation, 10 s window | Voiceprint, 1.5 s | 3 s | 5 s | 10 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 108 ms | 251 ms | 507 ms | 838 ms | 1721 ms |
| 2 | 62 ms | 139 ms | 274 ms | 457 ms | 918 ms |
| 4 | 145 ms | 299 ms | 568 ms | 928 ms | 1619 ms |

Two threads are fastest for both. The voice model costs about 92 ms per second
of audio, so it dominates: for a 60-minute recording, segmentation takes 1.5
minutes and voiceprints up to 5.5 minutes if every second of audio is embedded
once.

WeSpeaker CAM++ as an alternative voice model (`--voice-model PATH`, staged,
not bundled), same phone and day, 2 threads: 68 ms for 1.5 s, 122 ms for 3 s,
197 ms for 5 s, 390 ms for 10 s. That is about 2.3x faster than ResNet34-LM,
or up to 2.4 minutes for a 60-minute recording.

**ONNX Runtime 1.26 computes WeSpeaker CAM++ wrongly when graph optimisation
is on** (any level above `NO_OPT`): voiceprints have a mean cosine of 0.08
against the reference. Reproduced on the host: 1.26 optimised is wrong, 1.26
with optimisation off and 1.32 at any level agree. With optimisation off all
seven phone runs match the host (cosine 1.000000) at the speed above. pyannote
and ResNet34-LM are not affected. `CamPlusPlusEmbedder` always creates its
session with `NO_OPT`.

The table above is ResNet34-LM, measured with `--voice-model` semantics. The
app bundled WeSpeaker CAM++ from 2026-11-19 to 2026-11-21 and bundles
3D-Speaker CAM++ since (next section).

### Bundled voice model: 3D-Speaker CAM++ (Chinese and English)

`speech_campplus_sv_zh_en_16k-common_advanced` (sherpa-onnx ONNX export, 27 MB,
192-number voiceprints, samples in -1..1, input `x`, output `embedding`).

Why it replaced WeSpeaker CAM++ (host, hand-marked 20-minute two-voice
recording, 14 solo recordings as the profile; one recording, so differences of
a point are within noise):

| Voice model | Labelling chain, balanced | Spurious speaker changes | 1.5 s voiceprint vs profile, error rate |
| --- | ---: | ---: | ---: |
| WeSpeaker CAM++ (VoxCeleb) | 95.4% | 60 | 6.2% |
| 3D-Speaker CAM++ zh+en, padded (bundled) | 96.2% | 10 | 3.0% |
| 3D-Speaker CAM++ zh+en, unpadded, ONNX Runtime 1.32 | 95.7% | 18 | 2.1% |
| 3D-Speaker CAM++ zh+en, unpadded, ONNX Runtime 1.26 | 95.6% | 22 | 4.0% |
| w2v-BERT 2.0, TidyVoice 2026 checkpoint (580M parameters) | 95.6% | 40 | 2.1% |

w2v-BERT 2.0 (the backbone of the TidyVoice 2026 systems) is no better in the
chain, 21 to 33 times slower than CAM++ on a laptop CPU, 1.1 GB at 16 bit, and
its speaker-trained weights are CC BY-NC-SA. The published backbone without
speaker training is unusable (12% and worse).

**ONNX Runtime 1.26 averages a partial pooling segment differently from
1.32.** CAM++ pools over segments of 100 frames after 2x subsampling
(`AveragePool`, kernel 100, `ceil_mode`). For this export the two versions
disagree whenever the last segment is partial, at every optimisation level:
1.5-second voiceprints have a cosine as low as 0.39 between them, 4-second and
10-second ones agree. The phone matches host 1.26 exactly. `CamPlusPlusEmbedder`
therefore repeats the feature frames from the start up to the next multiple of
200 (`FRAME_PADDING`); padded, 1.26 and 1.32 give the same voiceprints.

Result on a Pixel 7 Pro (2026-11-21, Android 17, bundled model with padding,
no thermal throttling). All seven runs matched the host (segmentation 100% of
frames, voiceprint cosine 1.000000).

| Threads | Segmentation, 10 s window | Voiceprint, 1.5 s | 3 s | 5 s | 10 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 107 ms | 114 ms | 223 ms | 339 ms | 584 ms |
| 2 | 62 ms | 77 ms | 144 ms | 218 ms | 382 ms |
| 4 | 145 ms | 138 ms | 238 ms | 368 ms | 634 ms |

The padding makes voiceprints up to 5 s 12 to 18% slower than WeSpeaker CAM++;
10 s is the same. Without the padding the phone run fails parity (minimum
cosine 0.39 to 0.92 per fixture) at 65 ms for 1.5 s.

The sherpa-onnx release re-uploaded the model file during this work: same
weights and graph, only the exporter version string differs
(SHA-256 `aa3cfc16…` before, `53cab20b…` now). The pinned file is the current
one.

To benchmark another voice model: `--voice-model PATH --voice-sample-scale N
--voice-frame-padding N` (WeSpeaker: 32768 and 0).

Voiceprints of the two models cannot be compared. Voice samples and analyses
carry the model id: recorded voice samples are re-embedded from their audio at
the next start, "this is me" samples when their recording is analysed again,
and recordings analysed with the earlier model keep their labels until then.

### Labelling chain end to end

`scripts/run-speaker-label-benchmark.py` runs `SpeakerLabelIndex` (analyse,
pause and resume, enrol, label) against an in-memory database in the isolated
package and compares the 10 ms me/other timeline with a numpy/scipy reference
of the same chain. The "profile" is the first voiceprint found, on both sides,
so this checks phone against host, not accuracy.

Chain settings: 10-second windows every 2.5 s, at most 4 s of each
window-local speaker through the voice model, adaptation from those same
voiceprints. On the hand-marked recording this scored 95.5% balanced with
WeSpeaker CAM++ (all 19 and 13 long turns found), the same as the spike's
chain, which embedded about 4.6x the recording length instead of 1.8x.

Result on a Pixel 7 Pro (2026-11-21, bundled 3D-Speaker CAM++; the 2026-11-19
run with WeSpeaker CAM++ gave the same picture): identical timelines on all
three fixtures (100% of frames, same window and voiceprint counts), pause and
resume gives the same result as an uninterrupted run, analysis takes 5.5 to
5.9 minutes per hour of audio, and re-labelling from stored voiceprints takes
0.06 to 0.14 seconds.

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
