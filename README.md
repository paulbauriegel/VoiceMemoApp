# Voice Memo

Voice Memo is a Material 3 Android app for recording, playing, organizing, and
transcribing voice notes entirely on the device. It has no `INTERNET`
permission, cloud account, analytics SDK, or automatic cloud backup. Manual
backup export and restore remain entirely user initiated.

## Offline model

The APK bundles the arm64 LiteRT model and tokenizer:

- Repository: `litert-community/whisper-large-v3-turbo`
- Revision: `4e3af6a1597a54b6e47d2899cf4f6c39932b0b7e`
- Model: `whisper_large_v3_turbo_30s_i8_explicit_key_transpose.tflite`
- Size: `1094571944` bytes
- SHA-256:
  `8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418`
- Source model SHA-256:
  `c40690d926fe9841ef6e0296eb0c4168c2910171c1d67b4bcf00a4b7cd7989ca`
- Rewrite: every Whisper attention-score `BATCH_MATMUL(adj_y=true)` is
  converted to an explicit key `TRANSPOSE([0,2,1])` followed by
  `BATCH_MATMUL(adj_y=false)`. This preserves CPU output exactly and avoids
  LiteRT GPU's broken implicit-adjoint lowering.

Fetch and verify the pinned development assets before building:

```sh
scripts/fetch-whisper-assets.sh
scripts/fetch-silero-vad-assets.sh
```

The generated files live under
`app/src/main/generatedAssets/whisper/` and are intentionally ignored by Git.
The Android build independently checks every required file's exact size and
SHA-256. The `.tflite` entry is packaged without compression so LiteRT can
memory-map it from the standalone APK.

Voice activity detection uses the separately bundled Silero VAD v6.2.1 ONNX
model pinned at revision `7e30209a3e901f9842f81b225f3e93d8199902b1`
(2,327,524 bytes, SHA-256
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`).
It runs locally through ONNX Runtime on one CPU thread and never changes or
trims the audio. The app fails open to Whisper if VAD inference fails. Because
ONNX Runtime Android supports API 24 and later, Voice Memo's minimum Android
version is 7.0.

## Build

Use Android Studio's bundled JBR (Java 17 or newer) and Android SDK 36:

```sh
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
export ANDROID_HOME="$HOME/Library/Android/sdk"
./gradlew testDebugUnitTest lintDebug assembleDebug
```

The app packages only `arm64-v8a`. Because the model is bundled, expect the
standalone sideload APK to exceed 1.09 GB.

## Transcription benchmarks

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

The default build now packages the complete FLOAT32-decoder rewrite and uses
one LiteRT GPU worker with source activation quantization disabled. LiteRT
runs the encoder and almost all decoder neural operations through OpenCL FP32;
the unsupported token `GATHER`, frontend, and greedy token selection stay on
CPU. Model-open or inference failures automatically fall back to
Interpreter/XNNPACK:

```sh
./gradlew assembleDebug
```

To build the former two-worker CPU configuration explicitly:

```sh
./gradlew \
  -PuseFullFloatDecoderProduction=false \
  -PwhisperBackend=cpu \
  -PwhisperWorkerCount=2 \
  -PwhisperThreadsPerWorker=3 \
  assembleDebug
```

The app still has no network permission.

The guarded decoder quantization A/B uses a live CPU oracle, the same CPU
encoder state for both GPU decoders, and every reference token:

```sh
scripts/run-gpu-parity-diagnostic.sh run \
  --memo-id MEMO_ID \
  --reference-job CPU_JOB_ID \
  --decoder-quant-ab
```

On the 14.2-second Pixel 7 Pro memo, disabling
`allowSrcQuantizedFcConvOps` improved teacher-forced top-1 agreement only
from 23/45 to 25/45 tokens. The first mismatch remained token 7, mean logit
RMSE changed from 3.042 to 2.781, and GPU teacher-forced time increased from
9.0 to 20.1 seconds. The whole-memo no-source-quant paths were slower and
worse: full GPU and CPU-encoder/GPU-decoder both measured 141.94% WER,
versus 74.19% and 80.65% respectively with the option enabled. Source
quantization is therefore not the primary decoder correctness fault.

The rewritten INT8 decoder prefix bisection locates the remaining first
material divergence in the first decoder block's quantized-weight
`FULLY_CONNECTED` projections:

| Boundary, decoder step 7 | Source activation quantization | Active-row max absolute error | Cosine |
| --- | --- | ---: | ---: |
| Position add | off | 0 | 1.00000000 |
| Self layer norm | off | 1.49e-8 | 1.00000000 |
| K projection | on | 0.0385867 | 0.99998831 |
| K projection | off | 0.00598049 | 0.99999861 |
| Q projection | off | 0.000370145 | approximately 1 |
| Attention score matmul | off | 0.0200715 | 0.99999541 |
| Attention softmax | off | 0.296314 | 0.9003 |

The implicit `adj_y=true` key path is one separate LiteRT GPU bug and is
already avoided by the explicit key transpose. Disabling
`allowSrcQuantizedFcConvOps` does not fix the next issue. Same-input probes and
targeted FLOAT32 projection rewrites distinguish two additional faults:

- The rewritten `adj_y=false` attention score `BATCH_MATMUL` is bit-exact for
  all 327,680 values at decoder steps 0 and 7.
- Isolated softmax differs only by floating-point ULPs (active maximum
  5.96e-8, cosine approximately 1).
- The exact production INT8 K `FULLY_CONNECTED`, given the same
  CPU-normalized input, still differs by 0.00598037 at step 7. This reproduces
  the full-prefix K divergence.
- Dequantizing only the first K matrix reduces its step-7 active-row maximum
  from 0.00598037 to 3.58e-7. Dequantizing first-block K/Q/V/output matrices
  similarly keeps Q and V near 1e-7 and the attention score BMM at 1.43e-6.
  This confirms a LiteRT GPU per-channel INT8 `FULLY_CONNECTED` correctness
  problem rather than ordinary autoregressive floating-point drift.
- Even with those four FLOAT32 projections, the integrated causal-mask output
  was wrong on valid (unmasked) logits: step-7 maximum 2.81293 and mean
  absolute error 0.42248. Running the identical CPU score tensor and causal
  mask through an isolated GPU broadcast `ADD` is bit-exact, so ADD arithmetic
  and mask construction are not the cause. The failure is the compiled
  BMM-to-broadcast-ADD tensor-layout/fusion path.
- Inserting an explicit rank reshape from `[20,128,128]` to
  `[1,20,128,128]` before the causal-mask ADD keeps the path GPU-delegatable
  and reduces the step-7 valid-logit maximum to 7.15e-7 with cosine 1.0.
  The isolated following softmax remains within 1.19e-7.
- The attention-value BMM is also bit-exact when given identical CPU
  probabilities and values. Its earlier large error was inherited from the
  incorrect integrated mask/softmax input.

Both fixes have now been applied across the complete decoder:

- all 41 rank-2 INT8 `FULLY_CONNECTED` weight tensors in the decoder are
  dequantized to FLOAT32;
- all four causal-mask ADDs receive an explicit
  `[20,128,128] -> [1,20,128,128]` rank reshape.

Generate the pinned artifact with:

```sh
tools/whisper-decoder-split/build/install/whisper-decoder-split/bin/whisper-decoder-split \
  generate-full-float-decoder \
  app/src/main/generatedAssets/whisper/whisper_large_v3_turbo_30s_i8_explicit_key_transpose.tflite \
  build/generated/whisper-full-float-decoder/whisper_large_v3_turbo_full_float_decoder.tflite
```

The resulting model is 1,779,622,096 bytes with SHA-256
`a2613d64e6cb73ff40db7c548215fd086f096aa088c6cc113c833ccc87501e89`.
The default production build packages exactly this artifact, while diagnostic
builds can also place it in the instrumentation APK.

The guarded Pixel 7 Pro whole-memo benchmark for a 14.2-second German memo
measured:

| Backend | Inference | WER versus CPU | Result |
| --- | ---: | ---: | --- |
| CPU 2 x 3 threads | 82.20 s | 0% reference | expected 45 tokens |
| GPU encoder + rewritten GPU decoder | 26.81 s | 12.90% | source-quantized FC/Conv enabled |
| GPU encoder + rewritten GPU decoder | 34.36 s | 0% | source-quantized FC/Conv disabled |
| GPU encoder + CPU decoder | 47.68 s | 12.90% | isolates the source-quantized encoder difference |
| CPU encoder + rewritten GPU decoder | 60.25 s | 3.23% | one substitution: `kackig` versus `kacken` |

This confirms that the FLOAT32-weight and mask-rank fixes remove the major
decoder faults. It also corrects the earlier prefix-only conclusion about
`allowSrcQuantizedFcConvOps`: on the complete rewritten graph, disabling that
GPU compiler path removes the encoder-driven whole-transcript difference and
produces the exact CPU transcript for this memo. The isolated rewritten GPU
decoder still has one borderline autoregressive token difference.
Strict GPU-only compilation also still fails; the working decoder delegates
`GATHER` to CPU, while the frontend and token selection remain CPU work.

A second promotion gate used a dense 2:06 German memo across five overlapping
Whisper windows:

| Backend | Inference | Words | WER versus CPU | Peak PSS |
| --- | ---: | ---: | ---: | ---: |
| CPU, 2 x 3 threads | 315.57 s | 189 | 0% reference | 3.26 GB |
| Full GPU, no source quantization | 217.92 s | 189 | 1.06% | 4.43 GB |

The GPU run completed all five windows without truncation, retry, allocation
failure, or crash. Its two substitutions exceeded the accepted 90% similarity
gate by a wide margin, so this configuration was promoted. One GPU worker is
mandatory because two 4.4 GB model instances would be unsafe.

Reproduce the non-persisting benchmark with:

```sh
scripts/run-gpu-parity-diagnostic.sh whole \
  --memo-id MEMO_ID \
  --full-float-decoder \
  --promotion-gate \
  --serial DEVICE_SERIAL
```

The promotion is evidence for the tested Pixel 7 Pro and German memos, not a
claim of exact multilingual token parity. Continue expanding the fixture/memo
corpus before treating the GPU transcript as a universal CPU oracle.

### Decoder-split diagnostic artifacts

The developer-only decoder-split generator can isolate Whisper's terminal
vocabulary projection without rebuilding the model from TensorFlow sources:

```sh
scripts/generate-whisper-decoder-split.sh self-test
scripts/generate-whisper-decoder-split.sh generate
scripts/generate-whisper-decoder-split.sh verify
```

It verifies the pinned production model's full SHA-256 before reading it, then
writes four ignored files under `build/generated/whisper-decoder-split/`:

- a same-size GPU-core model whose `decode` signature returns the final
  `[1,128,1280]` hidden tensor; and
- a minimal CPU/XNNPACK INT8 projection model that maps only the active
  `[1,1280]` token row to `[1,51866]` logits;
- a second same-size GPU-core model cut immediately before the final layer
  norm, whose `decode` signature returns residual tensor 423 as
  `[1,128,1280]`; and
- a builtin-only CPU/XNNPACK tail that applies the original final layer norm
  and INT8 vocabulary projection to one active `[1,1280]` row. Its
  `normalize_and_project` signature returns `[1,51866]` logits.

The output sizes, hashes, tensor contracts, projection weights, per-channel
scales, zero points, layer-norm epsilon, beta/gamma bytes, operator topology,
and builtin options are verified again after generation and recorded in a
content-free manifest. The Gradle tooling runs offline by default using pinned
cached Java FlatBuffer dependencies; pass `--allow-dependency-download` only
when preparing a new development machine.

These artifacts are not part of the Android source set, do not replace the
production model, and are not selectable by normal transcription. They exist
only for controlled CPU-versus-GPU parity diagnostics.

For transformer-block boundary diagnostics, generate one shared-weight stage
at a time. The first device candidate keeps decoder operators 0–232 in the GPU
prefix and moves operators 233–309 (the final decoder block) to a CPU suffix:

```sh
scripts/generate-whisper-decoder-split.sh generate-boundary \
  --stage after-3-blocks
scripts/generate-whisper-decoder-split.sh verify-boundary \
  --stage after-3-blocks
```

The resulting diagnostic FlatBuffer is only 268 bytes larger than the pinned
model. Its `decode` signature returns residual tensor 346 as
`[1,128,1280]`; `decode_tail_after_3_blocks` consumes that residual plus the
encoder state and compatibility mask, then returns pre-final-layer-norm tensor
423 with the same shape. Both subgraphs point to the exact same original
decoder tensor, operator, and constant tables. The existing
`normalize_and_project` active-row model completes the CPU path without
materializing `[1,128,51866]` logits for every row.

This compact diagnostic repurposes the copied model's encoder subgraph
metadata, so encoding still uses the untouched production model. The generated
manifest records the stage, tensor/operator contracts, size, and hashes.
`after-1-block`, `after-2-blocks`, and the CPU-prepared
`after-position-add` boundary are also structurally defined, but should be
generated only when needed to avoid retaining several 755 MB diagnostics.

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

## Privacy and storage

In normal app use, audio, waveform sidecars, transcript versions, and resumable
transcription state stay in app-private storage. The developer CPU matrix
exception is disclosed above. Android's sharesheet receives only explicit,
temporary `FileProvider` grants. App backup is disabled, and the manifest
deliberately omits `android.permission.INTERNET`.

OpenAI Whisper's MIT notice and LiteRT's Apache-2.0 notice are available in the
in-app Licenses screen.
