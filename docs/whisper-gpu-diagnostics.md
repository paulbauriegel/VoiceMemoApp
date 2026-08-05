# Whisper GPU implementation and diagnostics

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
- A compact standalone regression now reproduces that projection without
  loading the 1.1 GB decoder:

  ```sh
  scripts/run-whisper-int8-fc-regression.sh --serial DEVICE_SERIAL
  ```

  It derives a fixed `[1,128,1280]` step-7 activation from the pinned model,
  verifies its SHA-256, packages only the 1.65 MB K-projection graph, and runs
  CPU plus six OpenCL configurations. On the Pixel 7 Pro, direct INT8 FC
  measured `0.00597990` active-row maximum error; every source-quantized
  configuration measured `0.03858685`, independent of default, buffer,
  texture, or weight-storage preference. The generated ML Drift caches expose
  three binding families: direct dequantize-to-float FC, source-quantized
  IMAGE2D FC/Conv, and source-quantized BUFFER FC/Conv. The Maven accelerator
  strips the implementation RTTI, so `ConvWaveMatrixMali` versus
  `ConvWaveMatrix`/`ConvWaveMemory` is deliberately reported as unknown until
  the source-built diagnostic logger is available.
- A second compact arithmetic matrix separates FC accumulation from dynamic
  activation quantization:

  ```sh
  scripts/run-whisper-int8-fc-arithmetic-matrix.sh --serial DEVICE_SERIAL
  ```

  It tests seven pinned per-channel, per-tensor, explicit-zero-point,
  synthetic, and FLOAT32 models against real and controlled activations. On
  the Pixel 7 Pro, closed-form diagonal and accumulation oracles measured
  maximum errors of `0.0125490` for CPU hybrid quantization, `0.00004935` for
  direct GPU FC, and `0.0250592` for source-quantized GPU FC. The direct GPU
  accumulator is therefore mathematically sound. The source-quantized error
  is proportional to the quantized weight-row sum, and its normalized slope is
  `1/510`: exactly half of one 8-bit activation quantization step. The remaining
  parity target is ML Drift's `quantize_float32_to_int8w4c4` scale/offset
  contract, not the FC dot-product kernel or weight zero-point metadata.
- Dequantizing only the first K matrix reduces its step-7 active-row maximum
  from 0.00598037 to 3.58e-7. Dequantizing first-block K/Q/V/output matrices
  similarly keeps Q and V near 1e-7 and the attention score BMM at 1.43e-6.
  This removes the CPU/GPU hybrid-quantization contract mismatch; the
  arithmetic matrix above shows that the underlying direct GPU FC accumulator
  itself is not faulty.
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

## Original-model FP16 Android GPU experiment

`scripts/run-original-fp16-gpu-experiment.py` provides a fail-closed,
developer-only comparison against the pinned original OpenAI
`whisper-large-v3-turbo` revision
`41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. It:

1. runs the original PyTorch checkpoint on the host CPU in FLOAT32;
2. exports fixed `encode` and `decode` signatures;
3. converts constants to FLOAT16 while retaining FLOAT32
   inputs, activations, and outputs;
4. applies the explicit attention-key transpose and causal-mask rank-barrier
   fixes, with SHA-256 verification between every artifact; and
5. streams the candidate and WAV into app-private diagnostic directories,
   runs the whole-memo Android GPU benchmark, and reports text, tokens, WER,
   token error rate, timing, and memory.

Use Python 3.11 and allow at least 18 GiB of free disk space:

```sh
python3.11 -m venv build/venvs/original-fp16
build/venvs/original-fp16/bin/pip install \
  -r scripts/requirements-original-fp16.txt
build/venvs/original-fp16/bin/python \
  scripts/run-original-fp16-gpu-experiment.py all \
  --memo-id latest \
  --language de \
  --serial DEVICE_SERIAL
```

For inspection or recovery, the same script exposes `baseline`, `convert`,
`rewrite`, and `device` stages. `--local-files-only` prevents Hugging Face
downloads when the pinned checkpoint is already cached. `--skip-build` reuses
existing debug benchmark APKs. Pass `--audio /absolute/path/to/memo.wav`
instead of `--memo-id latest` to use a host PCM16 WAV.

The experiment never edits the bundled production asset. Android deployment
uses `adb install -r`; it never uninstalls VoiceMemo or clears app data.
Staged candidate files are removed after the run, while the host-side models
and JSON reports remain under `build/experiments/original-fp16`.

## Decoder-split diagnostic artifacts

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

For the broader benchmark runners and comparison methodology, see [Transcription benchmarks](transcription-benchmarks.md).
