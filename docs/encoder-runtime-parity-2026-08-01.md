# Encoder runtime parity on Pixel 7 Pro

Date: 2026-08-01

This diagnostic compares encoder implementations only. Every runtime receives the same app-generated `[1, 128, 3000]` log-mel tensor (`SHA-256 5d3d4ad327c24876d55298e0c3706536e3cfa04efd7c626c192ae7d88101cd36`). The decoder is never executed. LiteRT uses the original OpenAI FP32 encoder model; whisper.cpp uses the pinned `large-v3-turbo` model with FP16 weights.

The numerical reference is LiteRT CPU FP32. Times below are single diagnostic executions and are included for orientation, not as a thermally controlled performance benchmark.

## Final encoder state

| Runtime | Encoder time | RMSE vs CPU FP32 | MAE | Cosine similarity |
|---|---:|---:|---:|---:|
| LiteRT CPU FP32 | 114.140 s | 0 | 0 | 1 |
| LiteRT OpenCL FP32 | 10.461 s | 0.0000208 | 0.00000667 | 0.999999999 |
| LiteRT OpenCL native FP16 | 4.388 s | 0.112687 | 0.068810 | 0.967675 |
| LiteRT OpenCL strict-math FP16 | 5.797 s | 0.111379 | 0.067806 | 0.968414 |
| LiteRT FP16 storage, FP32 accumulation | 5.604 s | 0.104449 | 0.065803 | 0.972265 |
| whisper.cpp CPU, FP16 weights | 240.901 s | 0.023269 | 0.007149 | 0.998598 |
| whisper.cpp Vulkan, FP16 weights | 35.825 s | 0.060240 | 0.025944 | 0.990603 |

The mixed LiteRT mode reduces final RMSE by about 7.3% relative to normal native FP16, while strict compiler math reduces it by only about 1.2%. Neither closes the gap to whisper.cpp Vulkan.

## Encoder checkpoints

Each cell is RMSE against LiteRT CPU FP32 at the same boundary.

| Boundary | LiteRT strict FP16 | LiteRT FP16 + FP32 accumulation | whisper.cpp Vulkan |
|---|---:|---:|---:|
| Convolution stem | 0.004416 | 0.000308 | 0.003266 |
| Block 0 | 0.010665 | 0.001160 | 0.010553 |
| Block 3 | 0.012705 | 0.002951 | 0.014268 |
| Block 7 | 0.009524 | 0.003690 | 0.010381 |
| Block 15 | 0.005254 | 0.001786 | 0.005096 |
| Block 23 | 0.027409 | 0.020756 | 0.036149 |
| Block 31 | 0.088565 | 0.075099 | 0.063574 |
| Final layer-normalized output | 0.111379 | 0.104449 | 0.060240 |

## Interpretation

- LiteRT OpenCL FP32 reproduces CPU FP32 almost exactly, so the converted encoder graph, transpose fixes, mask behavior, and mel input are sound.
- LiteRT mixed accumulation is nearly exact through block 15. Its remaining error grows in the late transformer stack, especially between blocks 23 and 31, and is amplified further by the final layer normalization.
- Disabling relaxed OpenCL math does not materially fix the result, so compiler fast-math is not the primary cause.
- whisper.cpp's FP16 model file does not imply full-FP16 execution for every activation and reduction. Its Vulkan result is substantially closer to the FP32 reference than LiteRT full FP16.
- The next useful isolation is per-operation capture inside blocks 24-31, concentrating on layer normalization/reductions, attention softmax, residual additions, and matrix multiplication output precision. Token-level comparison can be added later by feeding these saved encoder states to one fixed FP32 decoder, but it is not needed to show that the present divergence originates in the encoder.

The first attempted whisper.cpp block capture was discarded because GGML buffer reuse made multiple checkpoints identical. The values in this report come from the corrected evaluation callback, which copies each tensor before reuse; all checkpoint hashes are distinct.

## Reproduction artifacts

- Standard final-state report: `build/reports/encoder-runtime-parity/encoder-parity-standard-reference/summary.json`
- Strict-math final-state report: `build/reports/encoder-runtime-parity/encoder-parity-strict-math-fp16-reference/summary.json`
- Corrected whisper.cpp checkpoint report: `build/reports/encoder-runtime-parity/encoder-parity-whispercpp-corrected-blocks/summary.json`
- Mixed-accumulation final-state report: `build/reports/encoder-runtime-parity/encoder-parity-fp16-fp32-accum-reference/summary.json`
- Mixed-accumulation checkpoint report: `build/reports/encoder-runtime-parity/encoder-parity-fp16-fp32-accum-blocks/summary.json`
- Test harness: `app/src/androidTest/java/com/example/voicememo/transcription/EncoderRuntimeParityDiagnosticTest.kt`
- Runner: `scripts/run-encoder-runtime-parity.py`

## Precision-localization diagnostics

The follow-up suite keeps the decoder out of the experiment and provides three levels of narrowing:

1. `late-blocks` captures residual outputs for every block from 23 through 31 in both LiteRT and whisper.cpp Vulkan.
2. `stage` creates a temporary dependency-pruned encoder model for one operation boundary in blocks 24-31. It compares LiteRT CPU FP32, OpenCL FP32, and OpenCL FP16, then deletes the temporary multi-gigabyte model after staging/running it.
3. `final-layernorm` cross-feeds saved CPU and FP16 block-31 states through a standalone 12.9 KiB copy of the final encoder LayerNorm. It separately reports GPU-LayerNorm-only error and propagated block-31 error after CPU LayerNorm.

Build/install the diagnostic-enabled app and test APKs before running these commands. Each runner waits for Android thermal status 0 and preserves app data.

```bash
python3 scripts/run-encoder-precision-localization.py late-blocks

python3 scripts/run-encoder-precision-localization.py \
  stage --block 24 --stage attn_ln_variance

python3 scripts/run-encoder-precision-localization.py \
  final-layernorm --late-block-report \
  build/reports/encoder-precision-localization/<completed-late-block-run>
```

Available stage names are `input`, `attn_ln_mean`, `attn_ln_variance`, `attn_ln_rsqrt`, `attn_ln_output`, `k_projection`, `q_projection`, `qk_logits`, `softmax`, `v_projection`, `attention_value`, `attention_projection`, `attention_residual`, `ffn_ln_mean`, `ffn_ln_variance`, `ffn_ln_rsqrt`, `ffn_ln_output`, `fc1`, `gelu`, `fc2`, and `block_output`.

Run `late-blocks` first. Then run stage probes only for the first block whose local RMSE growth is materially larger than the preceding blocks. Very large tensors such as `qk_logits` and `softmax` should be tested one at a time.

## 2026-08-03 localization result

All runs below started and ended at Android thermal status 0. They used the original OpenAI FP32 encoder graph, the same saved mel input, LiteRT CPU FP32 as the reference, LiteRT OpenCL native FP16 as the candidate, and no decoder.

The late-block capture shows that the largest residual-state jump occurs in block 31:

| Boundary | Native FP16 RMSE vs CPU FP32 | MAE | Cosine similarity |
|---|---:|---:|---:|
| Block 29 | 0.042129 | 0.013530 | 0.999218 |
| Block 30 | 0.053403 | 0.020549 | 0.998756 |
| Block 31 | 0.085578 | 0.033928 | 0.996897 |
| Final encoder output | 0.112687 | 0.068810 | 0.967675 |

Cross-feeding the saved block-31 states through the standalone final LayerNorm separates propagated error from LayerNorm compute error:

| Comparison | RMSE | MAE | Cosine similarity |
|---|---:|---:|---:|
| FP16 block-31 state through CPU FP32 LayerNorm vs CPU reference | 0.112112 | 0.068680 | 0.968017 |
| CPU block-31 state through GPU FP16 LayerNorm vs CPU reference | 0.011451 | 0.000643 | 0.999660 |

Both standalone CPU and GPU LayerNorm executions reproduce their corresponding full-encoder outputs bit-for-bit. The final LayerNorm is therefore implemented consistently; most final-state divergence has already entered the block-31 input to that LayerNorm.

Inside block 31, attention is not the main source. The attention residual is RMSE 0.055524, only 0.002121 above the block-30 output. The first sharp directional loss occurs in the feed-forward LayerNorm:

| Block-31 checkpoint | Native FP16 RMSE | MAE | Cosine similarity |
|---|---:|---:|---:|
| Attention residual | 0.055524 | 0.023873 | 0.998654 |
| FFN LayerNorm output | 0.177349 | 0.119938 | 0.958746 |
| GELU output | 0.040438 | 0.013987 | 0.963111 |
| FC2 output | 0.050173 | 0.020666 | 0.939127 |
| Block output | 0.085578 | 0.033928 | 0.996897 |

The LayerNorm math probe identifies the variance reduction as a concrete failure point:

| Block-31 FFN LayerNorm tensor | Finite pairs | RMSE | MAE | Cosine similarity |
|---|---:|---:|---:|---:|
| Mean | 1500 / 1500 | 0.002907 | 0.002714 | 0.999997 |
| Variance | 1494 / 1500 | 0.190939 | 0.007633 | 0.991751 |
| Reciprocal square root | 1500 / 1500 | 0.631937 | 0.228895 | 0.997897 |
| Normalized output | 1920000 / 1920000 | 0.177349 | 0.119938 | 0.958746 |

For six frames, the FP16 variance is positive infinity while CPU FP32 produces finite values from 199.69 through 217.28. Their subsequent FP16 reciprocal square root is exactly zero. The graph implements variance as a `MEAN` over a `[1, 1500, 1280]` squared-difference tensor. A straightforward FP16 sum before division can overflow even when the final mean is only about 200, because `1280 * 200` is well above the maximum finite FP16 value. Other low-variance frames also show large relative errors; for example, frame 381 is `0.418076` on CPU and `0.006634` on FP16, which then changes reciprocal square root from `1.54656` to `12.265625`.

This is stronger evidence than a general “FP16 is less accurate” explanation: LiteRT OpenCL's native-FP16 reduction and/or intermediate policy is numerically unsuitable for this LayerNorm. The next fix experiment should keep LayerNorm mean, squared-difference mean, and reciprocal-square-root math in FP32 while leaving the surrounding matrix multiplications and storage in FP16. Merely disabling relaxed compiler math is not expected to solve it, matching the already measured strict-math result.

The failure was reproduced independently of Whisper with a 13 KB LayerNorm-only model and six byte-identical finite input rows. CPU FP32 returned variances from `196.971939` through `216.049103`; OpenCL FP16 returned `+Infinity` for all six. The upstream-ready report and 32 KB attachment bundle are in `repro/litert-fp16-layernorm-overflow/`.

New reports:

- `build/reports/encoder-precision-localization/encoder-late-blocks-standard-streaming-20260803/summary.json`
- `build/reports/encoder-precision-localization/encoder-final-layernorm-saved-states-v2-20260803/summary.json`
- `build/reports/encoder-precision-localization/encoder-block31-attention-residual-20260803/summary.json`
- `build/reports/encoder-precision-localization/encoder-block31-mlp-checkpoints-20260803/summary.json`
- `build/reports/encoder-precision-localization/encoder-block31-ffn-layernorm-math-20260803/summary.json`
