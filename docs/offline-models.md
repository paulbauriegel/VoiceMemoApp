# Offline models

The APK bundles two selectable arm64 LiteRT transcription models. Whisper
remains the default and existing baseline:

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
scripts/fetch-parakeet-assets.sh
scripts/fetch-silero-vad-assets.sh
```

The generated files live under
`app/src/main/generatedAssets/{whisper,parakeet}/` and are intentionally
ignored by Git.
The Android build independently checks every required file's exact size and
SHA-256. The `.tflite` entry is packaged without compression so LiteRT can
memory-map it from the standalone APK.

Parakeet uses the stateful five-second INT8 LiteRT graph:

- Conversion repository: `litert-community/parakeet-tdt-0.6b-v3`
- Conversion revision: `e3a6f2dec6800733f97c87ff55822f32c405983a`
- Upstream/tokenizer revision: `7c35754d166cca382ad1e53e68b01e7c575f3a1d`
- Model: `parakeet_tdt_0.6b_v3_5s_i8_stateful.tflite`
- Size: `614261072` bytes
- SHA-256:
  `334745b8bc7fd372b1c213516f0b6338bb827b1a2abb3e77ad35fe6fea5cd16b`

Its frontend and stateful TDT decoder follow Google's LiteRT speech-recognition
sample. Long memos use five-second windows with 50% overlap. CPU and GPU use
the same graph; GPU requests FP32 arithmetic and constant-tensor sharing with
CPU fallback for unsupported operations.

Voice activity detection uses the separately bundled Silero VAD v6.2.1 ONNX
model pinned at revision `7e30209a3e901f9842f81b225f3e93d8199902b1`
(2,327,524 bytes, SHA-256
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`).
It runs locally through ONNX Runtime on one CPU thread and never changes or
trims the audio. The app fails open to Whisper if VAD inference fails. Because
ONNX Runtime Android supports API 24 and later, Voice Memo's minimum Android
version is 7.0.

For build instructions, see the [project README](../README.md).
