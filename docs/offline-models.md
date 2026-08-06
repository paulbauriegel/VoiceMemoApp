# Offline models

The APK bundles two selectable arm64 LiteRT transcription models. Whisper
remains the default. Its production GPU configuration uses the corrected
FP16-safe encoder and original-weight decoder:

- Repository: `openai/whisper-large-v3-turbo`
- Revision: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Source checkpoint SHA-256:
  `542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1`
- Encoder: `whisper_large_v3_turbo_split_encoder.tflite`, `2548210772` bytes,
  SHA-256 `14e580c6ce2815a67c8ba0a4db334c8c016b904ecbf464d6f31911191700a87e`
- Decoder: `whisper_large_v3_turbo_split_decoder.tflite`, `951674764` bytes,
  SHA-256 `d661702d7753165cfaf01bf5b060f46065df500959e42e5254fd2598a54dedb7`
- Encoder rewrite: all 65 LayerNorm variance branches scale centered
  activations by `0.25` before squaring and restore variance by `16`, avoiding
  native-FP16 overflow while preserving the intended FP32 calculation.
- Runtime: OpenCL native FP16 for both encoder and decoder, with LiteRT source
  quantization disabled.

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
