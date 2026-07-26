# Offline Whisper assets

The APK bundles one model and its matching processor metadata. These files are
generated inputs, not runtime downloads.

| Artifact | Upstream revision | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `whisper_large_v3_turbo_30s_i4.tflite` | `litert-community/whisper-large-v3-turbo@4e3af6a1597a54b6e47d2899cf4f6c39932b0b7e` | 755273648 | `da3c91fcd149174cbb5abd3a5583ea95982c5e401c2d68cabac89117f5ce1a4c` |
| `tokenizer.json` | `openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | 2710337 | `297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd` |
| `generation_config.json` | same OpenAI revision | 3772 | `cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c` |
| `preprocessor_config.json` | same OpenAI revision | 340 | `7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711` |

Run `scripts/fetch-whisper-assets.sh` to populate
`app/src/main/generatedAssets/whisper`, then
`scripts/verify-whisper-assets.sh` to audit an existing checkout. The app has
no `INTERNET` permission and does not contain a downloader.
