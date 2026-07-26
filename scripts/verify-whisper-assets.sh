#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSET_DIR="${1:-${PROJECT_DIR}/app/src/main/generatedAssets/whisper}"

declare -a FILES=(
  "whisper_large_v3_turbo_30s_i8_explicit_key_transpose.tflite|1094571944|8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418"
  "tokenizer.json|2710337|297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"
  "generation_config.json|3772|cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c"
  "preprocessor_config.json|340|7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"
)

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

size_of() {
  if stat -f '%z' "$1" >/dev/null 2>&1; then
    stat -f '%z' "$1"
  else
    stat -c '%s' "$1"
  fi
}

for item in "${FILES[@]}"; do
  IFS='|' read -r name expected_size expected_sha <<<"${item}"
  path="${ASSET_DIR}/${name}"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required offline model asset: ${path}" >&2
    echo "Run scripts/fetch-whisper-assets.sh before building." >&2
    exit 1
  fi
  actual_size="$(size_of "${path}")"
  actual_sha="$(sha256_of "${path}")"
  if [[ "${actual_size}" != "${expected_size}" || "${actual_sha}" != "${expected_sha}" ]]; then
    echo "Offline model asset verification failed: ${path}" >&2
    echo "Expected size/SHA: ${expected_size} ${expected_sha}" >&2
    echo "Actual size/SHA:   ${actual_size} ${actual_sha}" >&2
    exit 1
  fi
  echo "Verified ${name}"
done
