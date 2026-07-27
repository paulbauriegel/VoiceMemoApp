#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DESTINATION="${1:-${PROJECT_DIR}/app/src/main/generatedAssets/parakeet}"
MODEL_REPOSITORY="litert-community/parakeet-tdt-0.6b-v3"
MODEL_REVISION="e3a6f2dec6800733f97c87ff55822f32c405983a"
TOKENIZER_REPOSITORY="nvidia/parakeet-tdt-0.6b-v3"
TOKENIZER_REVISION="7c35754d166cca382ad1e53e68b01e7c575f3a1d"

mkdir -p "${DESTINATION}"

download() {
  local repository="$1"
  local revision="$2"
  local name="$3"
  local target="${DESTINATION}/${name}"
  local partial="${target}.part"
  curl --location --fail --show-error --retry 5 --retry-all-errors \
    --continue-at - --output "${partial}" \
    "https://huggingface.co/${repository}/resolve/${revision}/${name}?download=true"
  mv "${partial}" "${target}"
}

download "${MODEL_REPOSITORY}" "${MODEL_REVISION}" \
  "parakeet_tdt_0.6b_v3_5s_i8_stateful.tflite"
download "${TOKENIZER_REPOSITORY}" "${TOKENIZER_REVISION}" "tokenizer.json"
download "${TOKENIZER_REPOSITORY}" "${TOKENIZER_REVISION}" "processor_config.json"
download "${TOKENIZER_REPOSITORY}" "${TOKENIZER_REVISION}" "config.json"
"${SCRIPT_DIR}/verify-parakeet-assets.sh" "${DESTINATION}"
