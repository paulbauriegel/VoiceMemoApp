#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/build/model-assets/qwen"
MODEL_REVISION="dd8ed8ba3c32bccfe0a76c0cc81e9b7cf4f2ad81"
TOKENIZER_REVISION="7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_FILE="${OUTPUT_DIR}/qwen3_asr_0.6b_5s_i8.tflite"
TOKENIZER_FILE="${OUTPUT_DIR}/tokenizer.json"
MODEL_SHA256="952aab6a43f1fc1a76537e72852ea98e12ed407087eb6835bf15a96d4ad4c385"
TOKENIZER_SHA256="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"

verify() {
  local file="$1"
  local bytes="$2"
  local sha256="$3"
  [[ -f "${file}" ]] || return 1
  [[ "$(stat -f %z "${file}")" == "${bytes}" ]] || return 1
  [[ "$(shasum -a 256 "${file}" | awk '{print $1}')" == "${sha256}" ]]
}

download() {
  local url="$1"
  local destination="$2"
  local bytes="$3"
  local sha256="$4"
  if verify "${destination}" "${bytes}" "${sha256}"; then
    echo "Verified: ${destination}"
    return
  fi
  mkdir -p "${OUTPUT_DIR}"
  curl --fail --location --retry 3 --output "${destination}.partial" "${url}"
  mv "${destination}.partial" "${destination}"
  verify "${destination}" "${bytes}" "${sha256}" || {
    echo "Downloaded asset failed size or SHA-256 verification: ${destination}" >&2
    exit 1
  }
}

download \
  "https://huggingface.co/litert-community/Qwen3-ASR-0.6B/resolve/${MODEL_REVISION}/qwen3_asr_0.6b_5s_i8.tflite" \
  "${MODEL_FILE}" \
  "793931296" \
  "${MODEL_SHA256}"
download \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/${TOKENIZER_REVISION}/tokenizer.json" \
  "${TOKENIZER_FILE}" \
  "7031645" \
  "${TOKENIZER_SHA256}"
