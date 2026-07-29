#!/usr/bin/env bash
set -euo pipefail

# Downloads the exact, reviewed LiteRT-LM model used to generate recording
# titles and summaries. The Android app itself remains fully offline and has no
# INTERNET permission.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DESTINATION="${1:-${PROJECT_DIR}/app/src/main/generatedAssets/granite}"
REPOSITORY="litert-community/granite-4.0-350m-litert-lm"
REVISION="6f1e9bce89b174930a79de82d0dcdede708f8c34"
MODEL_NAME="granite-4.0-350m_q8_ekv1280.litertlm"
MODEL_SIZE="468209584"
MODEL_SHA256="c8e9a29493f62b7c44461fb36980987c4c1454c75e95f57ba0539a8edc9dce76"

mkdir -p "${DESTINATION}"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "Neither sha256sum nor shasum is available." >&2
    return 1
  fi
}

size_of() {
  if stat -f '%z' "$1" >/dev/null 2>&1; then
    stat -f '%z' "$1"
  else
    stat -c '%s' "$1"
  fi
}

verify_model() {
  local path="$1"
  local actual_size
  local actual_sha

  actual_size="$(size_of "${path}")"
  if [[ "${actual_size}" != "${MODEL_SIZE}" ]]; then
    echo "Size verification failed: expected ${MODEL_SIZE}, got ${actual_size}." >&2
    return 1
  fi

  actual_sha="$(sha256_of "${path}")"
  if [[ "${actual_sha}" != "${MODEL_SHA256}" ]]; then
    echo "SHA-256 verification failed for ${path}." >&2
    echo "Expected: ${MODEL_SHA256}" >&2
    echo "Actual:   ${actual_sha}" >&2
    return 1
  fi
}

target="${DESTINATION}/${MODEL_NAME}"
partial="${target}.part"
url="https://huggingface.co/${REPOSITORY}/resolve/${REVISION}/${MODEL_NAME}?download=true"

if [[ -f "${target}" ]] && verify_model "${target}"; then
  echo "Verified existing ${MODEL_NAME}"
  exit 0
fi

if [[ -f "${partial}" ]]; then
  echo "Resuming ${REPOSITORY}@${REVISION}/${MODEL_NAME}"
else
  echo "Downloading ${REPOSITORY}@${REVISION}/${MODEL_NAME}"
fi

curl --location --fail --show-error \
  --retry 5 --retry-delay 2 --retry-all-errors \
  --continue-at - \
  --output "${partial}" "${url}"

verify_model "${partial}"
mv "${partial}" "${target}"
echo "Verified ${MODEL_NAME}"
