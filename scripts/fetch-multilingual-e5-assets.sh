#!/usr/bin/env bash
set -euo pipefail

# Downloads the exact reviewed multilingual-e5-small sentence-embedding model
# (MIT) used by semantic find-in-transcript. The Android app bundles the
# verified result and has no network permission.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DESTINATION="${1:-${PROJECT_DIR}/app/src/main/generatedEmbeddingAssets/embedding}"
REVISION="614241f622f53c4eeff9890bdc4f31cfecc418b3"
BASE_URL="https://huggingface.co/intfloat/multilingual-e5-small/resolve/${REVISION}/onnx"

# remote name|local name|bytes|sha256
FILES=(
  "model_qint8_avx512_vnni.onnx|multilingual_e5_small_qint8.onnx|118346824|dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88"
  "sentencepiece.bpe.model|multilingual_e5_small_sentencepiece.model|5069051|cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"
)

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

verify() {
  local path="$1" expected_size="$2" expected_sha256="$3"
  local actual_size actual_sha256
  actual_size="$(size_of "${path}")"
  [[ "${actual_size}" == "${expected_size}" ]] || {
    echo "$(basename "${path}") size mismatch: expected ${expected_size}, got ${actual_size}." >&2
    return 1
  }
  actual_sha256="$(sha256_of "${path}")"
  [[ "${actual_sha256}" == "${expected_sha256}" ]] || {
    echo "$(basename "${path}") SHA-256 mismatch." >&2
    echo "Expected: ${expected_sha256}" >&2
    echo "Actual:   ${actual_sha256}" >&2
    return 1
  }
}

for entry in "${FILES[@]}"; do
  IFS='|' read -r remote local expected_size expected_sha256 <<<"${entry}"
  target="${DESTINATION}/${local}"
  if [[ -f "${target}" ]] && verify "${target}" "${expected_size}" "${expected_sha256}"; then
    echo "Verified existing ${local}."
    continue
  fi
  partial="${target}.part"
  curl --location --fail --show-error \
    --retry 5 --retry-delay 2 --retry-all-errors \
    --continue-at - \
    --output "${partial}" "${BASE_URL}/${remote}"
  verify "${partial}" "${expected_size}" "${expected_sha256}"
  mv "${partial}" "${target}"
  echo "Downloaded and verified ${local}."
done
