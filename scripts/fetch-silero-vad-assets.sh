#!/usr/bin/env bash
set -euo pipefail

# Downloads the exact reviewed Silero VAD model on the development machine.
# The Android app bundles the verified result and has no network permission.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DESTINATION="${1:-${PROJECT_DIR}/app/src/main/generatedVadAssets/silero}"
REVISION="7e30209a3e901f9842f81b225f3e93d8199902b1"
EXPECTED_SIZE="2327524"
EXPECTED_SHA256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
TARGET="${DESTINATION}/silero_vad.onnx"
PARTIAL="${TARGET}.part"
URL="https://raw.githubusercontent.com/snakers4/silero-vad/${REVISION}/src/silero_vad/data/silero_vad.onnx"

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
  local path="$1"
  local actual_size
  local actual_sha256
  actual_size="$(size_of "${path}")"
  actual_sha256="$(sha256_of "${path}")"
  [[ "${actual_size}" == "${EXPECTED_SIZE}" ]] || {
    echo "Silero VAD size mismatch: expected ${EXPECTED_SIZE}, got ${actual_size}." >&2
    return 1
  }
  [[ "${actual_sha256}" == "${EXPECTED_SHA256}" ]] || {
    echo "Silero VAD SHA-256 mismatch." >&2
    echo "Expected: ${EXPECTED_SHA256}" >&2
    echo "Actual:   ${actual_sha256}" >&2
    return 1
  }
}

if [[ -f "${TARGET}" ]] && verify "${TARGET}"; then
  echo "Verified existing Silero VAD v6.2.1 model."
  exit 0
fi

curl --location --fail --show-error \
  --retry 5 --retry-delay 2 --retry-all-errors \
  --continue-at - \
  --output "${PARTIAL}" "${URL}"
verify "${PARTIAL}"
mv "${PARTIAL}" "${TARGET}"
echo "Downloaded and verified Silero VAD v6.2.1."
