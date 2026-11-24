#!/usr/bin/env bash
set -euo pipefail

# Downloads the exact reviewed speaker-labelling models used for "me / others"
# labels. The Android app bundles the verified result and has no network
# permission.
#  - pyannote segmentation-3.0 (MIT), ONNX export from the sherpa-onnx project
#  - 3D-Speaker CAM++ trained on Chinese and English (Apache 2.0), ONNX export
#    from the sherpa-onnx project

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DESTINATION="${1:-${PROJECT_DIR}/app/src/main/generatedSpeakerAssets/speaker}"
RELEASES="https://github.com/k2-fsa/sherpa-onnx/releases/download"

# url|member inside a tar.bz2 archive, or "-" for a plain file|local name|bytes|sha256
FILES=(
  "${RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2|sherpa-onnx-pyannote-segmentation-3-0/model.onnx|pyannote_segmentation_3_0.onnx|5992913|220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
  "${RELEASES}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx|-|3dspeaker_campplus_zh_en_advanced.onnx|28281165|53cab20ba8f2eb2f6e2d89ecc18e88ef3222169b61b4419cbc13666744a4574b"
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
  IFS='|' read -r url member local expected_size expected_sha256 <<<"${entry}"
  target="${DESTINATION}/${local}"
  if [[ -f "${target}" ]] && verify "${target}" "${expected_size}" "${expected_sha256}"; then
    echo "Verified existing ${local}."
    continue
  fi
  partial="${target}.part"
  if [[ "${member}" == "-" ]]; then
    curl --location --fail --show-error \
      --retry 5 --retry-delay 2 --retry-all-errors \
      --continue-at - \
      --output "${partial}" "${url}"
  else
    archive="${target}.archive"
    curl --location --fail --show-error \
      --retry 5 --retry-delay 2 --retry-all-errors \
      --continue-at - \
      --output "${archive}" "${url}"
    tar -xjOf "${archive}" "${member}" > "${partial}"
    rm -f "${archive}"
  fi
  verify "${partial}" "${expected_size}" "${expected_sha256}"
  mv "${partial}" "${target}"
  echo "Downloaded and verified ${local}."
done
