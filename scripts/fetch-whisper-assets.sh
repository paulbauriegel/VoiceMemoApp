#!/usr/bin/env bash
set -euo pipefail

# Downloads the exact, reviewed Whisper artifacts used by Voice Memo.
# This is a developer/build-machine operation. The Android application itself
# has no network permission and never executes this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DESTINATION="${1:-${PROJECT_DIR}/app/src/main/generatedAssets/whisper}"
TOOL_DIR="${PROJECT_DIR}/tools/whisper-decoder-split"
SOURCE_MODEL_DIR="${PROJECT_DIR}/build/model-assets/whisper"
REWRITE_OUTPUT_DIR="${PROJECT_DIR}/build/generated/whisper-key-transpose"
SOURCE_MODEL_NAME="whisper_large_v3_turbo_30s_i8.tflite"
REWRITTEN_MODEL_NAME="whisper_large_v3_turbo_30s_i8_explicit_key_transpose.tflite"
SOURCE_MODEL_SIZE="1088340944"
SOURCE_MODEL_SHA256="c40690d926fe9841ef6e0296eb0c4168c2910171c1d67b4bcf00a4b7cd7989ca"
REWRITTEN_MODEL_SIZE="1094571944"
REWRITTEN_MODEL_SHA256="8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418"

LITERT_REPOSITORY="litert-community/whisper-large-v3-turbo"
LITERT_REVISION="4e3af6a1597a54b6e47d2899cf4f6c39932b0b7e"
TOKENIZER_REPOSITORY="openai/whisper-large-v3-turbo"
TOKENIZER_REVISION="41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

mkdir -p "${DESTINATION}" "${SOURCE_MODEL_DIR}" "${REWRITE_OUTPUT_DIR}"

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

verify_file() {
  local path="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local actual_size
  local actual_sha

  actual_size="$(size_of "${path}")"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "Size verification failed for ${path}: expected ${expected_size}, got ${actual_size}." >&2
    return 1
  fi

  actual_sha="$(sha256_of "${path}")"
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    echo "SHA-256 verification failed for ${path}." >&2
    echo "Expected: ${expected_sha}" >&2
    echo "Actual:   ${actual_sha}" >&2
    return 1
  fi
}

download_verified() {
  local repository="$1"
  local revision="$2"
  local file_name="$3"
  local expected_size="$4"
  local expected_sha="$5"
  local target_directory="${6:-${DESTINATION}}"
  local target="${target_directory}/${file_name}"
  local partial="${target}.part"
  local url="https://huggingface.co/${repository}/resolve/${revision}/${file_name}?download=true"

  if [[ -f "${target}" ]]; then
    if verify_file "${target}" "${expected_size}" "${expected_sha}"; then
      echo "Verified existing ${file_name}"
      return
    fi
    echo "Existing ${file_name} is invalid; downloading a clean copy." >&2
  fi

  if [[ -f "${partial}" ]]; then
    echo "Resuming ${repository}@${revision}/${file_name}"
  else
    echo "Downloading ${repository}@${revision}/${file_name}"
  fi
  curl --location --fail --show-error \
    --retry 5 --retry-delay 2 --retry-all-errors \
    --continue-at - \
    --output "${partial}" "${url}"

  verify_file "${partial}" "${expected_size}" "${expected_sha}"
  mv "${partial}" "${target}"
  echo "Verified ${file_name}"
}

legacy_model="${DESTINATION}/${SOURCE_MODEL_NAME}"
cached_source_model="${SOURCE_MODEL_DIR}/${SOURCE_MODEL_NAME}"
if [[ ! -f "${cached_source_model}" && -f "${legacy_model}" ]] &&
    verify_file \
      "${legacy_model}" \
      "${SOURCE_MODEL_SIZE}" \
      "${SOURCE_MODEL_SHA256}"; then
  mv "${legacy_model}" "${cached_source_model}"
  echo "Reused the existing pinned source model from Android assets."
fi

download_verified \
  "${LITERT_REPOSITORY}" \
  "${LITERT_REVISION}" \
  "${SOURCE_MODEL_NAME}" \
  "${SOURCE_MODEL_SIZE}" \
  "${SOURCE_MODEL_SHA256}" \
  "${SOURCE_MODEL_DIR}"

rewritten_model="${DESTINATION}/${REWRITTEN_MODEL_NAME}"
if [[ -f "${rewritten_model}" ]] &&
    verify_file \
      "${rewritten_model}" \
      "${REWRITTEN_MODEL_SIZE}" \
      "${REWRITTEN_MODEL_SHA256}"; then
  echo "Verified existing ${REWRITTEN_MODEL_NAME}"
else
  java_home="${JAVA_HOME:-/Applications/Android Studio.app/Contents/jbr/Contents/Home}"
  if [[ ! -x "${java_home}/bin/java" ]]; then
    echo "JDK 17+ is required to rewrite the pinned Whisper model." >&2
    echo "Set JAVA_HOME or install Android Studio's bundled JBR." >&2
    exit 1
  fi
  JAVA_HOME="${java_home}" \
    PATH="${java_home}/bin:${PATH}" \
    "${PROJECT_DIR}/gradlew" \
      -p "${TOOL_DIR}" \
      --no-daemon \
      installDist
  rewrite_candidate="${REWRITE_OUTPUT_DIR}/${REWRITTEN_MODEL_NAME}"
  "${java_home}/bin/java" \
    -Xms128m \
    -Xmx1g \
    -cp "${TOOL_DIR}/build/install/whisper-decoder-split/lib/*" \
    com.example.voicememo.tools.WhisperAttentionKeyTransposeTool \
    generate \
    "${SOURCE_MODEL_DIR}/${SOURCE_MODEL_NAME}" \
    "${rewrite_candidate}"
  verify_file \
    "${rewrite_candidate}" \
    "${REWRITTEN_MODEL_SIZE}" \
    "${REWRITTEN_MODEL_SHA256}"
  mv "${rewrite_candidate}" "${rewritten_model}"
  echo "Generated and verified ${REWRITTEN_MODEL_NAME}"
fi

if [[ -f "${legacy_model}" ]]; then
  if verify_file \
      "${legacy_model}" \
      "${SOURCE_MODEL_SIZE}" \
      "${SOURCE_MODEL_SHA256}"; then
    mv "${legacy_model}" "${SOURCE_MODEL_DIR}/${SOURCE_MODEL_NAME}"
    echo "Moved the unmodified source model out of Android assets."
  else
    echo "Unexpected legacy model remains in Android assets: ${legacy_model}" >&2
    exit 1
  fi
fi

download_verified \
  "${TOKENIZER_REPOSITORY}" \
  "${TOKENIZER_REVISION}" \
  "tokenizer.json" \
  "2710337" \
  "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"

download_verified \
  "${TOKENIZER_REPOSITORY}" \
  "${TOKENIZER_REVISION}" \
  "generation_config.json" \
  "3772" \
  "cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c"

download_verified \
  "${TOKENIZER_REPOSITORY}" \
  "${TOKENIZER_REVISION}" \
  "preprocessor_config.json" \
  "340" \
  "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"

echo "All pinned Whisper assets are ready in ${DESTINATION}"
