#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIRECTORY="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"
SOURCE_REVISION="f049fff95a089aa9969deb009cdd4892b3e74916"
SOURCE_ARCHIVE_SHA256="279af4ce60dbf397362868f3bacc75b56a4332ac2541cae155070093f6aaf0e3"
SOURCE_DIRECTORY="${PROJECT_DIRECTORY}/build/third_party/whisper.cpp-v1.9.1"
SOURCE_PATCH="${PROJECT_DIRECTORY}/app/src/main/cpp/whispercpp-benchmark/voicememo-whispercpp-encoder-state.patch"
SOURCE_BLOCK_PATCH="${PROJECT_DIRECTORY}/app/src/main/cpp/whispercpp-benchmark/voicememo-whispercpp-encoder-blocks.patch"
VULKAN_HEADERS_VERSION="1.4.350.1"
VULKAN_HEADERS_ARCHIVE_SHA256="6d1bb65e49520344cc0a48af3dc02e993781efff14c7ebdcb8ae9fa23ddf7e83"
VULKAN_HEADERS_DIRECTORY="${PROJECT_DIRECTORY}/build/third_party/Vulkan-Headers-${VULKAN_HEADERS_VERSION}"
Q5_MODEL_REVISION="98aa99a0a9db05ae2342309f5096248665f7cba3"
Q5_MODEL_NAME="ggml-large-v3-turbo-q5_0.bin"
Q5_MODEL_BYTES="574041195"
Q5_MODEL_SHA256="394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"
Q8_MODEL_REVISION="0b364b566045a405be7225ee1e415a073e04da77"
Q8_MODEL_NAME="ggml-large-v3-turbo-q8_0.bin"
Q8_MODEL_BYTES="874188075"
Q8_MODEL_SHA256="317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1"
FP16_MODEL_REVISION="6034871ec87c84e342efab769d4c5c06cd126db3"
FP16_MODEL_NAME="ggml-large-v3-turbo.bin"
FP16_MODEL_BYTES="1624555275"
FP16_MODEL_SHA256="1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"
MODEL_DIRECTORY="${VOICEMEMO_WHISPER_BENCHMARK_MODEL_DIRECTORY:-${PROJECT_DIRECTORY}/build/model-assets/whisper.cpp}"
MODEL_VARIANT="${2:-q5_0}"
MODEL_REVISION=""
MODEL_NAME=""
MODEL_BYTES=""
MODEL_SHA256=""

usage() {
  cat <<'EOF'
Usage:
  scripts/fetch-whisper-cpp-benchmark.sh source
  scripts/fetch-whisper-cpp-benchmark.sh model [q5_0|q8_0|fp16]
  scripts/fetch-whisper-cpp-benchmark.sh all [q5_0|q8_0|fp16]
  scripts/fetch-whisper-cpp-benchmark.sh verify [q5_0|q8_0|fp16]

Fetches only benchmark inputs. Nothing is copied into production Android
assets. Downloads are pinned and rejected unless size and SHA-256 match.
EOF
}

fail() {
  echo "whisper.cpp benchmark assets: $*" >&2
  exit 1
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

select_model() {
  case "${MODEL_VARIANT}" in
    q5_0)
      MODEL_REVISION="${Q5_MODEL_REVISION}"
      MODEL_NAME="${Q5_MODEL_NAME}"
      MODEL_BYTES="${Q5_MODEL_BYTES}"
      MODEL_SHA256="${Q5_MODEL_SHA256}"
      ;;
    q8_0)
      MODEL_REVISION="${Q8_MODEL_REVISION}"
      MODEL_NAME="${Q8_MODEL_NAME}"
      MODEL_BYTES="${Q8_MODEL_BYTES}"
      MODEL_SHA256="${Q8_MODEL_SHA256}"
      ;;
    fp16)
      MODEL_REVISION="${FP16_MODEL_REVISION}"
      MODEL_NAME="${FP16_MODEL_NAME}"
      MODEL_BYTES="${FP16_MODEL_BYTES}"
      MODEL_SHA256="${FP16_MODEL_SHA256}"
      ;;
    *) fail "model must be q5_0, q8_0, or fp16." ;;
  esac
}

verify_source() {
  [[ -f "${SOURCE_DIRECTORY}/CMakeLists.txt" ]] ||
    fail "source is missing; run this script with 'source'."
  [[ -f "${SOURCE_DIRECTORY}/include/whisper.h" ]] ||
    fail "source checkout does not contain include/whisper.h."
  [[ -f "${SOURCE_DIRECTORY}/.voicememo-whispercpp-revision" ]] ||
    fail "source revision marker is missing."
  [[ "$(<"${SOURCE_DIRECTORY}/.voicememo-whispercpp-revision")" == "${SOURCE_REVISION}" ]] ||
    fail "source revision marker does not match ${SOURCE_REVISION}."
  grep -q "whisper_copy_encoder_state" "${SOURCE_DIRECTORY}/include/whisper.h" ||
    fail "benchmark encoder-state patch is missing; refetch the source directory."
  grep -q "whisper_copy_encoder_block_diagnostic" "${SOURCE_DIRECTORY}/include/whisper.h" ||
    fail "benchmark encoder-block patch is missing; refetch the source directory."
}

verify_model() {
  local model="${MODEL_DIRECTORY}/${MODEL_NAME}"
  [[ -f "${model}" ]] ||
    fail "${MODEL_VARIANT} model is missing; run this script with 'model ${MODEL_VARIANT}'."
  [[ "$(wc -c <"${model}" | tr -d ' ')" == "${MODEL_BYTES}" ]] ||
    fail "${MODEL_NAME} has the wrong size."
  [[ "$(sha256_file "${model}")" == "${MODEL_SHA256}" ]] ||
    fail "${MODEL_NAME} has the wrong SHA-256."
}

verify_vulkan_headers() {
  [[ -f "${VULKAN_HEADERS_DIRECTORY}/include/vulkan/vulkan.hpp" ]] ||
    fail "pinned Vulkan C++ headers are missing; run this script with 'source'."
}

fetch_source() (
  if [[ -d "${SOURCE_DIRECTORY}" ]]; then
    verify_source
    echo "Verified existing whisper.cpp source ${SOURCE_REVISION}."
    exit 0
  fi

  local temporary_directory
  local archive
  local extracted
  temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/voicememo-whispercpp.XXXXXX")"
  archive="${temporary_directory}/source.tar.gz"
  extracted="${temporary_directory}/whisper.cpp-${SOURCE_REVISION}"
  trap 'rm -rf "${temporary_directory}"' EXIT

  curl --location --fail --show-error --retry 5 --retry-all-errors \
    --output "${archive}" \
    "https://github.com/ggml-org/whisper.cpp/archive/${SOURCE_REVISION}.tar.gz"
  [[ "$(sha256_file "${archive}")" == "${SOURCE_ARCHIVE_SHA256}" ]] ||
    fail "whisper.cpp source archive has the wrong SHA-256."
  tar -xzf "${archive}" -C "${temporary_directory}"
  [[ -f "${extracted}/include/whisper.h" ]] ||
    fail "whisper.cpp archive layout changed unexpectedly."
  git -C "${extracted}" apply "${SOURCE_PATCH}"
  git -C "${extracted}" apply "${SOURCE_BLOCK_PATCH}"
  printf '%s\n' "${SOURCE_REVISION}" >"${extracted}/.voicememo-whispercpp-revision"
  mkdir -p "$(dirname "${SOURCE_DIRECTORY}")"
  mv "${extracted}" "${SOURCE_DIRECTORY}"
  verify_source
  echo "Fetched whisper.cpp ${SOURCE_REVISION}."
)

fetch_vulkan_headers() (
  if [[ -d "${VULKAN_HEADERS_DIRECTORY}" ]]; then
    verify_vulkan_headers
    echo "Verified existing Vulkan-Headers ${VULKAN_HEADERS_VERSION}."
    exit 0
  fi

  local temporary_directory
  local archive
  local extracted
  temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/voicememo-vulkan-headers.XXXXXX")"
  archive="${temporary_directory}/source.tar.gz"
  extracted="${temporary_directory}/Vulkan-Headers-vulkan-sdk-${VULKAN_HEADERS_VERSION}"
  trap 'rm -rf "${temporary_directory}"' EXIT

  curl --location --fail --show-error --retry 5 --retry-all-errors \
    --output "${archive}" \
    "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-${VULKAN_HEADERS_VERSION}.tar.gz"
  [[ "$(sha256_file "${archive}")" == "${VULKAN_HEADERS_ARCHIVE_SHA256}" ]] ||
    fail "Vulkan-Headers source archive has the wrong SHA-256."
  tar -xzf "${archive}" -C "${temporary_directory}"
  [[ -f "${extracted}/include/vulkan/vulkan.hpp" ]] ||
    fail "Vulkan-Headers archive layout changed unexpectedly."
  mkdir -p "$(dirname "${VULKAN_HEADERS_DIRECTORY}")"
  mv "${extracted}" "${VULKAN_HEADERS_DIRECTORY}"
  verify_vulkan_headers
  echo "Fetched Vulkan-Headers ${VULKAN_HEADERS_VERSION}."
)

fetch_model() {
  local model="${MODEL_DIRECTORY}/${MODEL_NAME}"
  local partial="${model}.part"
  mkdir -p "${MODEL_DIRECTORY}"
  if [[ -f "${model}" ]]; then
    verify_model
    echo "Verified existing ${MODEL_NAME}."
    return
  fi
  curl --location --fail --show-error --retry 5 --retry-all-errors \
    --continue-at - --output "${partial}" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/${MODEL_REVISION}/${MODEL_NAME}?download=true"
  [[ "$(wc -c <"${partial}" | tr -d ' ')" == "${MODEL_BYTES}" ]] ||
    fail "downloaded ${MODEL_NAME} has the wrong size."
  [[ "$(sha256_file "${partial}")" == "${MODEL_SHA256}" ]] ||
    fail "downloaded ${MODEL_NAME} has the wrong SHA-256."
  mv "${partial}" "${model}"
  verify_model
  echo "Fetched ${MODEL_NAME}."
}

[[ $# -ge 1 && $# -le 2 ]] || {
  usage >&2
  exit 2
}

select_model
case "$1" in
  source)
    [[ $# -eq 1 ]] || fail "'source' does not accept a model argument."
    fetch_source
    fetch_vulkan_headers
    ;;
  model) fetch_model ;;
  all) fetch_source; fetch_vulkan_headers; fetch_model ;;
  verify) verify_source; verify_vulkan_headers; verify_model; echo "whisper.cpp benchmark inputs verified." ;;
  -h | --help) usage ;;
  *) usage >&2; exit 2 ;;
esac
