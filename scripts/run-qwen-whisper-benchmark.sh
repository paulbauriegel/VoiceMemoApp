#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
PACKAGE="com.example.voicememo"
TEST_PACKAGE="com.example.voicememo.test"
RUNNER="androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS="com.example.voicememo.transcription.QwenAsrWhisperBenchmarkTest"
DEVICE_CACHE="/data/user/0/${PACKAGE}/cache/qwen-benchmark"
DEVICE_MODEL="${DEVICE_CACHE}/model.tflite"
DEVICE_TOKENIZER="${DEVICE_CACHE}/tokenizer.json"
TEMP_MODEL="/data/local/tmp/qwen3_asr_0.6b_5s_i8.tflite"
TEMP_TOKENIZER="/data/local/tmp/qwen3_asr_tokenizer.json"
memo_id=""
reference_file=""
backends="cpu,gpu"
output_directory="${PROJECT_DIR}/build/reports/qwen-benchmark"

while (($#)); do
  case "$1" in
    --memo-id) memo_id="${2:?Missing memo ID}"; shift 2 ;;
    --whisper-reference) reference_file="${2:?Missing reference file}"; shift 2 ;;
    --backends) backends="${2:?Missing backends}"; shift 2 ;;
    --output) output_directory="${2:?Missing output directory}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${memo_id}" && -f "${reference_file}" ]] || {
  echo "Usage: $0 --memo-id ID --whisper-reference FILE [--backends cpu,gpu] [--output DIRECTORY]" >&2
  exit 2
}
[[ "${backends}" == "cpu" || "${backends}" == "gpu" || "${backends}" == "cpu,gpu" ]] || {
  echo "--backends must be cpu, gpu, or cpu,gpu" >&2
  exit 2
}

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
reference_base64="$(base64 <"${reference_file}" | tr -d '\n')"
model_file="${PROJECT_DIR}/build/model-assets/qwen/qwen3_asr_0.6b_5s_i8.tflite"
tokenizer_file="${PROJECT_DIR}/build/model-assets/qwen/tokenizer.json"
mkdir -p "${output_directory}"

cleanup_device_assets() {
  "${ADB}" shell rm -f "${TEMP_MODEL}" "${TEMP_TOKENIZER}" >/dev/null 2>&1 || true
  "${ADB}" shell run-as "${PACKAGE}" rm -f \
    "${DEVICE_MODEL}" "${DEVICE_TOKENIZER}" >/dev/null 2>&1 || true
}
trap cleanup_device_assets EXIT

cd "${PROJECT_DIR}"
scripts/fetch-qwen-asr-benchmark-assets.sh
JAVA_HOME="${JAVA_HOME:-/Applications/Android Studio.app/Contents/jbr/Contents/Home}" \
  ./gradlew --no-daemon :app:assembleDebugAndroidTest

# Updating only the test APK and using app cache keeps recordings and app data intact.
"${ADB}" install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
"${ADB}" push "${model_file}" "${TEMP_MODEL}"
"${ADB}" push "${tokenizer_file}" "${TEMP_TOKENIZER}"
"${ADB}" shell run-as "${PACKAGE}" mkdir -p "${DEVICE_CACHE}"
"${ADB}" shell run-as "${PACKAGE}" cp "${TEMP_MODEL}" "${DEVICE_MODEL}"
"${ADB}" shell run-as "${PACKAGE}" cp "${TEMP_TOKENIZER}" "${DEVICE_TOKENIZER}"
"${ADB}" shell am instrument -w -r \
  -e class "${TEST_CLASS}" \
  -e memo_id "${memo_id}" \
  -e run_id "${run_id}" \
  -e qwen_backends "${backends}" \
  -e qwen_model_path "${DEVICE_MODEL}" \
  -e qwen_tokenizer_path "${DEVICE_TOKENIZER}" \
  -e whisper_reference_text_b64 "${reference_base64}" \
  "${TEST_PACKAGE}/${RUNNER}"

report_name="qwen-${run_id}.json"
"${ADB}" exec-out run-as "${PACKAGE}" \
  cat "cache/qwen-benchmark/${report_name}" \
  >"${output_directory}/${report_name}"
echo "Report: ${output_directory}/${report_name}"
