#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIRECTORY="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"
PACKAGE_NAME="com.example.voicememo"
TEST_PACKAGE_NAME="com.example.voicememo.test"
TEST_RUNNER="androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS="com.example.voicememo.transcription.GpuWholeMemoWerBenchmarkTest"
Q5_MODEL_NAME="ggml-large-v3-turbo-q5_0.bin"
Q5_MODEL_BYTES="574041195"
Q5_MODEL_SHA256="394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"
Q8_MODEL_NAME="ggml-large-v3-turbo-q8_0.bin"
Q8_MODEL_BYTES="874188075"
Q8_MODEL_SHA256="317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1"
FP16_MODEL_NAME="ggml-large-v3-turbo.bin"
FP16_MODEL_BYTES="1624555275"
FP16_MODEL_SHA256="1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"
MODEL_NAME=""
MODEL_BYTES=""
MODEL_SHA256=""
MODEL_PATH=""
ADB="${ADB:-${HOME}/Library/Android/sdk/platform-tools/adb}"
JAVA_HOME_VALUE="${JAVA_HOME:-/Applications/Android Studio.app/Contents/jbr/Contents/Home}"
SQLITE3="${SQLITE3:-sqlite3}"

device_serial=""
memo_id=""
audio_path=""
audio_name=""
audio_bytes=""
audio_sha256=""
threads=4
backend="vulkan"
model_variant="q5_0"
language="auto"
output_directory="${PROJECT_DIRECTORY}/build/reports/whispercpp-benchmark"
adb_arguments=()
staged_device_model=false
staged_device_audio=false
instrumentation_output=""

usage() {
  cat <<'EOF'
Usage:
  scripts/run-whisper-cpp-benchmark.sh run (--memo-id ID | --audio FILE) [options]

Options:
  --audio FILE         Read a host WAV staged as benchmark-only input.
  --backend MODE       whisper.cpp backend: vulkan or cpu (default: vulkan).
  --model MODEL        weights: q5_0, q8_0, or fp16 (default: q5_0).
  --threads N          whisper.cpp CPU threads, 1-8 (default: 4).
  --language LANGUAGE  auto or a lowercase 2-3 letter code (default: auto).
  --serial SERIAL      Select one authorized adb device.
  --output DIRECTORY   Host report directory.
  -h, --help           Show this help.

The explicit "run" subcommand is required. The command refuses to proceed
while VoiceMemo recording/transcription services are active, installs only
with adb install -r, reads one existing memo or explicit WAV, and never writes
transcript data.
EOF
}

fail() {
  echo "whisper.cpp benchmark stopped: $*" >&2
  exit 1
}

adb_command() {
  "${ADB}" "${adb_arguments[@]}" "$@"
}

select_model() {
  case "${model_variant}" in
    q5_0)
      MODEL_NAME="${Q5_MODEL_NAME}"
      MODEL_BYTES="${Q5_MODEL_BYTES}"
      MODEL_SHA256="${Q5_MODEL_SHA256}"
      ;;
    q8_0)
      MODEL_NAME="${Q8_MODEL_NAME}"
      MODEL_BYTES="${Q8_MODEL_BYTES}"
      MODEL_SHA256="${Q8_MODEL_SHA256}"
      ;;
    fp16)
      MODEL_NAME="${FP16_MODEL_NAME}"
      MODEL_BYTES="${FP16_MODEL_BYTES}"
      MODEL_SHA256="${FP16_MODEL_SHA256}"
      ;;
    *) fail "--model must be q5_0, q8_0, or fp16." ;;
  esac
  MODEL_PATH="${PROJECT_DIRECTORY}/build/model-assets/whisper.cpp/${MODEL_NAME}"
}

cleanup() {
  if [[ "${staged_device_model}" == true ]]; then
    adb_command shell run-as "${PACKAGE_NAME}" \
      rm -f "files/whispercpp-models/${MODEL_NAME}" >/dev/null 2>&1 || true
  fi
  if [[ "${staged_device_audio}" == true ]]; then
    adb_command shell run-as "${PACKAGE_NAME}" \
      rm -f "files/gpu-audio/${audio_name}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${instrumentation_output}" ]]; then
    rm -f "${instrumentation_output}"
  fi
}

trap cleanup EXIT

resolve_device() {
  if [[ -n "${device_serial}" ]]; then
    adb_arguments=(-s "${device_serial}")
    [[ "$(adb_command get-state 2>/dev/null || true)" == "device" ]] ||
      fail "device ${device_serial} is not connected and authorized."
    return
  fi
  local devices=()
  local connected_serial
  while IFS= read -r connected_serial; do
    [[ -n "${connected_serial}" ]] && devices+=("${connected_serial}")
  done < <("${ADB}" devices | awk 'NR > 1 && $2 == "device" {print $1}')
  [[ "${#devices[@]}" -eq 1 ]] ||
    fail "connect exactly one authorized device or pass --serial."
  device_serial="${devices[0]}"
  adb_arguments=(-s "${device_serial}")
}

assert_device_idle() {
  local services
  local notifications
  adb_command shell pm path "${PACKAGE_NAME}" | grep -q '^package:' ||
    fail "VoiceMemo is not installed; an existing memo is required."
  adb_command shell run-as "${PACKAGE_NAME}" id >/dev/null 2>&1 ||
    fail "the installed VoiceMemo build is not debuggable with run-as."
  services="$(adb_command shell dumpsys activity services "${PACKAGE_NAME}" | tr -d '\r')"
  notifications="$(adb_command shell dumpsys notification --noredact | tr -d '\r')"
  if grep -E -q 'RecordingService|SystemForegroundService' <<<"${services}"; then
    fail "VoiceMemo recording or transcription is active."
  fi
  if grep -E -q \
    "pkg=${PACKAGE_NAME}.*id=(1001|1002)([^0-9]|$)|id=(1001|1002)([^0-9]|$).*pkg=${PACKAGE_NAME}" \
    <<<"${notifications}"; then
    fail "VoiceMemo recording or transcription notification is active."
  fi
  assert_database_idle
}

database_snapshot_query_once() {
  local snapshot_directory
  local database_path
  snapshot_directory="$(mktemp -d "${TMPDIR:-/tmp}/voicememo-whispercpp-db.XXXXXX")"
  database_path="${snapshot_directory}/voice-memo.db"
  adb_command exec-out run-as "${PACKAGE_NAME}" \
    cat databases/voice-memo.db >"${database_path}" ||
    fail "could not read the VoiceMemo database."
  for suffix in -wal -shm; do
    if adb_command shell run-as "${PACKAGE_NAME}" \
      test -f "databases/voice-memo.db${suffix}"; then
      adb_command exec-out run-as "${PACKAGE_NAME}" \
        cat "databases/voice-memo.db${suffix}" >"${database_path}${suffix}" ||
        fail "could not read the VoiceMemo database${suffix}."
    fi
  done
  "${SQLITE3}" -readonly "${database_path}" \
    "PRAGMA query_only=ON;
     SELECT
       (SELECT COUNT(*) FROM transcription_jobs
       WHERE status IN ('QUEUED','RUNNING','PAUSED_FOR_RECORDING','CANCEL_REQUESTED'))
       || '|' ||
       CASE WHEN '${memo_id}' = '' THEN 1 ELSE
         (SELECT COUNT(*) FROM memos
          WHERE id = '${memo_id}' AND pendingDeletion = 0)
       END;" ||
    fail "could not inspect a read-only VoiceMemo database snapshot."
  rm -rf "${snapshot_directory}"
}

assert_database_idle() {
  local first
  local second
  local active_jobs
  local memo_matches
  first="$(database_snapshot_query_once)"
  second="$(database_snapshot_query_once)"
  [[ "${first}" == "${second}" ]] ||
    fail "two VoiceMemo database snapshots disagreed; device state is changing."
  IFS='|' read -r active_jobs memo_matches <<<"${first}"
  [[ "${active_jobs}" == "0" ]] ||
    fail "${active_jobs} queued or active VoiceMemo transcription job(s) exist."
  [[ -z "${memo_id}" || "${memo_matches}" == "1" ]] ||
    fail "memo ${memo_id} does not exist or is pending deletion."
}

verify_host_model() {
  [[ -f "${MODEL_PATH}" ]] ||
    fail "model missing; run scripts/fetch-whisper-cpp-benchmark.sh model ${model_variant}."
  [[ "$(wc -c <"${MODEL_PATH}" | tr -d ' ')" == "${MODEL_BYTES}" ]] ||
    fail "host model size does not match the pinned artifact."
  [[ "$(shasum -a 256 "${MODEL_PATH}" | awk '{print $1}')" == "${MODEL_SHA256}" ]] ||
    fail "host model SHA-256 does not match the pinned artifact."
}

stage_model() {
  adb_command shell run-as "${PACKAGE_NAME}" mkdir -p files/whispercpp-models
  adb_command shell run-as "${PACKAGE_NAME}" \
    tee "files/whispercpp-models/${MODEL_NAME}" \
    <"${MODEL_PATH}" >/dev/null
  staged_device_model=true

  local device_size
  local device_sha
  device_size="$(
    adb_command shell run-as "${PACKAGE_NAME}" \
      wc -c "files/whispercpp-models/${MODEL_NAME}" |
      awk '{print $1}' | tr -d '\r'
  )"
  device_sha="$(
    adb_command shell run-as "${PACKAGE_NAME}" \
      sha256sum "files/whispercpp-models/${MODEL_NAME}" |
      awk '{print $1}' | tr -d '\r'
  )"
  [[ "${device_size}" == "${MODEL_BYTES}" ]] ||
    fail "staged device model size does not match."
  [[ "${device_sha}" == "${MODEL_SHA256}" ]] ||
    fail "staged device model SHA-256 does not match."
}

verify_host_audio() {
  [[ -f "${audio_path}" ]] || fail "benchmark audio does not exist: ${audio_path}"
  [[ "${audio_path##*.}" == "wav" ]] || fail "--audio must be a WAV file."
  audio_path="$(cd "$(dirname "${audio_path}")" && pwd)/$(basename "${audio_path}")"
  audio_name="$(basename "${audio_path}")"
  [[ "${audio_name}" =~ ^[A-Za-z0-9._-]+$ ]] ||
    fail "benchmark audio file name contains unsupported characters."
  audio_bytes="$(wc -c <"${audio_path}" | tr -d ' ')"
  audio_sha256="$(shasum -a 256 "${audio_path}" | awk '{print $1}')"
}

stage_audio() {
  adb_command shell run-as "${PACKAGE_NAME}" mkdir -p files/gpu-audio
  adb_command shell run-as "${PACKAGE_NAME}" \
    tee "files/gpu-audio/${audio_name}" \
    <"${audio_path}" >/dev/null
  staged_device_audio=true

  local device_size
  local device_sha
  device_size="$(
    adb_command shell run-as "${PACKAGE_NAME}" \
      wc -c "files/gpu-audio/${audio_name}" |
      awk '{print $1}' | tr -d '\r'
  )"
  device_sha="$(
    adb_command shell run-as "${PACKAGE_NAME}" \
      sha256sum "files/gpu-audio/${audio_name}" |
      awk '{print $1}' | tr -d '\r'
  )"
  [[ "${device_size}" == "${audio_bytes}" ]] ||
    fail "staged benchmark audio size does not match."
  [[ "${device_sha}" == "${audio_sha256}" ]] ||
    fail "staged benchmark audio SHA-256 does not match."
}

[[ "${1:-}" == "run" ]] || {
  usage >&2
  exit 2
}
shift
while (($#)); do
  case "$1" in
    --memo-id) memo_id="${2:?Missing memo ID}"; shift 2 ;;
    --audio) audio_path="${2:?Missing audio path}"; shift 2 ;;
    --backend) backend="${2:?Missing backend}"; shift 2 ;;
    --model) model_variant="${2:?Missing model}" ; shift 2 ;;
    --threads) threads="${2:?Missing thread count}"; shift 2 ;;
    --language) language="${2:?Missing language}"; shift 2 ;;
    --serial) device_serial="${2:?Missing device serial}"; shift 2 ;;
    --output) output_directory="${2:?Missing output directory}"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ -z "${memo_id}" || "${memo_id}" =~ ^[A-Za-z0-9-]+$ ]] ||
  fail "--memo-id is invalid."
[[ -z "${memo_id}" || -z "${audio_path}" ]] ||
  fail "pass exactly one of --memo-id or --audio."
[[ -n "${memo_id}" || -n "${audio_path}" ]] ||
  fail "pass exactly one of --memo-id or --audio."
[[ "${threads}" =~ ^[1-8]$ ]] || fail "--threads must be between 1 and 8."
[[ "${backend}" == "vulkan" || "${backend}" == "cpu" ]] ||
  fail "--backend must be vulkan or cpu."
[[ "${language}" == "auto" || "${language}" =~ ^[a-z]{2,3}$ ]] ||
  fail "--language must be auto or a lowercase 2-3 letter code."
select_model
command -v "${ADB}" >/dev/null 2>&1 || fail "adb was not found at ${ADB}."
command -v "${SQLITE3}" >/dev/null 2>&1 || fail "sqlite3 was not found."
verify_host_model
[[ -z "${audio_path}" ]] || verify_host_audio
resolve_device
assert_device_idle

cd "${PROJECT_DIRECTORY}"
JAVA_HOME="${JAVA_HOME_VALUE}" ./gradlew --no-daemon \
  -PincludeWhisperCppBenchmark=true \
  :app:assembleDebug :app:assembleDebugAndroidTest

assert_device_idle
adb_command install -r app/build/outputs/apk/debug/app-debug.apk
adb_command install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
stage_model
[[ -z "${audio_path}" ]] || stage_audio

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${output_directory}"
instrumentation_arguments=(
  -e class "${TEST_CLASS}" \
  -e runId "${run_id}" \
  -e candidateOnly true \
  -e candidateBackend whispercpp \
  -e whisperCppThreads "${threads}" \
  -e whisperCppBackend "${backend}" \
  -e externalModelName "${MODEL_NAME}" \
  -e externalModelSha256 "${MODEL_SHA256}"
)
if [[ -n "${audio_path}" ]]; then
  instrumentation_arguments+=(-e externalAudioName "${audio_name}")
else
  instrumentation_arguments+=(-e memoId "${memo_id}")
fi
if [[ "${language}" != "auto" ]]; then
  instrumentation_arguments+=(-e forcedLanguage "${language}")
fi
instrumentation_output="$(mktemp "${TMPDIR:-/tmp}/voicememo-whispercpp-instrumentation.XXXXXX")"
adb_command shell am instrument -w \
  "${instrumentation_arguments[@]}" \
  "${TEST_PACKAGE_NAME}/${TEST_RUNNER}" |
  tee "${instrumentation_output}"
grep -q '^OK (' "${instrumentation_output}" ||
  fail "instrumentation did not complete successfully; no benchmark report was accepted."
if grep -q '^INSTRUMENTATION_CODE:' "${instrumentation_output}"; then
  grep -q '^INSTRUMENTATION_CODE: -1$' "${instrumentation_output}" ||
    fail "instrumentation returned an unexpected completion code; no benchmark report was accepted."
fi

report_name="whispercpp-${model_variant}-${backend}-${run_id}.json"
adb_command exec-out run-as "${PACKAGE_NAME}" \
  cat "cache/gpu-wer/whole-${run_id}.json" \
  >"${output_directory}/${report_name}"
adb_command shell run-as "${PACKAGE_NAME}" \
  rm -f "files/whispercpp-models/${MODEL_NAME}"
staged_device_model=false
if [[ -n "${audio_path}" ]]; then
  adb_command shell run-as "${PACKAGE_NAME}" rm -f "files/gpu-audio/${audio_name}"
  staged_device_audio=false
fi
echo "Report: ${output_directory}/${report_name}"
