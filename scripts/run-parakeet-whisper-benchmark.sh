#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
PACKAGE="com.example.voicememo"
TEST_PACKAGE="com.example.voicememo.test"
RUNNER="androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS="com.example.voicememo.transcription.ParakeetWhisperBenchmarkTest"
memo_id=""
output_directory="${PROJECT_DIR}/build/reports/parakeet-benchmark"

while (($#)); do
  case "$1" in
    --memo-id) memo_id="${2:?Missing memo ID}"; shift 2 ;;
    --output) output_directory="${2:?Missing output directory}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${memo_id}" ]] || {
  echo "Usage: $0 --memo-id ID [--output DIRECTORY]" >&2
  exit 2
}

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${output_directory}"

cd "${PROJECT_DIR}"
JAVA_HOME="${JAVA_HOME:-/Applications/Android Studio.app/Contents/jbr/Contents/Home}" \
  ./gradlew --no-daemon :app:assembleDebug :app:assembleDebugAndroidTest

# In-place updates preserve app-private recordings and transcripts.
"${ADB}" install -r app/build/outputs/apk/debug/app-debug.apk
"${ADB}" install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
"${ADB}" shell am instrument -w \
  -e class "${TEST_CLASS}" \
  -e memo_id "${memo_id}" \
  -e run_id "${run_id}" \
  "${TEST_PACKAGE}/${RUNNER}"

report_name="parakeet-${run_id}.json"
"${ADB}" exec-out run-as "${PACKAGE}" \
  cat "cache/parakeet-benchmark/${report_name}" \
  >"${output_directory}/${report_name}"
echo "Report: ${output_directory}/${report_name}"
