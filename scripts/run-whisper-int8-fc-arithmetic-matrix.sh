#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
package_name="com.example.voicememo"
test_package_name="com.example.voicememo.test"
runner_name="androidx.test.runner.AndroidJUnitRunner"
test_class="com.example.voicememo.transcription.WhisperInt8FcArithmeticMatrixTest"
test_apk="$project_dir/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
adb_binary="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
jbr="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
device_serial=""
output_directory=""
skip_build=false
adb_arguments=()

fail() {
  echo "Whisper INT8 FC arithmetic matrix failed: $*" >&2
  exit 1
}

adb_command() {
  "$adb_binary" "${adb_arguments[@]}" "$@"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --serial)
      [[ "$#" -ge 2 ]] || fail "--serial requires a value."
      device_serial="$2"
      shift 2
      ;;
    --output)
      [[ "$#" -ge 2 ]] || fail "--output requires a value."
      output_directory="$2"
      shift 2
      ;;
    --no-build)
      skip_build=true
      shift
      ;;
    -h | --help)
      cat <<'EOF'
Usage: scripts/run-whisper-int8-fc-arithmetic-matrix.sh [options]

Builds seven isolated K-projection models, installs only the instrumentation
APK, runs controlled CPU/direct-GPU/source-quantized-GPU inputs, and retrieves
an arithmetic diagnosis. The installed Voice Memo app and recordings are not
replaced, cleared, or uninstalled.

Options:
  --serial SERIAL       Specific authorized Android device.
  --output DIRECTORY    New host report directory.
  --no-build            Reuse the existing instrumentation APK and probe assets.
EOF
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

command -v "$adb_binary" >/dev/null 2>&1 || fail "adb not found: $adb_binary"
[[ -x "$jbr/bin/java" ]] || fail "Android Studio JBR is missing."
if [[ -n "$device_serial" ]]; then
  adb_arguments=(-s "$device_serial")
else
  devices=()
  while IFS= read -r serial; do
    [[ -n "$serial" ]] && devices+=("$serial")
  done < <("$adb_binary" devices | awk 'NR > 1 && $2 == "device" {print $1}')
  [[ "${#devices[@]}" -eq 1 ]] ||
    fail "Expected exactly one authorized device; pass --serial."
  device_serial="${devices[0]}"
  adb_arguments=(-s "$device_serial")
fi

if [[ "$skip_build" != true ]]; then
  (
    cd "$project_dir"
    JAVA_HOME="$jbr" PATH="$jbr/bin:$PATH" \
      ./gradlew --no-daemon \
        -p tools/whisper-decoder-split installDist
    JAVA_HOME="$jbr" "$jbr/bin/java" -Xmx1g \
      -cp 'tools/whisper-decoder-split/build/install/whisper-decoder-split/lib/*' \
      com.example.voicememo.tools.WhisperAttentionKeyTransposeTool \
      generate-k-projection-arithmetic-probes \
      build/generated/whisper-decoder-prefix/k-projection-probe.tflite \
      build/generated/whisper-int8-fc-matrix
    JAVA_HOME="$jbr" \
      ANDROID_HOME="$HOME/Library/Android/sdk" \
      PATH="$jbr/bin:$HOME/Library/Android/sdk/platform-tools:$PATH" \
      ./gradlew --no-daemon \
        -Pkotlin.compiler.execution.strategy=in-process \
        -PincludeInt8FcArithmeticDiagnostic=true \
        :app:assembleDebugAndroidTest
  )
fi

[[ -f "$test_apk" ]] || fail "Instrumentation APK is missing: $test_apk"
adb_command shell pm path "$package_name" >/dev/null ||
  fail "Voice Memo is not installed on the device."
adb_command install -r "$test_apk"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
if [[ -z "$output_directory" ]]; then
  output_directory="$project_dir/build/reports/device-gpu/int8-fc-arithmetic-$run_id"
fi
[[ ! -e "$output_directory" ]] || fail "Output already exists: $output_directory"
mkdir -p "$output_directory"

set +e
adb_command shell am instrument -w -r \
  -e runId "$run_id" \
  -e class "$test_class" \
  "$test_package_name/$runner_name" 2>&1 |
  tee "$output_directory/instrumentation.txt"
pipeline_status=("${PIPESTATUS[@]}")
set -e
[[ "${pipeline_status[1]}" -eq 0 ]] || fail "Could not save instrumentation output."

device_report="cache/gpu-int8-fc-arithmetic/arithmetic-$run_id.json"
adb_command exec-out run-as "$package_name" cat "$device_report" \
  >"$output_directory/report.json" ||
  fail "Could not retrieve the device report."
python3 "$project_dir/scripts/analyze-whisper-int8-fc-arithmetic-matrix.py" \
  "$output_directory/report.json" |
  tee "$output_directory/summary.txt"

[[ "${pipeline_status[0]}" -eq 0 ]] ||
  fail "Instrumentation exited with ${pipeline_status[0]}."
echo "Saved Whisper INT8 FC arithmetic report: $output_directory"
