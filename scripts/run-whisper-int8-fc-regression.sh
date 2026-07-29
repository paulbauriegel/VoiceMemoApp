#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
package_name="com.example.voicememo"
test_package_name="com.example.voicememo.test"
runner_name="androidx.test.runner.AndroidJUnitRunner"
test_class="com.example.voicememo.transcription.WhisperInt8FullyConnectedRegressionTest"
app_apk="$project_dir/app/build/outputs/apk/debug/app-debug.apk"
test_apk="$project_dir/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
adb_binary="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
device_serial=""
output_directory=""
skip_build=false
benchmark_mode=true
adb_arguments=()

fail() {
  echo "Whisper INT8 FC regression failed: $*" >&2
  exit 1
}

adb_command() {
  "$adb_binary" "${adb_arguments[@]}" "$@"
}

host_sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
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
    --no-benchmark-mode)
      benchmark_mode=false
      shift
      ;;
    -h | --help)
      cat <<'EOF'
Usage: scripts/run-whisper-int8-fc-regression.sh [options]

Runs a standalone CPU/GPU comparison of Whisper's exact first-block,
per-channel INT8 K projection using a fixed realistic step-7 input.

Options:
  --serial SERIAL       Specific authorized Android device.
  --output DIRECTORY    New host report directory.
  --no-build            Reuse the existing instrumentation APK.
  --no-benchmark-mode   Do not request ML Drift kernel reporting.
EOF
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

command -v "$adb_binary" >/dev/null 2>&1 || fail "adb not found: $adb_binary"
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

fixture_dir="$project_dir/build/generated/whisper-int8-fc"
fixture="$fixture_dir/whisper-first-block-k-projection-step-7-input.f32"
if [[ ! -f "$fixture" ]]; then
  python3 "$project_dir/scripts/generate-whisper-int8-fc-fixture.py" \
    "$project_dir/app/src/main/generatedAssets/whisper/"\
"whisper_large_v3_turbo_30s_i8_explicit_key_transpose.tflite" \
    "$fixture_dir"
fi
[[ "$(host_sha256 "$fixture")" == \
  "105b86c9c20463e3920d871f0f2a653a9da88ce8739572f489e077754f6c29cb" ]] ||
  fail "Fixed K-projection input SHA-256 mismatch."

if [[ "$skip_build" != true ]]; then
  (
    cd "$project_dir"
    JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
      ANDROID_HOME="$HOME/Library/Android/sdk" \
      PATH="/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin:"\
"$HOME/Library/Android/sdk/platform-tools:$PATH" \
      ./gradlew --no-daemon \
        -Pkotlin.compiler.execution.strategy=in-process \
        -PincludeInt8FcRegressionDiagnostic=true \
        :app:assembleDebugAndroidTest
  )
fi

[[ -f "$app_apk" ]] || fail "Debug app APK is missing: $app_apk"
[[ -f "$test_apk" ]] || fail "Instrumentation APK is missing: $test_apk"
installed_app_path="$(
  adb_command shell pm path "$package_name" |
    tr -d '\r' |
    sed -n 's/^package://p' |
    head -n 1
)"
[[ -n "$installed_app_path" ]] || fail "Install the debug Voice Memo app first."
installed_sha="$(
  adb_command shell sha256sum "$installed_app_path" |
    tr -d '\r' |
    awk '{print $1}'
)"
[[ "$installed_sha" == "$(host_sha256 "$app_apk")" ]] ||
  fail "Installed app differs from app-debug.apk; refusing to replace app data implicitly."

adb_command install -r "$test_apk"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
if [[ -z "$output_directory" ]]; then
  output_directory="$project_dir/build/reports/device-gpu/int8-fc-$run_id"
fi
[[ ! -e "$output_directory" ]] || fail "Output already exists: $output_directory"
mkdir -p "$output_directory"

adb_command logcat -c
set +e
adb_command shell am instrument -w -r \
  -e runId "$run_id" \
  -e benchmarkMode "$benchmark_mode" \
  -e class "$test_class" \
  "$test_package_name/$runner_name" 2>&1 |
  tee "$output_directory/instrumentation.txt"
pipeline_status=("${PIPESTATUS[@]}")
set -e
[[ "${pipeline_status[1]}" -eq 0 ]] || fail "Could not save instrumentation output."

device_report="cache/gpu-int8-fc/int8-fc-$run_id.json"
for _ in $(seq 1 60); do
  if adb_command shell run-as "$package_name" test -f "$device_report"; then
    break
  fi
  sleep 1
done
adb_command exec-out run-as "$package_name" cat "$device_report" \
  >"$output_directory/report.json" ||
  fail "Could not retrieve the device report."
adb_command logcat -d >"$output_directory/logcat.txt" || true
grep -E "MLDRIFT_KERNEL|Benchmark mode|fully_connected|ConvWave" \
  "$output_directory/logcat.txt" >"$output_directory/kernel-selection.txt" || true

while IFS= read -r variant_name; do
  [[ "$variant_name" =~ ^[a-z0-9-]+$ ]] ||
    fail "Unsafe variant name in device report: $variant_name"
  device_cache="cache/gpu-int8-fc/program-cache/$variant_name/"\
"${variant_name}_mldrift_program_cache.bin"
  adb_command exec-out run-as "$package_name" cat "$device_cache" \
    >"$output_directory/${variant_name}_mldrift_program_cache.bin" ||
    fail "Could not retrieve GPU program cache for $variant_name."
done < <(
  python3 - "$output_directory/report.json" <<'PY'
import json
import sys

for variant in json.load(open(sys.argv[1], encoding="utf-8"))["variants"]:
    print(variant["name"])
PY
)

python3 "$project_dir/scripts/analyze-whisper-int8-fc-kernel-caches.py" \
  "$output_directory"

python3 - "$output_directory/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "completed":
    raise SystemExit(
        "device regression failed: %s: %s"
        % (report.get("error_type"), report.get("error_message"))
    )
for variant in report["variants"]:
    active = variant["active_row"]
    print(
        "%s storage=%s max_abs=%.9g mae=%.9g rmse=%.9g cosine=%.12g "
        "load_ms=%s inference_ms=%s cache_sha=%s"
        % (
            variant["name"],
            variant["buffer_storage_type"],
            active["maximum_absolute_error"],
            active["mean_absolute_error"],
            active["root_mean_square_error"],
            active["cosine_similarity"],
            variant["model_load_millis"],
            variant["inference_millis"],
            variant["program_cache_sha256"],
        )
    )
PY

[[ "${pipeline_status[0]}" -eq 0 ]] ||
  fail "Instrumentation exited with ${pipeline_status[0]}."
echo "Saved Whisper INT8 FC report: $output_directory"
