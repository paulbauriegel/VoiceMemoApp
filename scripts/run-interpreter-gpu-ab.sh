#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
probe_dir="$project_dir/tools/interpreter-gpu-probe"
probe_package="com.example.voicememo.interpretergpuprobe"
test_package="$probe_package.test"
test_class="$probe_package.InterpreterGpuAbTest"
runner="androidx.test.runner.AndroidJUnitRunner"
voice_memo_package="com.example.voicememo"
model_asset="whisper/whisper_large_v3_turbo_30s_i8_explicit_key_transpose.tflite"
host_model_file=""
expected_model_sha256=""
adb_binary="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
device_serial=""
output_directory=""
skip_build=false
adb_arguments=()
device_directory=""
private_model_name=""

usage() {
  cat <<'EOF'
Usage: scripts/run-interpreter-gpu-ab.sh [options]

Builds and installs a separate diagnostic package, then compares LiteRT's
standard Interpreter CPU path with its OpenCL GPU delegate for three
teacher-forced decoder steps. The probe opens Voice Memo's already-installed
uncompressed model read-only; it never reads or changes recordings,
transcripts, jobs, or app data.

Options:
  --serial SERIAL       Specific authorized Android device.
  --model-asset PATH    Voice Memo asset path to open.
  --model-file FILE     Stage a host model in the separate probe package.
  --model-sha HASH      Required SHA-256 for --model-file.
  --output DIRECTORY    New host report directory.
  --no-build            Reuse existing probe APKs.
  -h, --help            Show this help.

The probe uses the separately published LiteRT Interpreter/GPU 1.4.2 line.
LiteRT 2.1.x's Maven Interpreter is CPU-only and has no public addDelegate API.
Voice Memo is never reinstalled or uninstalled by this script.
EOF
}

fail() {
  echo "Voice Memo Interpreter GPU A/B failed: $*" >&2
  exit 1
}

adb_command() {
  "$adb_binary" "${adb_arguments[@]}" "$@"
}

cleanup_staged_model() {
  if [[ -n "$private_model_name" ]]; then
    adb_command shell run-as "$probe_package" rm -f \
      "files/$private_model_name" >/dev/null 2>&1 || true
  fi
  if [[ "$device_directory" =~ ^/data/local/tmp/voicememo-interpreter-gpu-[A-Za-z0-9-]+$ ]]; then
    adb_command shell rm -f "$device_directory/model.tflite" \
      >/dev/null 2>&1 || true
    adb_command shell rmdir "$device_directory" >/dev/null 2>&1 || true
  fi
}

resolve_device() {
  local serial
  local state
  local ignored
  local connected=()
  if [[ -n "$device_serial" ]]; then
    adb_arguments=(-s "$device_serial")
    [[ "$(adb_command get-state 2>/dev/null || true)" == "device" ]] ||
      fail "Device $device_serial is not connected and authorized."
    return
  fi
  while IFS=$'\t' read -r serial state ignored; do
    if [[ -n "$serial" && "$state" == "device" ]]; then
      connected+=("$serial")
    fi
  done < <("$adb_binary" devices)
  [[ "${#connected[@]}" -eq 1 ]] ||
    fail "Expected exactly one authorized device; pass --serial when needed."
  device_serial="${connected[0]}"
  adb_arguments=(-s "$device_serial")
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --serial)
      [[ "$#" -ge 2 ]] || fail "--serial requires a value."
      device_serial="$2"
      shift 2
      ;;
    --model-asset)
      [[ "$#" -ge 2 ]] || fail "--model-asset requires a value."
      model_asset="$2"
      shift 2
      ;;
    --model-file)
      [[ "$#" -ge 2 ]] || fail "--model-file requires a value."
      host_model_file="$2"
      shift 2
      ;;
    --model-sha)
      [[ "$#" -ge 2 ]] || fail "--model-sha requires a value."
      expected_model_sha256="$2"
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
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

command -v "$adb_binary" >/dev/null 2>&1 || fail "adb not found: $adb_binary"
command -v python3 >/dev/null 2>&1 || fail "python3 not found."
[[ "$model_asset" =~ ^[A-Za-z0-9_./-]+$ ]] || fail "Invalid model asset path."
if [[ -n "$host_model_file" ]]; then
  [[ -f "$host_model_file" ]] || fail "Model file not found: $host_model_file"
  [[ "$expected_model_sha256" =~ ^[0-9a-f]{64}$ ]] ||
    fail "--model-file requires --model-sha with 64 lowercase hex characters."
  [[ "$(shasum -a 256 "$host_model_file" | awk '{print $1}')" == \
      "$expected_model_sha256" ]] ||
    fail "Host model SHA-256 mismatch."
fi

resolve_device
if [[ -z "$host_model_file" ]]; then
  adb_command shell pm path "$voice_memo_package" >/dev/null ||
    fail "Voice Memo is not installed on $device_serial."
fi

app_apk="$probe_dir/app/build/outputs/apk/debug/app-debug.apk"
test_apk="$probe_dir/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
if [[ "$skip_build" != true ]]; then
  java_home="${JAVA_HOME:-/Applications/Android Studio.app/Contents/jbr/Contents/Home}"
  android_home="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
  JAVA_HOME="$java_home" \
    ANDROID_HOME="$android_home" \
    PATH="$java_home/bin:$android_home/platform-tools:$PATH" \
    "$project_dir/gradlew" --no-daemon -p "$probe_dir" \
      :app:assembleDebug :app:assembleDebugAndroidTest
fi
[[ -f "$app_apk" ]] || fail "Probe APK not found: $app_apk"
[[ -f "$test_apk" ]] || fail "Probe test APK not found: $test_apk"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
[[ "$run_id" =~ ^[A-Za-z0-9-]+$ ]] || fail "Could not create a safe run ID."
if [[ -z "$output_directory" ]]; then
  output_directory="$project_dir/build/reports/device-gpu/interpreter-ab-$run_id"
fi
[[ ! -e "$output_directory" ]] || fail "Output already exists: $output_directory"
mkdir -p "$output_directory"
trap cleanup_staged_model EXIT

echo "Installing only the separate probe package in place."
adb_command install -r "$app_apk"
adb_command install -r "$test_apk"
instrument_model_arguments=()
if [[ -n "$host_model_file" ]]; then
  private_model_name="candidate-$run_id.tflite"
  device_directory="/data/local/tmp/voicememo-interpreter-gpu-$run_id"
  adb_command shell mkdir -p "$device_directory"
  adb_command push "$host_model_file" "$device_directory/model.tflite"
  adb_command shell chmod 644 "$device_directory/model.tflite"
  adb_command shell run-as "$probe_package" mkdir -p files
  adb_command shell run-as "$probe_package" cp \
    "$device_directory/model.tflite" "files/$private_model_name"
  device_model_sha="$(
    adb_command shell run-as "$probe_package" sha256sum \
      "files/$private_model_name" | tr -d '\r' | awk '{print $1}'
  )"
  [[ "$device_model_sha" == "$expected_model_sha256" ]] ||
    fail "Probe-private model SHA-256 mismatch."
  instrument_model_arguments=(-e modelFile "$private_model_name")
fi
adb_command logcat -c

echo "Running standard Interpreter CPU/OpenCL A/B: $run_id"
set +e
adb_command shell am instrument -w -r \
  -e runId "$run_id" \
  -e modelAsset "$model_asset" \
  "${instrument_model_arguments[@]}" \
  -e class "$test_class" \
  "$test_package/$runner" 2>&1 |
  tee "$output_directory/instrumentation.txt"
pipeline_status=("${PIPESTATUS[@]}")
set -e
instrument_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
[[ "$tee_status" -eq 0 ]] || fail "Could not save instrumentation output."

adb_command logcat -d |
  python3 -c '
import re, sys
pattern = re.compile(r"VoiceMemoInterpGpu|TfLite|TFLite|GpuDelegate|OpenCL")
for line in sys.stdin:
    if pattern.search(line):
        sys.stdout.write(line)
' > "$output_directory/logcat.txt"

device_report="cache/interpreter-gpu-ab/$run_id.json"
adb_command exec-out run-as "$probe_package" cat "$device_report" \
  > "$output_directory/report.json" ||
  fail "Could not read probe report $device_report (instrumentation status $instrument_status)."

python3 -c '
import json, sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("run_id") != sys.argv[2]:
    raise SystemExit("Interpreter GPU A/B run ID mismatch.")
if report.get("status") != "completed":
    raise SystemExit(
        "Probe failed: %s: %s"
        % (report.get("error_type"), report.get("error_message"))
    )
expected_sha = sys.argv[3]
if expected_sha and report.get("model_sha256") != expected_sha:
    raise SystemExit("Probe report model SHA-256 mismatch.")
gpu = report["gpu"]
print(
    "CPU: encode_ms=%s plan=%s"
    % (report["cpu"]["encode_millis"], report["cpu"]["execution_plan_length"])
)
print(
    "GPU: status=%s load_ms=%s plan=%s full_decode_delegation=%s"
    % (
        gpu.get("status"),
        gpu.get("model_load_millis"),
        gpu.get("delegated_execution_plan_length"),
        gpu.get("full_decode_delegation_indicated"),
    )
)
encoder = gpu.get("encoder_state", {})
if encoder:
    print(
        "GPU encoder: rmse=%s cosine=%s gpu_ms=%s"
        % (
            encoder.get("root_mean_square_error"),
            encoder.get("cosine_similarity"),
            encoder.get("gpu_inference_millis"),
        )
    )
for step in gpu.get("teacher_forced_steps", []):
    print(
        "step_%s: top1_match=%s cpu=%s gpu=%s rmse=%s cosine=%s gpu_ms=%s"
        % (
            step["step"],
            step["top1_match"],
            step["cpu_top1_token_id"],
            step["gpu_top1_token_id"],
            step["root_mean_square_error"],
            step["cosine_similarity"],
            step["gpu_inference_millis"],
        )
    )
if gpu.get("status") != "completed":
    raise SystemExit(
        "GPU candidate failed: %s: %s"
        % (gpu.get("error_type"), gpu.get("error_message"))
    )
' "$output_directory/report.json" "$run_id" "$expected_model_sha256"

[[ "$instrument_status" -eq 0 ]] ||
  fail "Instrumentation returned status $instrument_status after writing its report."
echo "Saved report: $output_directory/report.json"
