#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
probe_dir="$project_dir/tools/interpreter-gpu-probe"
probe_package="com.example.voicememo.interpretergpuprobe"
test_package="$probe_package.test"
test_class="$probe_package.InterpreterGpuAbTest"
runner="androidx.test.runner.AndroidJUnitRunner"
int4_model="$project_dir/build/generated/whisper-decoder-split/whisper_large_v3_turbo_decode_shared_after_2_blocks.tflite"
float_model="$project_dir/build/generated/whisper-float-decoder-probe/whisper_large_v3_turbo_decode_float32_after_2_blocks.tflite"
adb_binary="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
device_serial=""
output_directory=""
skip_build=false
adb_arguments=()
device_model_name="decoder-control.tflite"
device_staging_path="/data/local/tmp/voicememo-decoder-control.tflite"

usage() {
  cat <<'EOF'
Usage: scripts/run-float-decoder-control.sh [options]

Runs the same deterministic two-decoder-block CPU/OpenCL comparison twice:
first with the pinned INT4 boundary and then with its exact FLOAT32-weight
control. Models are streamed only into the separate probe package's private
storage and removed afterward. Voice Memo is never reinstalled, uninstalled,
or read by this diagnostic.

Options:
  --serial SERIAL       Specific authorized Android device.
  --int4-model FILE     Pinned two-block boundary model.
  --float-model FILE    Generated FLOAT32 two-block control.
  --output DIRECTORY    New host report directory.
  --no-build            Reuse existing probe APKs.
  -h, --help            Show this help.
EOF
}

fail() {
  echo "Voice Memo FLOAT32 decoder control failed: $*" >&2
  exit 1
}

adb_command() {
  "$adb_binary" "${adb_arguments[@]}" "$@"
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

cleanup_device_model() {
  adb_command shell run-as "$probe_package" \
    rm -f "files/$device_model_name" >/dev/null 2>&1 || true
  adb_command shell rm -f "$device_staging_path" >/dev/null 2>&1 || true
}

host_sha256() {
  python3 -c '
import hashlib, sys
digest = hashlib.sha256()
with open(sys.argv[1], "rb") as source:
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
print(digest.hexdigest())
' "$1"
}

device_sha256() {
  local output
  output="$(adb_command shell run-as "$probe_package" \
    sha256sum "files/$device_model_name" | tr -d '\r')"
  echo "${output%% *}"
}

stream_model_to_probe() {
  local host_model="$1"
  local expected_size
  local actual_size
  local expected_sha
  local actual_sha
  expected_size="$(stat -f %z "$host_model")"
  expected_sha="$(host_sha256 "$host_model")"
  cleanup_device_model
  adb_command shell run-as "$probe_package" mkdir -p files
  echo "Transferring $(basename "$host_model") into the separate probe package."
  adb_command push "$host_model" "$device_staging_path"
  adb_command shell run-as "$probe_package" \
    cp "$device_staging_path" "files/$device_model_name"
  adb_command shell rm -f "$device_staging_path"
  actual_size="$(adb_command shell run-as "$probe_package" \
    stat -c %s "files/$device_model_name" | tr -d '\r')"
  [[ "$actual_size" == "$expected_size" ]] ||
    fail "Device model size mismatch: expected $expected_size, got $actual_size."
  actual_sha="$(device_sha256)"
  [[ "$actual_sha" == "$expected_sha" ]] ||
    fail "Device model SHA-256 mismatch."
}

run_variant() {
  local label="$1"
  local host_model="$2"
  local variant_directory="$output_directory/$label"
  local run_id="float-control-$label-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
  local instrument_status
  local device_report="cache/interpreter-gpu-ab/$run_id.json"
  local pipeline_status

  mkdir -p "$variant_directory"
  stream_model_to_probe "$host_model"
  adb_command logcat -c
  echo "Running $label two-block CPU/OpenCL control: $run_id"
  set +e
  adb_command shell am instrument -w -r \
    -e runId "$run_id" \
    -e mode boundary-after-two \
    -e modelFile "$device_model_name" \
    -e class "$test_class" \
    "$test_package/$runner" 2>&1 |
    tee "$variant_directory/instrumentation.txt"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  instrument_status="${pipeline_status[0]}"
  [[ "${pipeline_status[1]}" -eq 0 ]] ||
    fail "Could not save $label instrumentation output."

  adb_command logcat -d |
    python3 -c '
import re, sys
pattern = re.compile(r"VoiceMemoInterpGpu|TfLite|TFLite|GpuDelegate|OpenCL")
for line in sys.stdin:
    if pattern.search(line):
        sys.stdout.write(line)
' > "$variant_directory/logcat.txt"

  adb_command exec-out run-as "$probe_package" cat "$device_report" \
    > "$variant_directory/report.json" ||
    fail "Could not read $label report after instrumentation status $instrument_status."

  python3 -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    "%s outer_status=%s model_bytes=%s sha256=%s"
    % (
        sys.argv[2],
        report.get("status"),
        report.get("model_bytes"),
        report.get("model_sha256"),
    )
)
cpu = report.get("cpu", {})
gpu = report.get("gpu", {})
print(
    "CPU load_ms=%s plan=%s; GPU status=%s load_ms=%s plan=%s full=%s"
    % (
        cpu.get("model_load_millis"),
        cpu.get("execution_plan_length"),
        gpu.get("status"),
        gpu.get("model_load_millis"),
        gpu.get("delegated_execution_plan_length"),
        gpu.get("full_decode_delegation_indicated"),
    )
)
for step in gpu.get("active_hidden_steps", []):
    print(
        "step_%s argmax_match=%s cpu=%s gpu=%s rmse=%s cosine=%s gpu_ms=%s"
        % (
            step["step"],
            step["argmax_match"],
            step["cpu_argmax_index"],
            step["gpu_argmax_index"],
            step["root_mean_square_error"],
            step["cosine_similarity"],
            step["gpu_inference_millis"],
        )
    )
if report.get("status") != "completed":
    print(
        "outer_error=%s: %s"
        % (report.get("error_type"), report.get("error_message"))
    )
elif gpu.get("status") != "completed":
    print(
        "gpu_error=%s: %s"
        % (gpu.get("error_type"), gpu.get("error_message"))
    )
' "$variant_directory/report.json" "$label"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --serial)
      [[ "$#" -ge 2 ]] || fail "--serial requires a value."
      device_serial="$2"
      shift 2
      ;;
    --int4-model)
      [[ "$#" -ge 2 ]] || fail "--int4-model requires a value."
      int4_model="$2"
      shift 2
      ;;
    --float-model)
      [[ "$#" -ge 2 ]] || fail "--float-model requires a value."
      float_model="$2"
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
[[ -f "$int4_model" ]] || fail "INT4 boundary model is missing: $int4_model"
[[ -f "$float_model" ]] || fail "FLOAT32 control model is missing: $float_model"

resolve_device
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
if [[ -z "$output_directory" ]]; then
  output_directory="$project_dir/build/reports/device-gpu/float-decoder-control-$run_id"
fi
[[ ! -e "$output_directory" ]] || fail "Output already exists: $output_directory"
mkdir -p "$output_directory"

echo "Installing only the separate probe package in place."
adb_command install -r "$app_apk"
adb_command install -r "$test_apk"
trap cleanup_device_model EXIT

run_variant int4 "$int4_model"
run_variant float32 "$float_model"

python3 -c '
import base64, json, math, pathlib, struct, sys
root = pathlib.Path(sys.argv[1])
rows = {}
cpu_vectors = {}
for label in ("int4", "float32"):
    report = json.loads((root / label / "report.json").read_text())
    cpu = report.get("cpu", {})
    gpu = report.get("gpu", {})
    rows[label] = {
        "outer_status": report.get("status"),
        "gpu_status": gpu.get("status"),
        "model_bytes": report.get("model_bytes"),
        "model_sha256": report.get("model_sha256"),
        "cpu_plan": cpu.get("execution_plan_length"),
        "gpu_plan": gpu.get("delegated_execution_plan_length"),
        "full_delegation": gpu.get("full_decode_delegation_indicated"),
        "steps": gpu.get("active_hidden_steps", []),
        "error_type": gpu.get("error_type") or report.get("error_type"),
        "error_message": gpu.get("error_message") or report.get("error_message"),
    }
    encoded_rows = cpu.get("active_hidden_rows_float32_le_base64", [])
    cpu_vectors[label] = [
        struct.unpack("<%sf" % (len(raw) // 4), raw)
        for raw in map(base64.b64decode, encoded_rows)
    ]

comparisons = []
if len(cpu_vectors.get("int4", [])) == len(cpu_vectors.get("float32", [])):
    for step, (int4, float32) in enumerate(
        zip(cpu_vectors["int4"], cpu_vectors["float32"])
    ):
        if len(int4) != len(float32):
            continue
        differences = [actual - expected for expected, actual in zip(int4, float32)]
        squared_error = sum(value * value for value in differences)
        int4_squared = sum(value * value for value in int4)
        float_squared = sum(value * value for value in float32)
        dot = sum(expected * actual for expected, actual in zip(int4, float32))
        denominator = math.sqrt(int4_squared * float_squared)
        comparison = {
            "step": step,
            "element_count": len(int4),
            "max_absolute_error": max(map(abs, differences)),
            "mean_absolute_error": sum(map(abs, differences)) / len(differences),
            "root_mean_square_error": math.sqrt(squared_error / len(differences)),
            "cosine_similarity": dot / denominator if denominator else None,
        }
        comparisons.append(comparison)
        print(
            "cpu_model_parity_step_%s rmse=%s max_abs=%s cosine=%s"
            % (
                step,
                comparison["root_mean_square_error"],
                comparison["max_absolute_error"],
                comparison["cosine_similarity"],
            )
        )

summary = {
    "variants": rows,
    "cpu_int4_vs_float32": comparisons,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
' "$output_directory"

cleanup_device_model
trap - EXIT
echo "Saved comparison: $output_directory/summary.json"
