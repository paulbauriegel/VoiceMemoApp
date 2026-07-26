#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
litert_dir="$project_dir/build/third_party/litert-main"
bazel_binary="$project_dir/build/tools/bazel"
bazel_cache="$project_dir/build/tools/bazel-cache"
probe_binary="$litert_dir/bazel-bin/tflite/delegates/gpu/cl/testing/delegate_testing"
gpu_library="$litert_dir/bazel-bin/tflite/delegates/gpu/libtensorflowlite_gpu_delegate.so"
int4_model="$project_dir/build/generated/whisper-decoder-split/whisper_large_v3_turbo_decode_shared_after_2_blocks.tflite"
float_model="$project_dir/build/generated/whisper-float-decoder-probe/whisper_large_v3_turbo_decode_float32_after_2_blocks.tflite"
adb_binary="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
expected_litert_commit="1921f3defc8413a3a6bd23ad6a4d5fe35520a2c0"
expected_int4_sha256="f5926743061104e49dd12f532fe3d35a3025f791bc1ec75c4e2a0b7eef136337"
expected_float_sha256="0dd5a7645060263c8b10a88f10e99b991b2ab592cfe098d509945fdd6331a08d"
device_serial=""
output_directory=""
skip_build=false
adb_arguments=()
device_directory=""

usage() {
  cat <<'EOF'
Usage: scripts/run-litert-main-interpreter-gpu-probe.sh [options]

Builds the standard Interpreter CPU/OpenCL comparison tool and GPU delegate
from the pinned public LiteRT main checkout, then runs the two-block Whisper
INT4 and FLOAT32 controls on one authorized arm64 Android device.

This is a native /data/local/tmp diagnostic. It never installs, uninstalls,
reinstalls, opens, or modifies the Voice Memo app or its private recordings.

Options:
  --serial SERIAL       Specific authorized Android device.
  --output DIRECTORY    New host report directory.
  --no-build            Reuse the existing source-built probe binary.
  -h, --help            Show this help.
EOF
}

fail() {
  echo "Current LiteRT Interpreter/GPU probe failed: $*" >&2
  exit 1
}

host_sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
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

cleanup_device_files() {
  [[ "$device_directory" =~ ^/data/local/tmp/voicememo-litert-main-[A-Za-z0-9-]+$ ]] ||
    return 0
  adb_command shell rm -f \
    "$device_directory/delegate_testing" \
    "$device_directory/int4.tflite" \
    "$device_directory/float32.tflite" >/dev/null 2>&1 || true
  adb_command shell rmdir "$device_directory" >/dev/null 2>&1 || true
}

run_variant() {
  local label="$1"
  local device_model="$2"
  local output_file="$output_directory/$label.txt"
  local pipeline_status
  local command_status

  echo "Running current-main $label CPU/OpenCL comparison."
  set +e
  adb_command shell timeout 1800 \
    "$device_directory/delegate_testing" "$device_directory/$device_model" 2>&1 |
    tee "$output_file"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  command_status="${pipeline_status[0]}"
  [[ "${pipeline_status[1]}" -eq 0 ]] ||
    fail "Could not save $label probe output."
  printf '%s\n' "$command_status" > "$output_directory/$label.exit-code"
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
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

command -v "$adb_binary" >/dev/null 2>&1 || fail "adb not found: $adb_binary"
command -v shasum >/dev/null 2>&1 || fail "shasum is required."
[[ -d "$litert_dir/.git" ]] || fail "Pinned LiteRT checkout is missing: $litert_dir"
[[ "$(git -C "$litert_dir" rev-parse HEAD)" == "$expected_litert_commit" ]] ||
  fail "LiteRT checkout is not at pinned commit $expected_litert_commit."
[[ -f "$int4_model" ]] || fail "INT4 decoder control is missing: $int4_model"
[[ -f "$float_model" ]] || fail "FLOAT32 decoder control is missing: $float_model"
[[ "$(host_sha256 "$int4_model")" == "$expected_int4_sha256" ]] ||
  fail "INT4 decoder control SHA-256 mismatch."
[[ "$(host_sha256 "$float_model")" == "$expected_float_sha256" ]] ||
  fail "FLOAT32 decoder control SHA-256 mismatch."

if [[ "$skip_build" != true ]]; then
  [[ -x "$bazel_binary" ]] || fail "Bazelisk launcher is missing: $bazel_binary"
  PATH="$project_dir/build/tools:$PATH" \
    BAZELISK_HOME="$bazel_cache" \
    "$bazel_binary" build \
      --config=android_arm64 \
      --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
      --define=tflite_with_xnnpack=false \
      --jobs=6 \
      //tflite/delegates/gpu/cl/testing:delegate_testing \
      //tflite/delegates/gpu:libtensorflowlite_gpu_delegate.so
fi

[[ -x "$probe_binary" ]] || fail "Source-built probe is missing: $probe_binary"
[[ -f "$gpu_library" ]] || fail "Source-built GPU library is missing: $gpu_library"
resolve_device

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
[[ "$run_id" =~ ^[A-Za-z0-9-]+$ ]] || fail "Could not create a safe run ID."
device_directory="/data/local/tmp/voicememo-litert-main-$run_id"
if [[ -z "$output_directory" ]]; then
  output_directory="$project_dir/build/reports/device-gpu/litert-main-interpreter-$run_id"
fi
[[ ! -e "$output_directory" ]] || fail "Output already exists: $output_directory"
mkdir -p "$output_directory"
trap cleanup_device_files EXIT

adb_command shell mkdir -p "$device_directory"
adb_command push "$probe_binary" "$device_directory/delegate_testing"
adb_command shell chmod 700 "$device_directory/delegate_testing"
adb_command push "$int4_model" "$device_directory/int4.tflite"
adb_command push "$float_model" "$device_directory/float32.tflite"

device_int4_sha="$(adb_command shell sha256sum "$device_directory/int4.tflite" | tr -d '\r' | awk '{print $1}')"
device_float_sha="$(adb_command shell sha256sum "$device_directory/float32.tflite" | tr -d '\r' | awk '{print $1}')"
[[ "$device_int4_sha" == "$expected_int4_sha256" ]] ||
  fail "Device INT4 SHA-256 mismatch."
[[ "$device_float_sha" == "$expected_float_sha256" ]] ||
  fail "Device FLOAT32 SHA-256 mismatch."

adb_command logcat -c
run_variant "int4" "int4.tflite"
run_variant "float32" "float32.tflite"
adb_command logcat -d -s TfLite GPUDelegate OpenCL \
  > "$output_directory/logcat.txt" || true

{
  printf 'litert_commit=%s\n' "$expected_litert_commit"
  printf 'device_serial=%s\n' "$device_serial"
  printf 'device_model=%s\n' "$(adb_command shell getprop ro.product.model | tr -d '\r')"
  printf 'device_build=%s\n' "$(adb_command shell getprop ro.build.fingerprint | tr -d '\r')"
  printf 'probe_sha256=%s\n' "$(host_sha256 "$probe_binary")"
  printf 'gpu_library_sha256=%s\n' "$(host_sha256 "$gpu_library")"
  printf 'int4_sha256=%s\n' "$expected_int4_sha256"
  printf 'float32_sha256=%s\n' "$expected_float_sha256"
} > "$output_directory/metadata.txt"

echo "Saved current-main CPU/OpenCL report: $output_directory"
