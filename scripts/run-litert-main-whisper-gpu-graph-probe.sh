#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
litert_dir="$project_dir/build/third_party/litert-main"
bazel_binary="$project_dir/build/tools/bazel"
bazel_cache="$project_dir/build/tools/bazel-cache"
probe_source_dir="$project_dir/tools/litert-main-whisper-probe"
probe_workspace_dir="$litert_dir/voicememo_probe"
probe_binary="$litert_dir/bazel-bin/voicememo_probe/whisper_gpu_probe"
reference_model="$project_dir/build/generated/whisper-float-decoder-probe/whisper_large_v3_turbo_decode_float32_after_2_blocks.tflite"
candidate_model="$project_dir/build/generated/whisper-gpu-graph/whisper_large_v3_turbo_decode_float32_aligned_residuals.tflite"
adb_binary="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
expected_litert_commit="1921f3defc8413a3a6bd23ad6a4d5fe35520a2c0"
expected_reference_sha256="0dd5a7645060263c8b10a88f10e99b991b2ab592cfe098d509945fdd6331a08d"
expected_candidate_sha256="b7d528f6b978bed123b30a24ed68a8c139eb9eabd75e57af43af78a82c7baeb0"
device_serial=""
output_directory=""
skip_build=false
adb_arguments=()
device_directory=""

usage() {
  cat <<'EOF'
Usage: scripts/run-litert-main-whisper-gpu-graph-probe.sh [options]

Runs the exact Whisper decoder fixture against:

1. The original FLOAT32 model on CPU.
2. The GPU-graph rewrite on CPU, proving arithmetic parity.
3. The GPU-graph rewrite with current LiteRT's OpenCL delegate.

This native diagnostic uses only a dedicated /data/local/tmp directory. It
never installs, uninstalls, opens, or modifies Voice Memo or its recordings.

Options:
  --serial SERIAL       Specific authorized Android device.
  --output DIRECTORY    New host report directory.
  --candidate FILE      Alternate pinned candidate model.
  --candidate-sha HASH  Required SHA-256 for --candidate.
  --no-build            Reuse the existing source-built probe binary.
  -h, --help            Show this help.
EOF
}

fail() {
  echo "Whisper GPU graph probe failed: $*" >&2
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
  [[ "$device_directory" =~ ^/data/local/tmp/voicememo-whisper-gpu-graph-[A-Za-z0-9-]+$ ]] ||
    return 0
  adb_command shell rm -f \
    "$device_directory/whisper_gpu_probe" \
    "$device_directory/reference.tflite" \
    "$device_directory/candidate.tflite" >/dev/null 2>&1 || true
  adb_command shell rmdir "$device_directory" >/dev/null 2>&1 || true
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
    --candidate)
      [[ "$#" -ge 2 ]] || fail "--candidate requires a value."
      candidate_model="$2"
      shift 2
      ;;
    --candidate-sha)
      [[ "$#" -ge 2 ]] || fail "--candidate-sha requires a value."
      expected_candidate_sha256="$2"
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
[[ "$expected_candidate_sha256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "Candidate SHA-256 must be 64 lowercase hexadecimal characters."
[[ -d "$litert_dir/.git" ]] || fail "Pinned LiteRT checkout is missing: $litert_dir"
[[ "$(git -C "$litert_dir" rev-parse HEAD)" == "$expected_litert_commit" ]] ||
  fail "LiteRT checkout is not at pinned commit $expected_litert_commit."
[[ -f "$reference_model" ]] || fail "CPU reference model is missing: $reference_model"
[[ -f "$candidate_model" ]] || fail "GPU candidate model is missing: $candidate_model"
[[ "$(host_sha256 "$reference_model")" == "$expected_reference_sha256" ]] ||
  fail "CPU reference model SHA-256 mismatch."
[[ "$(host_sha256 "$candidate_model")" == "$expected_candidate_sha256" ]] ||
  fail "GPU candidate model SHA-256 mismatch."

if [[ "$skip_build" != true ]]; then
  [[ -x "$bazel_binary" ]] || fail "Bazelisk launcher is missing: $bazel_binary"
  mkdir -p "$probe_workspace_dir"
  cp "$probe_source_dir/BUILD.bazel" "$probe_workspace_dir/BUILD.bazel"
  cp "$probe_source_dir/whisper_gpu_probe.cc" \
    "$probe_workspace_dir/whisper_gpu_probe.cc"
  (
    cd "$litert_dir"
    PATH="$project_dir/build/tools:$PATH" \
      BAZELISK_HOME="$bazel_cache" \
      "$bazel_binary" build \
        --config=android_arm64 \
        --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
        --define=tflite_with_xnnpack=false \
        --jobs=6 \
        //voicememo_probe:whisper_gpu_probe
  )
fi

[[ -x "$probe_binary" ]] || fail "Source-built exact probe is missing: $probe_binary"
resolve_device

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
[[ "$run_id" =~ ^[A-Za-z0-9-]+$ ]] || fail "Could not create a safe run ID."
device_directory="/data/local/tmp/voicememo-whisper-gpu-graph-$run_id"
if [[ -z "$output_directory" ]]; then
  output_directory="$project_dir/build/reports/device-gpu/whisper-gpu-graph-$run_id"
fi
[[ ! -e "$output_directory" ]] || fail "Output already exists: $output_directory"
mkdir -p "$output_directory"
trap cleanup_device_files EXIT

adb_command shell mkdir -p "$device_directory"
adb_command push "$probe_binary" "$device_directory/whisper_gpu_probe"
adb_command shell chmod 700 "$device_directory/whisper_gpu_probe"
adb_command push "$reference_model" "$device_directory/reference.tflite"
adb_command push "$candidate_model" "$device_directory/candidate.tflite"

device_reference_sha="$(
  adb_command shell sha256sum "$device_directory/reference.tflite" |
    tr -d '\r' | awk '{print $1}'
)"
device_candidate_sha="$(
  adb_command shell sha256sum "$device_directory/candidate.tflite" |
    tr -d '\r' | awk '{print $1}'
)"
[[ "$device_reference_sha" == "$expected_reference_sha256" ]] ||
  fail "Device CPU reference SHA-256 mismatch."
[[ "$device_candidate_sha" == "$expected_candidate_sha256" ]] ||
  fail "Device GPU candidate SHA-256 mismatch."

adb_command logcat -c
set +e
adb_command shell timeout 1800 \
  "$device_directory/whisper_gpu_probe" \
  "$device_directory/candidate.tflite" \
  "$device_directory/reference.tflite" 2>&1 |
  tee "$output_directory/probe.txt"
pipeline_status=("${PIPESTATUS[@]}")
set -e
[[ "${pipeline_status[1]}" -eq 0 ]] || fail "Could not save probe output."
printf '%s\n' "${pipeline_status[0]}" > "$output_directory/probe.exit-code"

adb_command logcat -d -s TfLite GPUDelegate OpenCL \
  > "$output_directory/logcat.txt" || true
{
  printf 'litert_commit=%s\n' "$expected_litert_commit"
  printf 'device_serial=%s\n' "$device_serial"
  printf 'device_model=%s\n' \
    "$(adb_command shell getprop ro.product.model | tr -d '\r')"
  printf 'device_build=%s\n' \
    "$(adb_command shell getprop ro.build.fingerprint | tr -d '\r')"
  printf 'probe_sha256=%s\n' "$(host_sha256 "$probe_binary")"
  printf 'reference_sha256=%s\n' "$expected_reference_sha256"
  printf 'candidate_sha256=%s\n' "$expected_candidate_sha256"
} > "$output_directory/metadata.txt"

[[ "${pipeline_status[0]}" -eq 0 ]] ||
  fail "Exact device probe exited with ${pipeline_status[0]}."
echo "Saved exact Whisper GPU graph report: $output_directory"
