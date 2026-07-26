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
prefix_only=false
targeted_ranges_only=false
rmse_threshold="0"
adb_arguments=()
device_directory=""

usage() {
  printf '%s\n' \
    "Usage: scripts/run-litert-main-whisper-gpu-op-bisect.sh [options]" \
    "" \
    "Builds a TFLITE_DEBUG_DELEGATE probe, compares CPU/GPU operation-family" \
    "fallbacks, then finds the smallest divergent delegated node prefix." \
    "" \
    "The probe runs only from a dedicated /data/local/tmp directory. It never" \
    "installs, uninstalls, opens, or modifies Voice Memo or its recordings." \
    "" \
    "Options:" \
    "  --serial SERIAL       Specific authorized Android device." \
    "  --output DIRECTORY    New host report directory." \
    "  --no-build            Reuse an existing debug probe binary." \
    "  --prefix-only         Skip operation-family cases." \
    "  --targeted-ranges-only" \
    "                        Run focused attention-node ranges only." \
    "  --rmse-threshold N    Treat step-0 RMSE <= N as non-divergent." \
    "  -h, --help            Show this help."
}

fail() {
  echo "Whisper GPU operation bisect failed: $*" >&2
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
  [[ "$device_directory" =~ ^/data/local/tmp/voicememo-whisper-gpu-bisect-[A-Za-z0-9-]+$ ]] ||
    return 0
  adb_command shell rm -f \
    "$device_directory/whisper_gpu_probe" \
    "$device_directory/reference.tflite" \
    "$device_directory/candidate.tflite" >/dev/null 2>&1 || true
  adb_command shell rmdir "$device_directory" >/dev/null 2>&1 || true
}

run_case() {
  local label="$1"
  local excluded_builtins="$2"
  local first_node="$3"
  local last_node="$4"
  local print_operations="$5"
  local report="$output_directory/$label.txt"
  local pipeline_status
  [[ "$label" =~ ^[a-z0-9-]+$ ]] || fail "Unsafe case label: $label"
  [[ ! -e "$report" ]] || fail "Case report already exists: $report"

  set +e
  adb_command shell env \
    "VOICEMEMO_GPU_EXCLUDED_BUILTINS=$excluded_builtins" \
    "VOICEMEMO_PRINT_OPS=$print_operations" \
    timeout 1800 \
    "$device_directory/whisper_gpu_probe" \
    "$device_directory/candidate.tflite" \
    "$device_directory/reference.tflite" \
    "$first_node" \
    "$last_node" 2>&1 |
    tee "$report"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  [[ "${pipeline_status[1]}" -eq 0 ]] || fail "Could not save $label output."
  printf '%s\n' "${pipeline_status[0]}" > "$output_directory/$label.exit-code"
  if [[ "${pipeline_status[0]}" -ne 0 ]]; then
    echo "$label exited with ${pipeline_status[0]}; retaining diagnostics." >&2
    return 1
  fi
}

run_required_case() {
  local label="$1"
  shift
  run_case "$label" "$@" || fail "Required case failed: $label"
}

run_optional_case() {
  local label="$1"
  shift
  if ! run_case "$label" "$@"; then
    echo "Optional case could not run: $label" >&2
  fi
}

prefix_report() {
  local last_node="$1"
  printf '%s/prefix-000-%03d.txt' "$output_directory" "$last_node"
}

run_prefix() {
  local last_node="$1"
  local report
  report="$(prefix_report "$last_node")"
  if [[ ! -e "$report" ]]; then
    run_case "$(printf 'prefix-000-%03d' "$last_node")" \
      "" 0 "$last_node" 0
  fi
}

prefix_is_divergent() {
  local last_node="$1"
  local report
  local exact_elements
  local rmse
  report="$(prefix_report "$last_node")"
  exact_elements="$(
    awk '
      /^step=0 / {
        for (field = 1; field <= NF; field++) {
          if ($field ~ /^gpu_exact_elements=/) {
            split($field, value, "=")
            print value[2]
            exit
          }
        }
      }
    ' "$report"
  )"
  [[ "$exact_elements" =~ ^[0-9]+$ ]] ||
    fail "Could not read exact-element count from $report"
  if [[ "$rmse_threshold" == "0" ]]; then
    [[ "$exact_elements" -lt 1280 ]]
    return
  fi
  rmse="$(
    awk '
      /^step=0 / {
        for (field = 1; field <= NF; field++) {
          if ($field ~ /^gpu_rmse=/) {
            split($field, value, "=")
            print value[2]
            exit
          }
        }
      }
    ' "$report"
  )"
  [[ "$rmse" =~ ^[0-9.eE+-]+$ ]] ||
    fail "Could not read RMSE from $report"
  awk -v actual="$rmse" -v threshold="$rmse_threshold" \
    'BEGIN { exit !(actual > threshold) }'
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
    --prefix-only)
      prefix_only=true
      shift
      ;;
    --targeted-ranges-only)
      targeted_ranges_only=true
      shift
      ;;
    --rmse-threshold)
      [[ "$#" -ge 2 ]] || fail "--rmse-threshold requires a value."
      rmse_threshold="$2"
      awk -v threshold="$rmse_threshold" \
        'BEGIN { exit !(threshold >= 0) }' ||
        fail "--rmse-threshold must be non-negative."
      shift 2
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
[[ -f "$reference_model" ]] || fail "CPU reference model is missing."
[[ -f "$candidate_model" ]] || fail "GPU candidate model is missing."
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
        --copt=-DTFLITE_DEBUG_DELEGATE \
        --jobs=6 \
        //voicememo_probe:whisper_gpu_probe
  )
fi

[[ -x "$probe_binary" ]] || fail "Source-built debug probe is missing."
resolve_device

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
[[ "$run_id" =~ ^[A-Za-z0-9-]+$ ]] || fail "Could not create a safe run ID."
device_directory="/data/local/tmp/voicememo-whisper-gpu-bisect-$run_id"
if [[ -z "$output_directory" ]]; then
  output_directory="$project_dir/build/reports/device-gpu/whisper-gpu-op-bisect-$run_id"
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

# Builtin groups:
# 25 SOFTMAX; 40 MEAN; 74 SUM; 76 RSQRT; 81 REDUCE_PROD;
# 82 REDUCE_MAX; 89 REDUCE_MIN; 91 REDUCE_ANY; 99 SQUARED_DIFFERENCE.
# 3 CONV_2D; 9 FULLY_CONNECTED; 126 BATCH_MATMUL.
if [[ "$targeted_ranges_only" == true ]]; then
  run_required_case projections-and-layout-020-027 "" 20 27 0
  run_optional_case attention-score-batch-matmul-028 "" 28 28 0
  run_required_case attention-score-path-020-028 "" 20 28 0
  run_optional_case attention-value-batch-matmul-036 "" 36 36 0
  run_optional_case cross-attention-score-batch-matmul-056 "" 56 56 0
  run_optional_case cross-attention-value-batch-matmul-059 "" 59 59 0
  run_optional_case block-2-attention-score-batch-matmul-105 "" 105 105 0
  run_optional_case block-2-attention-value-batch-matmul-113 "" 113 113 0
  run_optional_case block-2-cross-score-batch-matmul-133 "" 133 133 0
  run_optional_case block-2-cross-value-batch-matmul-136 "" 136 136 0
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
    printf 'mode=targeted_ranges\n'
  } > "$output_directory/metadata.txt"
  echo "Saved targeted Whisper GPU ranges: $output_directory"
  exit 0
fi

if [[ "$prefix_only" != true ]]; then
  run_required_case baseline "" 0 155 1
  run_required_case normalization-reductions-cpu "40,76,99" 0 155 0
  run_required_case softmax-cpu "25" 0 155 0
  run_required_case all-reductions-cpu \
    "25,40,74,76,81,82,89,91,99" 0 155 0
  run_required_case fully-connected-cpu "9" 0 155 0
  run_optional_case batch-matmul-cpu "126" 0 155 0
  run_optional_case all-projections-cpu "3,9,126" 0 155 0
fi

# Find the smallest delegated prefix whose final output is no longer bit-exact
# with CPU. This is a diagnostic boundary, not a proof that later nodes cannot
# dampen or amplify an earlier difference.
run_prefix 0
run_prefix 155
if prefix_is_divergent 0; then
  first_divergent_node=0
else
  lower_exact=0
  upper_divergent=155
  if ! prefix_is_divergent "$upper_divergent"; then
    fail "The full GPU prefix unexpectedly remained bit-exact."
  fi
  while (( upper_divergent - lower_exact > 1 )); do
    midpoint=$(( (lower_exact + upper_divergent) / 2 ))
    run_prefix "$midpoint"
    if prefix_is_divergent "$midpoint"; then
      upper_divergent="$midpoint"
    else
      lower_exact="$midpoint"
    fi
  done
  first_divergent_node="$upper_divergent"
fi
run_prefix "$first_divergent_node"

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
  printf 'rmse_threshold=%s\n' "$rmse_threshold"
  printf 'first_divergent_prefix_node=%s\n' "$first_divergent_node"
} > "$output_directory/metadata.txt"

echo "Saved Whisper GPU operation bisect: $output_directory"
