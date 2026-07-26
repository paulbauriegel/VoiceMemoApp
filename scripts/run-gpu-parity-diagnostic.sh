#!/usr/bin/env bash
set -euo pipefail

package_name="com.example.voicememo"
test_package_name="com.example.voicememo.test"
runner_name="androidx.test.runner.AndroidJUnitRunner"
test_class="com.example.voicememo.transcription.GpuParityDiagnosticTest"
whole_memo_test_class="com.example.voicememo.transcription.GpuWholeMemoWerBenchmarkTest"
rewritten_prefix_test_class="com.example.voicememo.transcription.RewrittenDecoderPrefixDiagnosticTest"
database_name="voice-memo.db"
adb_binary="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
sqlite_binary="${SQLITE3:-sqlite3}"
device_serial=""
memo_id=""
reference_job=""
chunk_index=0
hint_waiting_for_completion=false
kernel_batch_size=""
allow_source_quantization=true
constant_tensor_sharing=true
strict_gpu_only=false
gpu_backend="opencl"
buffer_storage="default"
prefer_texture_weights=""
benchmark_mode=false
run_autoregressive=true
teacher_forced_steps=""
live_cpu_reference=false
speculative_block_size=0
run_decoder_split=false
reverse_decoder_split=false
run_gpu_encoder_cpu_decoder=false
decoder_split_stage="post-layernorm"
gpu_profiles=""
decoder_quant_ab=false
app_apk="app/build/outputs/apk/debug/app-debug.apk"
test_apk="app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
output_directory=""
temporary_directory=""
diagnostic_run_id=""
report_wait_timeout_seconds=900
report_poll_interval_seconds=5
full_float_decoder_sha256="a2613d64e6cb73ff40db7c548215fd086f096aa088c6cc113c833ccc87501e89"
full_float_decoder_bytes=1779622096
adb_arguments=()
baseline_captured=false
memo_count_before=""
recording_manifest_before=""
transcript_state_before=""

preflight_decoder_split_apk() {
  local core_file
  local tail_file
  [[ "$run_decoder_split" == true ]] || return 0
  case "$decoder_split_stage" in
    post-layernorm)
      core_file="whisper_large_v3_turbo_30s_i4_decode_hidden.tflite"
      tail_file="whisper_large_v3_turbo_active_row_projection_i8.tflite"
      ;;
    pre-layernorm)
      core_file="whisper_large_v3_turbo_30s_i4_decode_pre_layer_norm.tflite"
      tail_file="whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite"
      ;;
    after-position-add)
      core_file="whisper_large_v3_turbo_decode_shared_after_position_add.tflite"
      tail_file="whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite"
      ;;
    after-1-block)
      core_file="whisper_large_v3_turbo_decode_shared_after_1_block.tflite"
      tail_file="whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite"
      ;;
    after-2-blocks)
      core_file="whisper_large_v3_turbo_decode_shared_after_2_blocks.tflite"
      tail_file="whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite"
      ;;
    after-3-blocks)
      core_file="whisper_large_v3_turbo_decode_shared_after_3_blocks.tflite"
      tail_file="whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite"
      ;;
    after-cross-attention)
      core_file="whisper_large_v3_turbo_decode_shared_after_cross_attention.tflite"
      tail_file="whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite"
      ;;
  esac
  python3 -c '
import sys, zipfile

apk = sys.argv[1]
with zipfile.ZipFile(apk) as archive:
    for file_name in sys.argv[2:]:
        entry = "assets/whisper-decoder-split/" + file_name
        try:
            info = archive.getinfo(entry)
        except KeyError:
            raise SystemExit("missing diagnostic APK entry: " + entry)
        if info.compress_type != zipfile.ZIP_STORED:
            raise SystemExit("diagnostic model must be uncompressed: " + entry)
' "$test_apk" "$core_file" "$tail_file" ||
    fail "Test APK does not contain the uncompressed $decoder_split_stage split assets."
}

preflight_rewritten_prefix_apk() {
  local expected_sha256="$1"
  local expected_bytes="$2"
  python3 -c '
import hashlib, sys, zipfile

apk, expected_sha256, expected_bytes = sys.argv[1:]
expected_bytes = int(expected_bytes)
entry = "assets/whisper-decoder-prefix/decoder-prefix.tflite"
with zipfile.ZipFile(apk) as archive:
    try:
        info = archive.getinfo(entry)
    except KeyError:
        raise SystemExit("missing diagnostic APK entry: " + entry)
    if info.compress_type != zipfile.ZIP_STORED:
        raise SystemExit("diagnostic model must be uncompressed: " + entry)
    if info.file_size != expected_bytes:
        raise SystemExit(
            "diagnostic model has %d bytes; expected %d"
            % (info.file_size, expected_bytes)
        )
    digest = hashlib.sha256()
    with archive.open(info) as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "diagnostic model SHA-256 is %s; expected %s"
            % (actual_sha256, expected_sha256)
        )
' "$test_apk" "$expected_sha256" "$expected_bytes" ||
    fail "Test APK does not contain the exact uncompressed rewritten decoder prefix."
}

preflight_full_float_decoder_apk() {
  python3 -c '
import hashlib, sys, zipfile

apk, expected_sha256, expected_bytes = sys.argv[1:]
expected_bytes = int(expected_bytes)
entry = "assets/whisper-full-float-decoder/decoder.tflite"
with zipfile.ZipFile(apk) as archive:
    try:
        info = archive.getinfo(entry)
    except KeyError:
        raise SystemExit("missing diagnostic APK entry: " + entry)
    if info.compress_type != zipfile.ZIP_STORED:
        raise SystemExit("full decoder model must be uncompressed")
    if info.file_size != expected_bytes:
        raise SystemExit(
            "full decoder model size mismatch: %s != %s"
            % (info.file_size, expected_bytes)
        )
    digest = hashlib.sha256()
    with archive.open(info) as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise SystemExit(
            "full decoder model SHA-256 mismatch: %s != %s"
            % (actual, expected_sha256)
        )
' "$test_apk" "$full_float_decoder_sha256" "$full_float_decoder_bytes" ||
    fail "Test APK does not contain the pinned uncompressed full FLOAT32 decoder."
}

usage() {
  cat <<'EOF'
Usage:
  scripts/run-gpu-parity-diagnostic.sh list [--serial SERIAL]

  scripts/run-gpu-parity-diagnostic.sh install \
    [--serial SERIAL] [--app-apk PATH]

  scripts/run-gpu-parity-diagnostic.sh whole \
    --memo-id ID [--serial SERIAL] [--output DIRECTORY]

  scripts/run-gpu-parity-diagnostic.sh prefix \
    --memo-id ID --reference-job ID --prefix-stage NAME \
    --prefix-sha256 SHA256 --prefix-bytes BYTES [options]

  scripts/run-gpu-parity-diagnostic.sh run \
    --memo-id ID --reference-job ID [options]

The read-only list command prints recent memo metadata and the latest trusted
CPU/XNNPACK reference job, when one exists.

The whole command creates a non-persisting live CPU reference and compares
whole-memo WER and latency for the viable CPU/GPU execution boundaries.
Pass --promotion-gate with --full-float-decoder to run only the CPU reference
and the rewritten full-GPU candidate with source quantization disabled.

The prefix command compares CPU and GPU activations at one generated boundary
of the rewritten production decoder or its diagnostic first-block FLOAT32-K
variant. It never decodes or persists text.

Installs the debug app and instrumentation APK in place, then compares:
  - CPU/XNNPACK encoder state with LiteRT GPU encoder state;
  - GPU decoder tokens when fed the CPU encoder state;
  - GPU decoder tokens when fed the GPU encoder state.

The diagnostic reads an existing successful CPU job as its token oracle. It
does not create, update, cancel, or delete transcription jobs and never prints
transcript text.

Options:
  --memo-id ID          Existing memo to read (required).
  --reference-job ID    Successful job for that memo (required).
  --prefix-stage NAME   Exact stage used to generate decoder-prefix.tflite.
  --prefix-sha256 HASH  Exact generated prefix-model SHA-256.
  --prefix-bytes BYTES  Exact generated prefix-model byte count.
  --model-variant NAME  int8-k, first-block-float-k,
                        first-block-float-self-attention, or
                        first-block-float-self-attention-mask-rank-reshape
                        (default: int8-k).
  --prefix-steps CSV    Teacher-forced token steps (default: 0,7).
  --reuse-installed-app Skip the 1 GB app install only when the installed
                        base APK SHA-256 exactly matches --app-apk.
  --isolation-id ID     Stable ID shared by a captured K run and isolated BMM.
  --capture-isolated-key
                        Save CPU K for a following isolated score-BMM run.
  --run-isolated-score  Compare CPU/GPU adj_y=false BMM with identical CPU Q/K.
  --capture-isolated-value
                        Save CPU V for a following isolated value-BMM run.
  --run-isolated-value  Compare CPU/GPU attention-value BMM with identical
                        CPU probabilities and V.
  --full-float-decoder  Whole-memo mode: use the complete FLOAT32 decoder
                        with all four causal-mask rank barriers.
  --promotion-gate      Whole-memo mode: run only CPU and the complete
                        full-GPU/no-source-quant production candidate.
  --chunk-index N       Chunk to test (default: 0).
  --mali-wait           Enable LiteRT's diagnostic OpenCL/Mali completion hint.
  --kernel-batch-size N Diagnostic OpenCL kernel flush batch size.
  --no-source-quant     Disable lossy GPU source activation quantization.
  --no-constant-sharing
                        Disable constant-tensor sharing for a decoder-split
                        GPU core. Requires --decoder-split.
  --strict-gpu-only     Compile a decoder-split core without CPU fallback.
                        Requires --decoder-split.
  --gpu-backend NAME    opencl, automatic, webgpu, or opengl (default: opencl).
  --buffer-storage NAME default, buffer, or texture2d.
  --prefer-texture-weights
                        Prefer texture-backed constant weights.
  --prefer-buffer-weights
                        Prefer buffer-backed constant weights.
  --benchmark-mode      Enable LiteRT's diagnostic benchmark mode.
  --teacher-forced-only Skip full autoregressive decoding for a fast probe.
  --teacher-forced-steps N|all
                        Compare the first N reference-token logits, or every
                        token in the selected reference chunk.
  --live-cpu-reference Generate the token oracle with the current CPU engine.
                        The persisted job supplies chunk/language metadata.
  --decoder-quant-ab    Run the controlled production-decoder comparison:
                        current GPU source quantization versus disabled source
                        quantization, using a live CPU oracle, every
                        teacher-forced token, and the same CPU encoder state.
  --speculative-block-size N
                        CPU-verify GPU proposal blocks of 1..16 tokens, or
                        1..32 with --decoder-split and stage after-1-block.
                        The boundary lane reuses its existing CPU side.
  --decoder-split       Test generated GPU decoder core plus CPU active-row
                        projection from the opt-in instrumentation APK.
  --reverse-decoder-split
                        Run a generated boundary with its prefix on CPU and
                        tail on GPU. Requires --decoder-split and a generated
                        after-* boundary stage.
  --gpu-encoder-cpu-decoder
                        Decode the GPU encoder state with CPU/XNNPACK after
                        closing the GPU model. Reports token IDs/hashes/timings
                        only and never persists transcript text.
  --decoder-split-stage NAME
                        post-layernorm, pre-layernorm, after-position-add,
                        after-1-block, after-2-blocks, after-3-blocks, or
                        after-cross-attention. after-position-add runs the
                        embedding/position prefix on GPU and decoder tail on
                        CPU; later boundary stages run the remaining decoder
                        operations on CPU. --reverse-decoder-split swaps the
                        prefix and tail backends. Final layer normalization and
                        projection stay on CPU.
  --profiles CSV        Run named profiles sequentially with one CPU oracle.
  --matrix              Run the complete predefined nine-profile matrix.
  --serial SERIAL       Specific adb device.
  --app-apk PATH        App APK installed with adb install -r.
  --test-apk PATH       Instrumentation APK installed with adb install -r.
  --output DIRECTORY    New result directory.
  -h, --help            Show this help.

Safety:
  The script refuses to start while recording/transcription is active. It
  verifies every active memo WAV and transcript/job/chunk row before and after
  the run.
  It never uninstalls VoiceMemo, clears app data, or changes a transcript.
EOF
}

fail() {
  echo "VoiceMemo GPU diagnostic failed: $*" >&2
  exit 1
}

is_generated_boundary_stage() {
  case "$1" in
    after-position-add | after-cross-attention | after-1-block | \
      after-2-blocks | after-3-blocks)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

cleanup() {
  local original_status=$?
  local postcondition_error=""
  local current_memo_count
  local current_recording_manifest
  local current_transcript_state
  trap - EXIT
  set +e
  if [[ "$baseline_captured" == true ]]; then
    if ! current_memo_count="$(memo_directory_count 2>/dev/null)"; then
      postcondition_error="Could not query the memo directory count for preservation verification."
    elif ! current_recording_manifest="$(recording_manifest_sha256 2>/dev/null)"; then
      postcondition_error="Could not hash memo audio files for preservation verification."
    elif ! current_transcript_state="$(transcript_state_sha256 2>/dev/null)"; then
      postcondition_error="Could not query transcript data for preservation verification."
    elif [[ -z "$current_memo_count" ||
            -z "$current_recording_manifest" ||
            -z "$current_transcript_state" ]]; then
      postcondition_error="VoiceMemo preservation verification returned an empty result."
    elif [[ "$current_memo_count" != "$memo_count_before" ]]; then
      postcondition_error="Memo directory count changed during the diagnostic."
    elif [[ "$current_recording_manifest" != "$recording_manifest_before" ]]; then
      postcondition_error="A memo audio file changed during the diagnostic."
    elif [[ "$current_transcript_state" != "$transcript_state_before" ]]; then
      postcondition_error="Transcript data changed during the diagnostic."
    fi
  fi
  if [[ -n "$temporary_directory" && -d "$temporary_directory" ]]; then
    rm -rf "$temporary_directory"
  fi
  if [[ -n "$postcondition_error" ]]; then
    echo "VoiceMemo GPU diagnostic failed: $postcondition_error" >&2
    original_status=1
  fi
  exit "$original_status"
}
trap cleanup EXIT

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

snapshot_database() {
  local destination="$1"
  mkdir -p "$destination" || return 1
  adb_command exec-out run-as "$package_name" cat "databases/$database_name" \
    >"$destination/$database_name" || return 1
  for suffix in -wal -shm; do
    if adb_command shell run-as "$package_name" \
      test -f "databases/$database_name$suffix"; then
      adb_command exec-out run-as "$package_name" \
        cat "databases/$database_name$suffix" \
        >"$destination/$database_name$suffix" || return 1
    fi
  done
}

database_query() {
  local query="$1"
  local snapshot="$temporary_directory/database-$RANDOM"
  snapshot_database "$snapshot" || return 1
  "$sqlite_binary" -readonly "$snapshot/$database_name" \
    "PRAGMA query_only=ON; $query" || return 1
}

memo_directory_count() {
  local directories
  directories="$(
    adb_command shell run-as "$package_name" \
      find files/memos -mindepth 1 -maxdepth 1 -type d
  )" || return 1
  printf '%s\n' "$directories" |
    tr -d '\r' |
    awk 'NF { count++ } END { print count + 0 }'
}

sha256_stream() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
  else
    sha256sum | awk '{print $1}'
  fi
}

audio_sha256() {
  local path="$1"
  adb_command exec-out run-as "$package_name" cat "$path" | sha256_stream
}

recording_manifest_sha256() {
  local database_rows
  local memo_id_from_database
  local audio_path
  local audio_hash
  local manifest=""
  database_rows="$(
    database_query \
      "SELECT id || '|' || audioPath FROM memos
       WHERE pendingDeletion=0 ORDER BY id;"
  )" || return 1
  while IFS='|' read -r memo_id_from_database audio_path; do
    [[ -n "$memo_id_from_database" && -n "$audio_path" ]] || continue
    audio_hash="$(audio_sha256 "$audio_path")" || return 1
    [[ -n "$audio_hash" ]] || return 1
    manifest+="$memo_id_from_database|$audio_hash"$'\n'
  done <<<"$database_rows"
  printf '%s' "$manifest" | sha256_stream
}

transcript_state_sha256() {
  local transcript_state
  transcript_state="$(
    database_query \
      "SELECT 'transcript|' || quote(memoId) || '|' ||
              quote(generatedText) || '|' || quote(editedText) || '|' ||
              quote(detectedLanguage) || '|' || quote(generatedAtMillis) || '|' ||
              quote(editedAtMillis) || '|' || quote(generatedRevision)
         FROM transcripts ORDER BY memoId;
       SELECT 'chunk|' || quote(jobId) || '|' || quote(chunkIndex) || '|' ||
              quote(memoId) || '|' || quote(windowStartMillis) || '|' ||
              quote(windowEndMillis) || '|' || quote(status) || '|' ||
              quote(rawText) || '|' || quote(tokenIds) || '|' ||
              quote(detectedLanguage) || '|' || quote(truncated) || '|' ||
              quote(warning) || '|' || quote(completedAtMillis)
         FROM transcript_chunks ORDER BY jobId, chunkIndex;
       SELECT 'job|' || quote(id) || '|' || quote(memoId) || '|' ||
              quote(workId) || '|' || quote(status) || '|' || quote(phase) || '|' ||
              quote(requestedLanguage) || '|' || quote(detectedLanguage) || '|' ||
              quote(totalChunks) || '|' || quote(completedChunks) || '|' ||
              quote(currentChunkIndex) || '|' || quote(currentToken) || '|' ||
              quote(currentTokenLimit) || '|' || quote(errorCode) || '|' ||
              quote(errorMessage) || '|' || quote(warning) || '|' ||
              quote(inferenceFingerprint) || '|' || quote(createdAtMillis) || '|' ||
              quote(updatedAtMillis)
         FROM transcription_jobs ORDER BY id;"
  )" || return 1
  printf '%s' "$transcript_state" | sha256_stream
}

wait_for_device_report() {
  local device_report="$1"
  local destination="$2"
  local partial_destination="${destination}.partial"
  local deadline=$((SECONDS + report_wait_timeout_seconds))
  local next_progress=$((SECONDS + 30))
  local device_state

  while (( SECONDS < deadline )); do
    if adb_command shell run-as "$package_name" \
      test -f "$device_report" >/dev/null 2>&1; then
      if adb_command exec-out run-as "$package_name" cat "$device_report" \
        >"$partial_destination" 2>/dev/null &&
        [[ -s "$partial_destination" ]]; then
        mv "$partial_destination" "$destination"
        return 0
      fi
    fi

    if (( SECONDS >= next_progress )); then
      device_state="$(adb_command get-state 2>/dev/null || true)"
      if [[ "$device_state" == "device" ]]; then
        echo "Still waiting for GPU diagnostic report $diagnostic_run_id."
      else
        echo "Device is temporarily unavailable; still waiting for report $diagnostic_run_id." >&2
      fi
      next_progress=$((SECONDS + 30))
    fi
    sleep "$report_poll_interval_seconds"
  done
  rm -f "$partial_destination"
  return 1
}

assert_idle() {
  local active_jobs
  local services
  adb_command exec-out run-as "$package_name" id >/dev/null 2>&1 ||
    fail "VoiceMemo must already be installed as a debuggable build."
  active_jobs="$(
    database_query \
      "SELECT COUNT(*) FROM transcription_jobs
       WHERE status IN ('QUEUED','RUNNING','PAUSED_FOR_RECORDING','CANCEL_REQUESTED');"
  )"
  [[ "$active_jobs" == "0" ]] ||
    fail "$active_jobs transcription job(s) are active."
  services="$(adb_command shell dumpsys activity services "$package_name" | tr -d '\r')"
  if grep -E -q "RecordingService|SystemForegroundService" <<<"$services"; then
    fail "A VoiceMemo recording or foreground transcription service is active."
  fi
}

subcommand="${1:-}"
if [[ "$subcommand" == "list" ]]; then
  shift
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --serial)
        [[ "$#" -ge 2 ]] || fail "--serial requires a value."
        device_serial="$2"
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        fail "Unknown list option: $1"
        ;;
    esac
  done
  command -v "$adb_binary" >/dev/null 2>&1 || fail "adb not found: $adb_binary"
  command -v "$sqlite_binary" >/dev/null 2>&1 || fail "sqlite3 not found."
  resolve_device
  temporary_directory="$(mktemp -d /tmp/voicememo-gpu-list.XXXXXX)"
  assert_idle
  printf '%s\n' \
    $'memo_id\ttitle\tduration_ms\tcreated_at_ms\treference_job\ttotal_chunks'
  database_query \
    "SELECT
       m.id || char(9) ||
       replace(replace(m.title, char(9), ' '), char(10), ' ') || char(9) ||
       m.durationMillis || char(9) ||
       m.createdAtMillis || char(9) ||
       COALESCE((
         SELECT j.id
         FROM transcription_jobs j
         WHERE j.memoId=m.id
           AND j.status='SUCCEEDED'
           AND j.completedChunks=j.totalChunks
           AND (
             j.inferenceFingerprint LIKE '%|runtime=cpu-xnnpack-%'
             OR j.inferenceFingerprint
                LIKE '%|runtime=gpu-parity-guard_cpu-xnnpack-%'
           )
         ORDER BY j.updatedAtMillis DESC
         LIMIT 1
       ), '') || char(9) ||
       COALESCE((
         SELECT j.totalChunks
         FROM transcription_jobs j
         WHERE j.memoId=m.id
           AND j.status='SUCCEEDED'
           AND j.completedChunks=j.totalChunks
           AND (
             j.inferenceFingerprint LIKE '%|runtime=cpu-xnnpack-%'
             OR j.inferenceFingerprint
                LIKE '%|runtime=gpu-parity-guard_cpu-xnnpack-%'
           )
         ORDER BY j.updatedAtMillis DESC
         LIMIT 1
       ), '')
     FROM memos m
     WHERE m.pendingDeletion=0
     ORDER BY m.createdAtMillis DESC
     LIMIT 20;"
  exit 0
fi

if [[ "$subcommand" == "install" ]]; then
  shift
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --serial)
        [[ "$#" -ge 2 ]] || fail "--serial requires a value."
        device_serial="$2"
        shift 2
        ;;
      --app-apk)
        [[ "$#" -ge 2 ]] || fail "--app-apk requires a path."
        app_apk="$2"
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        fail "Unknown install option: $1"
        ;;
    esac
  done
  command -v "$adb_binary" >/dev/null 2>&1 || fail "adb not found: $adb_binary"
  command -v "$sqlite_binary" >/dev/null 2>&1 || fail "sqlite3 not found."
  [[ -f "$app_apk" ]] || fail "App APK not found: $app_apk"
  resolve_device
  temporary_directory="$(mktemp -d /tmp/voicememo-install.XXXXXX)"
  assert_idle
  memo_count_before="$(memo_directory_count)"
  recording_manifest_before="$(recording_manifest_sha256)"
  transcript_state_before="$(transcript_state_sha256)"
  baseline_captured=true

  echo "Installing VoiceMemo in place (recordings preserved)."
  adb_command install -r "$app_apk"
  assert_idle
  installed_app_path="$(
    adb_command shell pm path "$package_name" |
      tr -d '\r' |
      sed -n 's/^package://p' |
      head -n 1
  )"
  [[ "$installed_app_path" =~ ^/data/app/[A-Za-z0-9_/.~+=-]+/base.apk$ ]] ||
    fail "Could not resolve a safe installed base APK path."
  installed_app_sha256="$(
    adb_command shell sha256sum "$installed_app_path" |
      tr -d '\r' |
      awk '{print $1}'
  )"
  local_app_sha256="$(shasum -a 256 "$app_apk" | awk '{print $1}')"
  [[ "$installed_app_sha256" == "$local_app_sha256" ]] ||
    fail "Installed VoiceMemo APK differs from the requested APK."
  echo "Installed APK SHA-256 verified; EXIT will verify every recording and transcript row."
  exit 0
fi

if [[ "$subcommand" == "prefix" ]]; then
  shift
  prefix_stage=""
  prefix_sha256=""
  prefix_bytes=""
  model_variant="int8-k"
  prefix_steps="0,7"
  reuse_installed_app=false
  isolation_id=""
  capture_isolated_key=false
  run_isolated_score=false
  capture_isolated_value=false
  run_isolated_value=false
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --memo-id)
        [[ "$#" -ge 2 ]] || fail "--memo-id requires a value."
        memo_id="$2"
        shift 2
        ;;
      --reference-job)
        [[ "$#" -ge 2 ]] || fail "--reference-job requires a value."
        reference_job="$2"
        shift 2
        ;;
      --prefix-stage)
        [[ "$#" -ge 2 ]] || fail "--prefix-stage requires a value."
        prefix_stage="$2"
        shift 2
        ;;
      --prefix-sha256)
        [[ "$#" -ge 2 ]] || fail "--prefix-sha256 requires a value."
        prefix_sha256="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
        shift 2
        ;;
      --prefix-bytes)
        [[ "$#" -ge 2 ]] || fail "--prefix-bytes requires a value."
        prefix_bytes="$2"
        shift 2
        ;;
      --model-variant)
        [[ "$#" -ge 2 ]] || fail "--model-variant requires a value."
        model_variant="$2"
        shift 2
        ;;
      --prefix-steps)
        [[ "$#" -ge 2 ]] || fail "--prefix-steps requires a value."
        prefix_steps="$2"
        shift 2
        ;;
      --reuse-installed-app)
        reuse_installed_app=true
        shift
        ;;
      --no-source-quant)
        allow_source_quantization=false
        shift
        ;;
      --isolation-id)
        [[ "$#" -ge 2 ]] || fail "--isolation-id requires a value."
        isolation_id="$2"
        shift 2
        ;;
      --capture-isolated-key)
        capture_isolated_key=true
        shift
        ;;
      --run-isolated-score)
        run_isolated_score=true
        shift
        ;;
      --capture-isolated-value)
        capture_isolated_value=true
        shift
        ;;
      --run-isolated-value)
        run_isolated_value=true
        shift
        ;;
      --serial)
        [[ "$#" -ge 2 ]] || fail "--serial requires a value."
        device_serial="$2"
        shift 2
        ;;
      --app-apk)
        [[ "$#" -ge 2 ]] || fail "--app-apk requires a value."
        app_apk="$2"
        shift 2
        ;;
      --test-apk)
        [[ "$#" -ge 2 ]] || fail "--test-apk requires a value."
        test_apk="$2"
        shift 2
        ;;
      --output)
        [[ "$#" -ge 2 ]] || fail "--output requires a directory."
        output_directory="$2"
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        fail "Unknown prefix option: $1"
        ;;
    esac
  done

  command -v "$adb_binary" >/dev/null 2>&1 || fail "adb not found: $adb_binary"
  command -v "$sqlite_binary" >/dev/null 2>&1 || fail "sqlite3 not found."
  command -v python3 >/dev/null 2>&1 || fail "python3 not found."
  [[ "$memo_id" =~ ^[A-Za-z0-9-]+$ ]] || fail "Invalid --memo-id."
  [[ "$reference_job" =~ ^[A-Za-z0-9-]+$ ]] || fail "Invalid --reference-job."
  [[ "$prefix_stage" =~ ^(after-position-add|after-self-layer-norm|after-self-key-projection|after-self-query-projection|after-self-query-reshape|after-self-key-transpose|after-self-score-matmul|after-self-mask-add|after-self-softmax|after-self-value-projection|after-self-value-reshape|after-self-value-matmul|after-self-context-reshape|after-self-output-projection|after-self-attention|after-cross-attention|after-1-block|after-2-blocks|after-3-blocks|after-4-blocks|after-final-layer-norm)$ ]] ||
    fail "Invalid --prefix-stage."
  [[ "$prefix_sha256" =~ ^[0-9a-f]{64}$ ]] || fail "Invalid --prefix-sha256."
  [[ "$prefix_bytes" =~ ^[0-9]{10}$ ]] ||
    fail "--prefix-bytes must be the exact ten-digit model size."
  [[ "$model_variant" =~ ^(int8-k|first-block-float-k|first-block-float-self-attention|first-block-float-self-attention-mask-rank-reshape)$ ]] ||
    fail "--model-variant is not a supported diagnostic variant."
  [[ "$prefix_steps" =~ ^([0-9]|[1-9][0-9]|1[01][0-9]|12[0-3])(,([0-9]|[1-9][0-9]|1[01][0-9]|12[0-3]))*$ ]] ||
    fail "--prefix-steps must be comma-separated values from 0 through 123."
  if [[ "$capture_isolated_key" == true ||
        "$run_isolated_score" == true ||
        "$capture_isolated_value" == true ||
        "$run_isolated_value" == true ]]; then
    [[ "$isolation_id" =~ ^[A-Za-z0-9-]+$ ]] ||
      fail "--isolation-id is required and must be safe for isolated BMM."
  fi
  if [[ "$capture_isolated_key" == true ]]; then
    [[ "$prefix_stage" == "after-self-key-transpose" ]] ||
      fail "--capture-isolated-key requires --prefix-stage after-self-key-transpose."
  fi
  if [[ "$run_isolated_score" == true ]]; then
    [[ "$prefix_stage" == "after-self-query-reshape" ]] ||
      fail "--run-isolated-score requires --prefix-stage after-self-query-reshape."
  fi
  if [[ "$capture_isolated_value" == true ]]; then
    [[ "$prefix_stage" == "after-self-value-reshape" ]] ||
      fail "--capture-isolated-value requires --prefix-stage after-self-value-reshape."
  fi
  if [[ "$run_isolated_value" == true ]]; then
    [[ "$prefix_stage" == "after-self-softmax" ]] ||
      fail "--run-isolated-value requires --prefix-stage after-self-softmax."
  fi
  [[ -f "$app_apk" ]] || fail "App APK not found: $app_apk"
  [[ -f "$test_apk" ]] || fail "Test APK not found: $test_apk"
  preflight_rewritten_prefix_apk "$prefix_sha256" "$prefix_bytes"

  resolve_device
  temporary_directory="$(mktemp -d /tmp/voicememo-gpu-prefix.XXXXXX)"
  diagnostic_run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
  [[ "$diagnostic_run_id" =~ ^[A-Za-z0-9-]+$ ]] ||
    fail "Could not generate a safe prefix run ID."
  if [[ -z "$output_directory" ]]; then
    output_directory="build/reports/device-gpu/prefix-$model_variant-$prefix_stage-$diagnostic_run_id"
  fi
  [[ ! -e "$output_directory" ]] || fail "Output already exists: $output_directory"
  mkdir -p "$output_directory"
  report_wait_timeout_seconds=1800

  assert_idle
  memo_row="$(
    database_query \
      "SELECT audioPath FROM memos
       WHERE id='$memo_id' AND pendingDeletion=0;"
  )"
  [[ -n "$memo_row" ]] || fail "Memo $memo_id does not exist."
  reference_count="$(
    database_query \
      "SELECT COUNT(*) FROM transcription_jobs
       WHERE id='$reference_job' AND memoId='$memo_id' AND status='SUCCEEDED'
         AND completedChunks=totalChunks;"
  )"
  [[ "$reference_count" == "1" ]] ||
    fail "Reference job is not complete chunk/language metadata for this memo."
  reference_chunk_count="$(
    database_query \
      "SELECT COUNT(*) FROM transcript_chunks
       WHERE jobId='$reference_job' AND chunkIndex=0 AND status='COMPLETED';"
  )"
  [[ "$reference_chunk_count" == "1" ]] ||
    fail "Reference chunk 0 is not complete."

  memo_count_before="$(memo_directory_count)"
  recording_manifest_before="$(recording_manifest_sha256)"
  transcript_state_before="$(transcript_state_sha256)"
  baseline_captured=true

  if [[ "$reuse_installed_app" == true ]]; then
    installed_app_path="$(
      adb_command shell pm path "$package_name" |
        tr -d '\r' |
        sed -n 's/^package://p' |
        head -n 1
    )"
    [[ "$installed_app_path" =~ ^/data/app/[A-Za-z0-9_/.~+=-]+/base.apk$ ]] ||
      fail "Could not resolve a safe installed base APK path."
    installed_app_sha256="$(
      adb_command shell sha256sum "$installed_app_path" |
        tr -d '\r' |
        awk '{print $1}'
    )"
    local_app_sha256="$(shasum -a 256 "$app_apk" | awk '{print $1}')"
    [[ "$installed_app_sha256" == "$local_app_sha256" ]] ||
      fail "Installed VoiceMemo APK differs from --app-apk; omit --reuse-installed-app."
    echo "Reusing the byte-identical installed app APK (recordings preserved)."
  else
    echo "Installing app in place (recordings preserved)."
    adb_command install -r "$app_apk"
  fi
  assert_idle
  echo "Installing instrumentation package in place."
  adb_command install -r "$test_apk"

  echo "Comparing CPU/GPU rewritten decoder boundary: $prefix_stage."
  echo "Model variant: $model_variant."
  echo "Diagnostic run ID: $diagnostic_run_id"
  set +e
  adb_command shell am instrument -w -r \
    -e memoId "$memo_id" \
    -e referenceJob "$reference_job" \
    -e runId "$diagnostic_run_id" \
    -e stage "$prefix_stage" \
    -e modelVariant "$model_variant" \
    -e steps "$prefix_steps" \
    -e allowSourceQuantization "$allow_source_quantization" \
    -e isolationId "${isolation_id:-none}" \
    -e captureIsolatedKey "$capture_isolated_key" \
    -e runIsolatedScore "$run_isolated_score" \
    -e captureIsolatedValue "$capture_isolated_value" \
    -e runIsolatedValue "$run_isolated_value" \
    -e class "$rewritten_prefix_test_class" \
    "$test_package_name/$runner_name" 2>&1 |
    tee "$output_directory/instrumentation.txt"
  instrument_pipeline_status=("${PIPESTATUS[@]}")
  set -e
  instrument_status="${instrument_pipeline_status[0]}"
  tee_status="${instrument_pipeline_status[1]}"
  [[ "$tee_status" -eq 0 ]] || fail "Could not save instrumentation output."
  if [[ "$instrument_status" -ne 0 ]]; then
    echo "Instrumentation ended with status $instrument_status; checking the report." >&2
  fi

  device_report="cache/gpu-prefix/prefix-$diagnostic_run_id.json"
  wait_for_device_report "$device_report" "$output_directory/report.json" ||
    fail "Timed out waiting for decoder-prefix report $diagnostic_run_id."
  python3 -c '
import json, sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("run_id") != sys.argv[2]:
    raise SystemExit("Decoder-prefix report run ID mismatch.")
if report.get("stage") != sys.argv[3]:
    raise SystemExit("Decoder-prefix report stage mismatch.")
if report.get("model_variant") != sys.argv[4]:
    raise SystemExit("Decoder-prefix report model variant mismatch.")
if report.get("status") != "completed":
    raise SystemExit(
        "Decoder-prefix diagnostic failed: %s: %s"
        % (report.get("error_type"), report.get("error_message"))
    )
for step in report["steps"]:
    full = step["full_boundary"]
    active = step["active_row"]
    unmasked = step.get("unmasked_active_row")
    print(
        "step=%s full_max_abs=%.9g full_mae=%.9g "
        "active_max_abs=%.9g active_mae=%.9g active_cosine=%.12g "
        "cpu_ms=%s gpu_ms=%s"
        % (
            step["step"],
            full["maximum_absolute_error"],
            full["mean_absolute_error"],
            active["maximum_absolute_error"],
            active["mean_absolute_error"],
            active["cosine_similarity"],
            step["cpu_inference_millis"],
            step["gpu_inference_millis"],
        )
    )
    if unmasked:
        print(
            "step=%s unmasked_active_count=%s unmasked_active_max_abs=%.9g "
            "unmasked_active_mae=%.9g unmasked_active_cosine=%.12g"
            % (
                step["step"],
                unmasked["element_count"],
                unmasked["maximum_absolute_error"],
                unmasked["mean_absolute_error"],
                unmasked["cosine_similarity"],
            )
        )
for label, key in (
    ("isolated_softmax", "isolated_softmax_same_cpu_input"),
    ("isolated_mask_add", "isolated_mask_add_same_cpu_input"),
    ("isolated_score_bmm", "isolated_score_same_cpu_inputs"),
    ("isolated_k_projection", "isolated_k_projection_same_cpu_input"),
    ("isolated_value_bmm", "isolated_value_same_cpu_inputs"),
):
    diagnostic = report.get(key)
    if not diagnostic:
        continue
    for step in diagnostic["steps"]:
        active = step["active_row"]
        print(
            "%s step=%s active_max_abs=%.9g active_mae=%.9g "
            "active_cosine=%.12g cpu_ms=%s gpu_ms=%s"
            % (
                label,
                step["step"],
                active["maximum_absolute_error"],
                active["mean_absolute_error"],
                active["cosine_similarity"],
                step["cpu_inference_millis"],
                step["gpu_inference_millis"],
            )
        )
        unmasked = step.get("unmasked_active_row")
        if unmasked:
            print(
                "%s step=%s unmasked_active_max_abs=%.9g "
                "unmasked_active_mae=%.9g unmasked_active_cosine=%.12g"
                % (
                    label,
                    step["step"],
                    unmasked["maximum_absolute_error"],
                    unmasked["mean_absolute_error"],
                    unmasked["cosine_similarity"],
                )
            )
' "$output_directory/report.json" "$diagnostic_run_id" "$prefix_stage" "$model_variant"
  echo "Prefix diagnostic completed; EXIT will verify every recording and transcript row."
  exit 0
fi

if [[ "$subcommand" == "whole" ]]; then
  shift
  reuse_installed_app=false
  use_full_float_decoder=false
  promotion_gate=false
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --memo-id)
        [[ "$#" -ge 2 ]] || fail "--memo-id requires a value."
        memo_id="$2"
        shift 2
        ;;
      --serial)
        [[ "$#" -ge 2 ]] || fail "--serial requires a value."
        device_serial="$2"
        shift 2
        ;;
      --app-apk)
        [[ "$#" -ge 2 ]] || fail "--app-apk requires a path."
        app_apk="$2"
        shift 2
        ;;
      --test-apk)
        [[ "$#" -ge 2 ]] || fail "--test-apk requires a path."
        test_apk="$2"
        shift 2
        ;;
      --output)
        [[ "$#" -ge 2 ]] || fail "--output requires a directory."
        output_directory="$2"
        shift 2
        ;;
      --reuse-installed-app)
        reuse_installed_app=true
        shift
        ;;
      --full-float-decoder)
        use_full_float_decoder=true
        shift
        ;;
      --promotion-gate)
        promotion_gate=true
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        fail "Unknown whole-memo option: $1"
        ;;
    esac
  done
  command -v "$adb_binary" >/dev/null 2>&1 || fail "adb not found: $adb_binary"
  command -v "$sqlite_binary" >/dev/null 2>&1 || fail "sqlite3 not found."
  command -v python3 >/dev/null 2>&1 || fail "python3 not found."
  [[ "$memo_id" =~ ^[A-Za-z0-9-]+$ ]] || fail "Invalid --memo-id."
  if [[ "$promotion_gate" == true && "$use_full_float_decoder" != true ]]; then
    fail "--promotion-gate requires --full-float-decoder."
  fi
  [[ -f "$app_apk" ]] || fail "App APK not found: $app_apk"
  [[ -f "$test_apk" ]] || fail "Test APK not found: $test_apk"
  if [[ "$use_full_float_decoder" == true ]]; then
    preflight_full_float_decoder_apk
  fi

  resolve_device
  temporary_directory="$(mktemp -d /tmp/voicememo-gpu-whole.XXXXXX)"
  diagnostic_run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
  [[ "$diagnostic_run_id" =~ ^[A-Za-z0-9-]+$ ]] ||
    fail "Could not generate a safe whole-memo run ID."
  if [[ -z "$output_directory" ]]; then
    output_directory="build/reports/device-gpu/whole-$diagnostic_run_id"
  fi
  [[ ! -e "$output_directory" ]] || fail "Output already exists: $output_directory"
  mkdir -p "$output_directory"
  report_wait_timeout_seconds=3600

  assert_idle
  memo_row="$(
    database_query \
      "SELECT audioPath FROM memos
       WHERE id='$memo_id' AND pendingDeletion=0;"
  )"
  [[ -n "$memo_row" ]] || fail "Memo $memo_id does not exist."
  memo_count_before="$(memo_directory_count)"
  recording_manifest_before="$(recording_manifest_sha256)"
  transcript_state_before="$(transcript_state_sha256)"
  baseline_captured=true

  if [[ "$reuse_installed_app" == true ]]; then
    installed_app_path="$(
      adb_command shell pm path "$package_name" |
        tr -d '\r' |
        sed -n 's/^package://p' |
        head -n 1
    )"
    [[ "$installed_app_path" =~ ^/data/app/[A-Za-z0-9_/.~+=-]+/base.apk$ ]] ||
      fail "Could not resolve a safe installed base APK path."
    installed_app_sha256="$(
      adb_command shell sha256sum "$installed_app_path" |
        tr -d '\r' |
        awk '{print $1}'
    )"
    local_app_sha256="$(shasum -a 256 "$app_apk" | awk '{print $1}')"
    [[ "$installed_app_sha256" == "$local_app_sha256" ]] ||
      fail "Installed VoiceMemo APK differs from --app-apk; omit --reuse-installed-app."
    echo "Reusing the byte-identical installed app APK (recordings preserved)."
  else
    echo "Installing app in place (recordings preserved)."
    adb_command install -r "$app_apk"
  fi
  assert_idle
  echo "Installing instrumentation package in place."
  adb_command install -r "$test_apk"

  echo "Running non-persisting whole-memo GPU WER benchmark."
  echo "Benchmark run ID: $diagnostic_run_id"
  set +e
  adb_command shell am instrument -w -r \
    -e memoId "$memo_id" \
    -e runId "$diagnostic_run_id" \
    -e useFullFloatDecoder "$use_full_float_decoder" \
    -e promotionGate "$promotion_gate" \
    -e class "$whole_memo_test_class" \
    "$test_package_name/$runner_name" 2>&1 |
    tee "$output_directory/instrumentation.txt"
  instrument_pipeline_status=("${PIPESTATUS[@]}")
  set -e
  instrument_status="${instrument_pipeline_status[0]}"
  tee_status="${instrument_pipeline_status[1]}"
  [[ "$tee_status" -eq 0 ]] || fail "Could not save instrumentation output."
  if [[ "$instrument_status" -ne 0 ]]; then
    echo "Instrumentation ended with status $instrument_status; checking the checkpoint report." >&2
  fi

  device_report="cache/gpu-wer/whole-$diagnostic_run_id.json"
  wait_for_device_report "$device_report" "$output_directory/report.json" ||
    fail "Timed out waiting for whole-memo report $diagnostic_run_id."
  python3 -c '
import json, sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("run_id") != sys.argv[2]:
    raise SystemExit("Whole-memo GPU report run ID mismatch.")
expected_full_float = sys.argv[3] == "true"
if report.get("use_full_float_decoder") is not expected_full_float:
    raise SystemExit("Whole-memo GPU model-variant flag mismatch.")
expected_promotion_gate = sys.argv[4] == "true"
if report.get("promotion_gate") is not expected_promotion_gate:
    raise SystemExit("Whole-memo GPU promotion-gate flag mismatch.")
print(
    "reference: mode=%s words=%s inference_ms=%s"
    % (
        report["reference"]["mode"],
        report["reference"]["word_count"],
        report["reference"]["inference_wall_millis"],
    )
)
for candidate in report.get("candidates", []):
    if candidate.get("status") != "completed":
        print(
            "%s: failed %s: %s"
            % (
                candidate.get("mode"),
                candidate.get("error_type"),
                candidate.get("error_message"),
            )
        )
        continue
    print(
        "%s: wer_percent=%.3f words=%s inference_ms=%s pss_kib=%s"
        % (
            candidate["mode"],
            candidate["word_error"]["percent"],
            candidate["word_count"],
            candidate["inference_wall_millis"],
            candidate["memory_after_inference"]["total_pss_kib"],
        )
    )
if report.get("status") != "completed":
    print(
        "Whole-memo report is a checkpoint with status %s."
        % report.get("status"),
        file=sys.stderr,
    )
' "$output_directory/report.json" "$diagnostic_run_id" "$use_full_float_decoder" "$promotion_gate"
  echo "Whole-memo benchmark completed; EXIT will verify every recording and transcript row."
  exit 0
fi

[[ "$subcommand" == "run" ]] || {
  usage
  exit 2
}
shift

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --memo-id)
      [[ "$#" -ge 2 ]] || fail "--memo-id requires a value."
      memo_id="$2"
      shift 2
      ;;
    --reference-job)
      [[ "$#" -ge 2 ]] || fail "--reference-job requires a value."
      reference_job="$2"
      shift 2
      ;;
    --chunk-index)
      [[ "$#" -ge 2 ]] || fail "--chunk-index requires a value."
      chunk_index="$2"
      shift 2
      ;;
    --mali-wait)
      hint_waiting_for_completion=true
      shift
      ;;
    --kernel-batch-size)
      [[ "$#" -ge 2 ]] || fail "--kernel-batch-size requires a value."
      kernel_batch_size="$2"
      shift 2
      ;;
    --no-source-quant)
      allow_source_quantization=false
      shift
      ;;
    --no-constant-sharing)
      constant_tensor_sharing=false
      shift
      ;;
    --strict-gpu-only)
      strict_gpu_only=true
      shift
      ;;
    --gpu-backend)
      [[ "$#" -ge 2 ]] || fail "--gpu-backend requires a value."
      gpu_backend="$2"
      shift 2
      ;;
    --buffer-storage)
      [[ "$#" -ge 2 ]] || fail "--buffer-storage requires a value."
      buffer_storage="$2"
      shift 2
      ;;
    --prefer-texture-weights)
      prefer_texture_weights=true
      shift
      ;;
    --prefer-buffer-weights)
      prefer_texture_weights=false
      shift
      ;;
    --benchmark-mode)
      benchmark_mode=true
      shift
      ;;
    --teacher-forced-only)
      run_autoregressive=false
      shift
      ;;
    --teacher-forced-steps)
      [[ "$#" -ge 2 ]] || fail "--teacher-forced-steps requires a value."
      teacher_forced_steps="$2"
      shift 2
      ;;
    --live-cpu-reference)
      live_cpu_reference=true
      shift
      ;;
    --decoder-quant-ab)
      decoder_quant_ab=true
      shift
      ;;
    --speculative-block-size)
      [[ "$#" -ge 2 ]] || fail "--speculative-block-size requires a value."
      speculative_block_size="$2"
      shift 2
      ;;
    --decoder-split)
      run_decoder_split=true
      shift
      ;;
    --reverse-decoder-split)
      reverse_decoder_split=true
      shift
      ;;
    --gpu-encoder-cpu-decoder)
      run_gpu_encoder_cpu_decoder=true
      shift
      ;;
    --decoder-split-stage)
      [[ "$#" -ge 2 ]] || fail "--decoder-split-stage requires a value."
      decoder_split_stage="$2"
      shift 2
      ;;
    --profiles)
      [[ "$#" -ge 2 ]] || fail "--profiles requires a comma-separated value."
      gpu_profiles="$2"
      shift 2
      ;;
    --matrix)
      gpu_profiles="current,opencl-buffer,opencl-texture2d,opencl-texture-weights,opencl-buffer-weights,automatic,opengl,benchmark,no-source-quant"
      shift
      ;;
    --serial)
      [[ "$#" -ge 2 ]] || fail "--serial requires a value."
      device_serial="$2"
      shift 2
      ;;
    --app-apk)
      [[ "$#" -ge 2 ]] || fail "--app-apk requires a value."
      app_apk="$2"
      shift 2
      ;;
    --test-apk)
      [[ "$#" -ge 2 ]] || fail "--test-apk requires a value."
      test_apk="$2"
      shift 2
      ;;
    --output)
      [[ "$#" -ge 2 ]] || fail "--output requires a directory."
      output_directory="$2"
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

if [[ "$decoder_quant_ab" == true ]]; then
  [[ -z "$gpu_profiles" ]] ||
    fail "--decoder-quant-ab cannot be combined with --profiles or --matrix."
  [[ -z "$teacher_forced_steps" ]] ||
    fail "--decoder-quant-ab cannot be combined with --teacher-forced-steps."
  [[ "$run_decoder_split" != true ]] ||
    fail "--decoder-quant-ab cannot be combined with --decoder-split."
  gpu_profiles="current,no-source-quant"
  teacher_forced_steps="all"
  live_cpu_reference=true
fi

command -v "$adb_binary" >/dev/null 2>&1 || fail "adb not found: $adb_binary"
command -v "$sqlite_binary" >/dev/null 2>&1 || fail "sqlite3 not found."
command -v python3 >/dev/null 2>&1 || fail "python3 not found."
[[ "$memo_id" =~ ^[A-Za-z0-9-]+$ ]] || fail "Invalid --memo-id."
[[ "$reference_job" =~ ^[A-Za-z0-9-]+$ ]] || fail "Invalid --reference-job."
[[ "$chunk_index" =~ ^[0-9]+$ ]] || fail "--chunk-index must be non-negative."
[[ "$gpu_backend" =~ ^(opencl|automatic|webgpu|opengl)$ ]] ||
  fail "--gpu-backend must be opencl, automatic, webgpu, or opengl."
[[ "$buffer_storage" =~ ^(default|buffer|texture2d)$ ]] ||
  fail "--buffer-storage must be default, buffer, or texture2d."
if [[ -n "$gpu_profiles" ]]; then
  [[ "$gpu_profiles" =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] ||
    fail "--profiles must be a comma-separated list of profile names."
fi
if [[ -n "$kernel_batch_size" ]]; then
  [[ "$kernel_batch_size" =~ ^[1-9][0-9]*$ ]] ||
    fail "--kernel-batch-size must be positive."
fi
if [[ -n "$teacher_forced_steps" ]]; then
  [[ "$teacher_forced_steps" == "all" ||
      "$teacher_forced_steps" =~ ^[1-9][0-9]*$ ]] ||
    fail "--teacher-forced-steps must be 'all' or a positive integer."
fi
[[ "$speculative_block_size" =~ ^([0-9]|[12][0-9]|3[0-2])$ ]] ||
  fail "--speculative-block-size must be between 0 and 32."
if [[ "$run_decoder_split" != true &&
      "$speculative_block_size" -gt 16 ]]; then
  fail "--speculative-block-size must be between 0 and 16 without --decoder-split."
fi
if [[ "$speculative_block_size" != "0" && "$gpu_profiles" == *,* ]]; then
  fail "Speculative decoding accepts only one GPU profile per run."
fi
if [[ "$run_decoder_split" == true && "$speculative_block_size" != "0" ]]; then
  [[ "$decoder_split_stage" == "after-1-block" ]] ||
    fail "--decoder-split speculation requires --decoder-split-stage after-1-block."
fi
if [[ "$run_decoder_split" == true && -n "$gpu_profiles" ]]; then
  fail "--decoder-split cannot be combined with --profiles or --matrix."
fi
if [[ "$run_decoder_split" == true &&
      "$run_gpu_encoder_cpu_decoder" == true ]]; then
  fail "--decoder-split cannot be combined with --gpu-encoder-cpu-decoder."
fi
if [[ "$reverse_decoder_split" == true &&
      "$run_decoder_split" != true ]]; then
  fail "--reverse-decoder-split requires --decoder-split."
fi
if [[ "$reverse_decoder_split" == true ]] &&
    ! is_generated_boundary_stage "$decoder_split_stage"; then
  fail "--reverse-decoder-split requires a generated after-* decoder boundary stage."
fi
[[ "$decoder_split_stage" =~ ^(post-layernorm|pre-layernorm|after-position-add|after-1-block|after-2-blocks|after-3-blocks|after-cross-attention)$ ]] ||
  fail "--decoder-split-stage must be post-layernorm, pre-layernorm, after-position-add, after-1-block, after-2-blocks, after-3-blocks, or after-cross-attention."
if [[ "$run_decoder_split" != true &&
      "$decoder_split_stage" != "post-layernorm" ]]; then
  fail "--decoder-split-stage requires --decoder-split."
fi
if [[ "$run_decoder_split" != true &&
      "$constant_tensor_sharing" != true ]]; then
  fail "--no-constant-sharing requires --decoder-split."
fi
if [[ "$run_decoder_split" != true && "$strict_gpu_only" == true ]]; then
  fail "--strict-gpu-only requires --decoder-split."
fi
[[ -f "$app_apk" ]] || fail "App APK not found: $app_apk"
[[ -f "$test_apk" ]] || fail "Test APK not found: $test_apk"
preflight_decoder_split_apk

resolve_device
temporary_directory="$(mktemp -d /tmp/voicememo-gpu-diagnostic.XXXXXX)"
diagnostic_run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
[[ "$diagnostic_run_id" =~ ^[A-Za-z0-9-]+$ ]] ||
  fail "Could not generate a safe diagnostic run ID."
if [[ -z "$output_directory" ]]; then
  output_directory="build/reports/device-gpu/$diagnostic_run_id"
fi
[[ ! -e "$output_directory" ]] || fail "Output already exists: $output_directory"
mkdir -p "$output_directory"

assert_idle
memo_row="$(
  database_query \
    "SELECT audioPath FROM memos
     WHERE id='$memo_id' AND pendingDeletion=0;"
)"
[[ -n "$memo_row" ]] || fail "Memo $memo_id does not exist."
if [[ "$live_cpu_reference" == true ]]; then
  reference_count="$(
    database_query \
      "SELECT COUNT(*) FROM transcription_jobs
       WHERE id='$reference_job' AND memoId='$memo_id' AND status='SUCCEEDED'
         AND completedChunks=totalChunks;"
  )"
  [[ "$reference_count" == "1" ]] ||
    fail "Reference job is not complete chunk/language metadata for this memo."
else
  reference_count="$(
    database_query \
      "SELECT COUNT(*) FROM transcription_jobs
       WHERE id='$reference_job' AND memoId='$memo_id' AND status='SUCCEEDED'
         AND completedChunks=totalChunks
         AND (
           inferenceFingerprint LIKE '%|runtime=cpu-xnnpack-%'
           OR inferenceFingerprint LIKE '%|runtime=gpu-parity-guard_cpu-xnnpack-%'
         );"
  )"
  [[ "$reference_count" == "1" ]] ||
    fail "Reference job is not a complete trusted CPU/XNNPACK job for this memo."
fi
reference_chunk_count="$(
  database_query \
    "SELECT COUNT(*) FROM transcript_chunks
     WHERE jobId='$reference_job' AND chunkIndex=$chunk_index
       AND status='COMPLETED';"
)"
[[ "$reference_chunk_count" == "1" ]] ||
  fail "Reference chunk $chunk_index is not complete."

memo_count_before="$(memo_directory_count)"
recording_manifest_before="$(recording_manifest_sha256)"
transcript_state_before="$(transcript_state_sha256)"
baseline_captured=true

echo "Installing app in place (recordings preserved)."
adb_command install -r "$app_apk"
assert_idle

echo "Installing instrumentation package in place."
adb_command install -r "$test_apk"

echo "Running isolated GPU boundary diagnostic for chunk $chunk_index."
echo "Diagnostic run ID: $diagnostic_run_id"
instrument_arguments=(
  shell am instrument -w -r
  -e memoId "$memo_id"
  -e referenceJob "$reference_job"
  -e diagnosticRunId "$diagnostic_run_id"
  -e chunkIndex "$chunk_index"
  -e hintWaitingForCompletion "$hint_waiting_for_completion"
  -e allowSourceQuantization "$allow_source_quantization"
  -e constantTensorSharing "$constant_tensor_sharing"
  -e strictGpuOnly "$strict_gpu_only"
  -e gpuBackend "$gpu_backend"
  -e bufferStorage "$buffer_storage"
  -e benchmarkMode "$benchmark_mode"
  -e runAutoregressive "$run_autoregressive"
  -e liveCpuReference "$live_cpu_reference"
  -e speculativeBlockSize "$speculative_block_size"
  -e runDecoderSplit "$run_decoder_split"
  -e reverseDecoderSplit "$reverse_decoder_split"
  -e runGpuEncoderCpuDecoder "$run_gpu_encoder_cpu_decoder"
  -e decoderSplitStage "$decoder_split_stage"
)
if [[ -n "$teacher_forced_steps" ]]; then
  instrument_arguments+=(-e teacherForcedSteps "$teacher_forced_steps")
fi
if [[ -n "$kernel_batch_size" ]]; then
  instrument_arguments+=(-e kernelBatchSize "$kernel_batch_size")
fi
if [[ -n "$prefer_texture_weights" ]]; then
  instrument_arguments+=(-e preferTextureWeights "$prefer_texture_weights")
fi
if [[ -n "$gpu_profiles" ]]; then
  instrument_arguments+=(-e gpuProfiles "$gpu_profiles")
fi
instrument_arguments+=(
  -e class "$test_class"
  "$test_package_name/$runner_name"
)
set +e
adb_command "${instrument_arguments[@]}" 2>&1 |
  tee "$output_directory/instrumentation.txt"
instrument_pipeline_status=("${PIPESTATUS[@]}")
set -e
instrument_status="${instrument_pipeline_status[0]}"
tee_status="${instrument_pipeline_status[1]}"
[[ "$tee_status" -eq 0 ]] ||
  fail "Could not save instrumentation output."
if [[ "$instrument_status" -ne 0 ]]; then
  echo "Instrumentation connection ended with status $instrument_status; waiting for the exact device report." >&2
fi

device_report="cache/gpu-parity/matrix-chunk-$chunk_index-$diagnostic_run_id.json"
wait_for_device_report "$device_report" "$output_directory/report.json" ||
  fail "Timed out after ${report_wait_timeout_seconds}s waiting for $device_report (instrumentation status $instrument_status)."

python3 -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("diagnostic_run_id") != sys.argv[2]:
    raise SystemExit(
        "GPU diagnostic report run ID mismatch: expected %s, got %s"
        % (sys.argv[2], report.get("diagnostic_run_id"))
    )
if report.get("status") != "completed":
    raise SystemExit(
        "GPU diagnostic failed: %s: %s"
        % (report.get("error_type"), report.get("error_message"))
    )
if not report.get("successful_candidate_count"):
    raise SystemExit("GPU diagnostic produced no successful candidate.")
for candidate in report["candidates"]:
    name = candidate["name"]
    if candidate["status"] != "completed":
        print(
            "%s: failed %s: %s"
            % (name, candidate.get("error_type"), candidate.get("error_message"))
        )
        continue
    print("%s: completed options=%s" % (name, candidate["options"]))
    if report.get("run_decoder_split"):
        cpu_control = candidate["cpu_control"]
        cpu_full = cpu_control.get("autoregressive")
        gpu_split = candidate["gpu_split"]
        gpu_full = gpu_split.get("autoregressive")
        for step in cpu_control["teacher_forced_steps"]:
            print(
                "  split_cpu_control_step_%s: exact_logits=%s top1_match=%s cpu=%s split_cpu=%s"
                % (
                    step["step"],
                    step["exact_logits_match"],
                    step["selectable_top1_match"],
                    step["cpu_selectable_top1_token_id"],
                    step["split_selectable_top1_token_id"],
                )
            )
        if cpu_full:
            print(
                "  split_cpu_control: exact=%s tokens=%s core_ms=%s projection_ms=%s wall_ms=%s pss_kib=%s"
                % (
                    cpu_full["exact_cpu_result_match"],
                    cpu_full["token_count"],
                    cpu_full["decoder_core_millis"],
                    cpu_full["active_row_projection_millis"],
                    cpu_full["wall_millis"],
                    cpu_control["memory_after_decode"]["total_pss_kib"],
                )
            )
        if gpu_full:
            cpu_reference_ms = report["cpu_reference_decode"]["inference_millis"]
            speedup = (
                float(cpu_reference_ms) / float(gpu_full["wall_millis"])
                if gpu_full["wall_millis"]
                else None
            )
            print(
                "  split_gpu: exact=%s tokens=%s first_mismatch=%s core_ms=%s projection_ms=%s wall_ms=%s cpu_reference_ms=%s speedup=%s pss_kib=%s"
                % (
                    gpu_full["exact_cpu_result_match"],
                    gpu_full["token_count"],
                    gpu_full["first_token_mismatch"],
                    gpu_full["decoder_core_millis"],
                    gpu_full["active_row_projection_millis"],
                    gpu_full["wall_millis"],
                    cpu_reference_ms,
                    speedup,
                    gpu_split["memory_after_decode"]["total_pss_kib"],
                )
            )
        speculative = candidate.get("cpu_verified_speculative_decode")
        if speculative:
            cpu_reference_ms = report["cpu_reference_decode"]["inference_millis"]
            speculative_ms = speculative["wall_millis"]
            speculative_inference_ms = speculative["inference_component_millis"]
            speedup = (
                float(cpu_reference_ms) / float(speculative_inference_ms)
                if speculative_inference_ms
                else None
            )
            print(
                "  split_speculative: block=%s exact=%s tokens=%s token_sha256=%s acceptance=%s cpu_corrections=%s gpu_calls=%s cpu_verifiers=%s inference_ms=%s wall_ms=%s cpu_reference_ms=%s speedup=%s pss_kib=%s"
                % (
                    speculative["block_size"],
                    speculative["exact_cpu_result_match"],
                    speculative["token_count"],
                    speculative["token_sha256"],
                    speculative["proposal_acceptance_rate"],
                    speculative["cpu_correction_count"],
                    speculative["gpu_decoder_invocations"],
                    speculative["verifier_invocations"],
                    speculative_inference_ms,
                    speculative_ms,
                    cpu_reference_ms,
                    speedup,
                    speculative["combined_memory_after_decode"]["total_pss_kib"],
                )
            )
        for step in gpu_split["teacher_forced_steps"]:
            hidden = step.get(
                "hidden_cpu_gpu_comparison",
                step.get("boundary_cpu_gpu_comparison"),
            )
            pre_layer_norm = step.get("pre_layer_norm_cpu_gpu_comparison")
            print(
                "  split_step_%s: top1_match=%s cpu=%s split_gpu=%s logits_rmse=%s boundary_rmse=%s boundary_cosine=%s pre_ln_rmse=%s pre_ln_cosine=%s"
                % (
                    step["step"],
                    step["selectable_top1_match"],
                    step["cpu_selectable_top1_token_id"],
                    step["split_selectable_top1_token_id"],
                    step["numeric_comparison"]["root_mean_square_error"],
                    hidden["root_mean_square_error"],
                    hidden["cosine_similarity"],
                    (
                        pre_layer_norm["root_mean_square_error"]
                        if pre_layer_norm
                        else None
                    ),
                    (
                        pre_layer_norm["cosine_similarity"]
                        if pre_layer_norm
                        else None
                    ),
                )
            )
        continue
    if report.get("run_autoregressive"):
        for lane in ("cpu_encoder_gpu_decoder", "gpu_encoder_gpu_decoder"):
            result = candidate[lane]
            print(
                "  %s: exact=%s tokens=%s first_mismatch=%s inference_ms=%s"
                % (
                    lane,
                    result["exact_token_match"],
                    result["token_count"],
                    result["first_token_mismatch"],
                    result["inference_millis"],
                )
            )
    hybrid = candidate.get("gpu_encoder_cpu_decoder")
    if hybrid:
        cpu_inference_ms = (
            report["cpu_encode_millis"]
            + report["cpu_reference_decode"]["inference_millis"]
        )
        hybrid_inference_ms = hybrid["inference_component_millis"]
        speedup = (
            float(cpu_inference_ms) / float(hybrid_inference_ms)
            if hybrid_inference_ms
            else None
        )
        print(
            "  gpu_encoder_cpu_decoder: exact=%s tokens=%s token_sha256=%s first_mismatch=%s gpu_encode_ms=%s gpu_close_ms=%s cpu_load_ms=%s cpu_decode_ms=%s inference_ms=%s sequential_ms=%s cpu_inference_ms=%s speedup=%s"
            % (
                hybrid["exact_cpu_result_match"],
                hybrid["token_count"],
                hybrid["token_sha256"],
                hybrid["first_token_mismatch"],
                hybrid["gpu_encode_millis"],
                hybrid["gpu_model_close_millis"],
                hybrid["cpu_decoder_model_load_millis"],
                hybrid["cpu_decode_millis"],
                hybrid_inference_ms,
                hybrid["sequential_component_millis"],
                cpu_inference_ms,
                speedup,
            )
        )
    speculative = candidate.get("cpu_verified_speculative_decode")
    if speculative:
        cpu_ms = report["cpu_reference_decode"]["inference_millis"]
        speculative_ms = speculative["wall_millis"]
        speedup = (
            float(cpu_ms) / float(speculative_ms)
            if speculative_ms
            else None
        )
        print(
            "  speculative: block=%s exact=%s acceptance=%s cpu_corrections=%s gpu_calls=%s cpu_verifiers=%s decode_ms=%s cpu_reference_ms=%s speedup=%s pss_kib=%s"
            % (
                speculative["block_size"],
                speculative["exact_cpu_result_match"],
                speculative["proposal_acceptance_rate"],
                speculative["cpu_correction_count"],
                speculative["gpu_decoder_invocations"],
                speculative["verifier_invocations"],
                speculative_ms,
                cpu_ms,
                speedup,
                speculative["combined_memory_after_decode"]["total_pss_kib"],
            )
        )
    teacher_forced = candidate["teacher_forced_cpu_encoder_gpu_decoder"]
    teacher_summary = teacher_forced.get("summary")
    if teacher_summary:
        print(
            "  teacher_summary: top1_match=%s/%s rate=%.6f first_mismatch=%s gpu_reference_match=%s/%s first_reference_mismatch=%s mean_rmse=%s max_rmse=%s gpu_ms=%s"
            % (
                teacher_summary["selectable_top1_match_count"],
                teacher_summary["step_count"],
                teacher_summary["selectable_top1_match_rate"],
                teacher_summary["first_selectable_top1_mismatch"],
                teacher_summary["gpu_top1_matches_reference_count"],
                teacher_summary["step_count"],
                teacher_summary["first_gpu_top1_reference_mismatch"],
                teacher_summary["mean_root_mean_square_error"],
                teacher_summary["maximum_root_mean_square_error"],
                teacher_summary["gpu_inference_millis"],
            )
        )
    for step in teacher_forced["steps"]:
        if (
            teacher_forced["step_count"] > 3
            and step["selectable_top1_match"]
            and step["gpu_selectable_top1_matches_reference"]
        ):
            continue
        print(
            "  teacher_step_%s: top1_match=%s cpu=%s gpu=%s cpu_margin=%s gpu_margin=%s rmse=%s centered_rmse=%s"
            % (
                step["step"],
                step["selectable_top1_match"],
                step["cpu_selectable_top1_token_id"],
                step["gpu_selectable_top1_token_id"],
                step["cpu_selectable_margin"],
                step["gpu_selectable_margin"],
                step["numeric_comparison"]["root_mean_square_error"],
                step["numeric_comparison"]["centered_root_mean_square_error"],
            )
        )
    encoder = candidate["encoder_comparison"]
    print(
        "  encoder: exact_elements=%s/%s rmse=%s max_abs=%s cosine=%s"
        % (
            encoder["exact_element_count"],
            encoder["element_count"],
            encoder["root_mean_square_error"],
            encoder["max_absolute_error"],
            encoder["cosine_similarity"],
        )
    )
' "$output_directory/report.json" "$diagnostic_run_id"

echo "GPU diagnostic completed; EXIT verification will check every memo audio file and transcript row."
echo "Results: $output_directory"
