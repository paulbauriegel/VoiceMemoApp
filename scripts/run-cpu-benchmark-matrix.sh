#!/usr/bin/env bash
set -euo pipefail

package_name="com.example.voicememo"
database_name="voice-memo.db"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd "$script_directory/.." && pwd)"
report_helper="$script_directory/summarize_cpu_benchmark_matrix.py"
apk_path="$project_directory/app/build/outputs/apk/debug/app-debug.apk"
configuration_csv="2x2,2x3,2x4,1x4,1x6"
timeout_minutes=120
device_serial=""
result_directory=""
adb_binary="${ADB:-adb}"
python_binary="${PYTHON3:-python3}"
sqlite_binary="${SQLITE3:-sqlite3}"
temporary_directory=""
benchmark_memo_id=""
accepted_reports=()
adb_arguments=()
configurations=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run-cpu-benchmark-matrix.sh run [options]

Builds and measures these CPU/XNNPACK configurations by default:
  2x2, 2x3, 2x4, 1x4, 1x6

Options:
  --serial SERIAL          Use one specific adb device.
  --output DIRECTORY       Host results directory. It must not already exist.
  --timeout-minutes N      Per-transcription timeout (default: 120).
  --configs LIST           Explicit comma-separated subset, for example 2x4,1x6.
  -h, --help               Show this help.

The required "run" subcommand and interactive prompts are intentional. This
tool never starts or retries a transcription on its own. It installs each APK
with "adb install -r", launches VoiceMemo, and waits for you to tap Transcribe
or Retranscribe on the same existing memo.

It never uninstalls VoiceMemo, clears app data, deletes a recording, or cancels
work. If recording or transcription is active, it refuses to install anything.
EOF
}

fail() {
  echo "CPU benchmark matrix stopped: $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "$temporary_directory" && -d "$temporary_directory" ]]; then
    rm -rf "$temporary_directory"
  fi
}

interrupted() {
  echo >&2
  echo "Interrupted. VoiceMemo work was not cancelled or modified." >&2
  cleanup
  exit 130
}

trap cleanup EXIT
trap interrupted INT TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

adb_command() {
  "$adb_binary" "${adb_arguments[@]}" "$@"
}

resolve_device() {
  local line_serial
  local line_state
  local ignored
  local connected_devices=()

  if [[ -n "$device_serial" ]]; then
    adb_arguments=(-s "$device_serial")
    [[ "$(adb_command get-state 2>/dev/null || true)" == "device" ]] ||
      fail "Device $device_serial is not connected and authorized."
    return
  fi

  while IFS=$'\t' read -r line_serial line_state ignored; do
    if [[ -n "$line_serial" && "$line_state" == "device" ]]; then
      connected_devices+=("$line_serial")
    fi
  done < <("$adb_binary" devices)

  if [[ "${#connected_devices[@]}" -eq 0 ]]; then
    fail "No authorized Android device is connected."
  fi
  if [[ "${#connected_devices[@]}" -ne 1 ]]; then
    fail "More than one device is connected; pass --serial SERIAL."
  fi

  device_serial="${connected_devices[0]}"
  adb_arguments=(-s "$device_serial")
}

package_is_installed() {
  local package_paths
  package_paths="$(adb_command shell pm path "$package_name" 2>/dev/null | tr -d '\r')" ||
    return 1
  grep -q '^package:' <<<"$package_paths"
}

run_as_is_available() {
  adb_command exec-out run-as "$package_name" id >/dev/null 2>&1
}

copy_private_file() {
  local remote_path="$1"
  local local_path="$2"
  if adb_command exec-out run-as "$package_name" cat "$remote_path" \
    >"$local_path" 2>/dev/null; then
    return 0
  fi
  rm -f "$local_path"
  return 1
}

private_file_state() {
  local remote_path="$1"
  local state
  state="$(
    adb_command exec-out run-as "$package_name" sh -c \
      'if [ -f "$1" ]; then printf exists; else printf missing; fi' \
      sh "$remote_path"
  )" || fail "Could not determine whether app-private file exists: $remote_path"
  case "$state" in
    exists | missing) printf '%s\n' "$state" ;;
    *) fail "Unexpected app-private file state for $remote_path: $state" ;;
  esac
}

copy_optional_private_file() {
  local remote_path="$1"
  local local_path="$2"
  if [[ "$(private_file_state "$remote_path")" == "missing" ]]; then
    return
  fi
  copy_private_file "$remote_path" "$local_path" ||
    fail "App-private file existed but could not be read: $remote_path"
}

database_snapshot_query_once() {
  local query="$1"
  local snapshot_directory
  local database_path
  local query_result

  snapshot_directory="$(mktemp -d "$temporary_directory/database.XXXXXX")"
  database_path="$snapshot_directory/$database_name"

  copy_private_file "databases/$database_name" "$database_path" ||
    fail "Cannot read the existing VoiceMemo database with run-as."
  copy_optional_private_file "databases/$database_name-wal" "$database_path-wal"
  copy_optional_private_file "databases/$database_name-shm" "$database_path-shm"

  if ! query_result="$(
    "$sqlite_binary" -readonly "$database_path" "PRAGMA query_only=ON; $query"
  )"; then
    fail "Could not verify VoiceMemo job state from a read-only database snapshot."
  fi
  rm -rf "$snapshot_directory"
  printf '%s\n' "$query_result"
}

database_snapshot_query() {
  local query="$1"
  local first_result
  local second_result
  first_result="$(database_snapshot_query_once "$query")"
  second_result="$(database_snapshot_query_once "$query")"
  if [[ "$first_result" != "$second_result" ]]; then
    fail "Two consecutive read-only database snapshots disagreed; device state may be changing."
  fi
  printf '%s\n' "$first_result"
}

database_snapshot_counts() {
  database_snapshot_query \
    "SELECT
       (SELECT COUNT(*) FROM transcription_jobs
        WHERE status IN ('QUEUED','RUNNING','PAUSED_FOR_RECORDING','CANCEL_REQUESTED'))
       || '|' ||
       (SELECT COUNT(*) FROM memos WHERE pendingDeletion = 0);"
}

latest_succeeded_job_identity() {
  database_snapshot_query \
    "SELECT memoId || '|' || updatedAtMillis
     FROM transcription_jobs
     WHERE status = 'SUCCEEDED'
     ORDER BY updatedAtMillis DESC
     LIMIT 1;"
}

notification_is_active() {
  local notification_dump="$1"
  local notification_id="$2"
  grep -E -q \
    "NotificationRecord.*pkg=${package_name}.*id=${notification_id}([^0-9]|$)|NotificationRecord.*id=${notification_id}([^0-9]|$).*pkg=${package_name}" \
    <<<"$notification_dump"
}

assert_device_idle() {
  local activity_dump
  local notification_dump
  local counts
  local active_jobs
  local memo_count

  [[ "$(adb_command get-state 2>/dev/null || true)" == "device" ]] ||
    fail "The selected device disconnected."
  package_is_installed ||
    fail "VoiceMemo is not installed. An existing in-app memo is required."
  run_as_is_available ||
    fail "The installed VoiceMemo build is not debuggable with run-as; refusing to replace it."

  activity_dump="$(adb_command shell dumpsys activity services "$package_name" | tr -d '\r')"
  notification_dump="$(adb_command shell dumpsys notification --noredact | tr -d '\r')"

  if grep -F -q "$package_name/.recording.RecordingService" <<<"$activity_dump"; then
    fail "A VoiceMemo recording service is active. Stop and save the recording first."
  fi
  if notification_is_active "$notification_dump" 1001; then
    fail "The VoiceMemo recording notification is active. Stop and save the recording first."
  fi
  if grep -F -q "androidx.work.impl.foreground.SystemForegroundService" \
    <<<"$activity_dump"; then
    fail "A VoiceMemo foreground transcription is active. Let it finish or cancel it in the app."
  fi
  if notification_is_active "$notification_dump" 1002; then
    fail "The VoiceMemo transcription notification is active. Let it finish or cancel it in the app."
  fi

  counts="$(database_snapshot_counts)"
  IFS='|' read -r active_jobs memo_count <<<"$counts"
  [[ "$active_jobs" =~ ^[0-9]+$ && "$memo_count" =~ ^[0-9]+$ ]] ||
    fail "Unexpected result while checking the VoiceMemo database."
  if [[ "$active_jobs" -ne 0 ]]; then
    fail "$active_jobs queued or active transcription job(s) exist. Resolve them in VoiceMemo first."
  fi
  if [[ "$memo_count" -eq 0 ]]; then
    fail "VoiceMemo has no existing memo to benchmark."
  fi

  activity_dump="$(adb_command shell dumpsys activity services "$package_name" | tr -d '\r')"
  notification_dump="$(adb_command shell dumpsys notification --noredact | tr -d '\r')"
  if grep -E -q "RecordingService|SystemForegroundService" <<<"$activity_dump"; then
    fail "VoiceMemo became active during the safety check; no install was attempted."
  fi
  if notification_is_active "$notification_dump" 1001 ||
    notification_is_active "$notification_dump" 1002; then
    fail "VoiceMemo became active during the safety check; no install was attempted."
  fi
}

assert_thermal_idle() {
  local thermal_dump
  local thermal_status

  thermal_dump="$(adb_command shell dumpsys thermalservice | tr -d '\r')"
  thermal_status="$(
    awk -F: '/Thermal Status:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' \
      <<<"$thermal_dump"
  )"
  if [[ "$thermal_status" =~ ^[0-9]+$ && "$thermal_status" -ne 0 ]]; then
    fail "Device thermal status is $thermal_status, not NONE (0). Let it cool, then invoke the run again."
  fi
  if [[ -z "$thermal_status" ]]; then
    echo "Warning: dumpsys did not expose a numeric thermal status; the report must still start at NONE." >&2
  fi
}

current_report_timestamp() {
  local report_path="$temporary_directory/current-latest.json"
  if [[ "$(private_file_state "files/benchmarks/latest.json")" == "missing" ]]; then
    printf '%s\n' "missing"
    return
  fi
  copy_private_file "files/benchmarks/latest.json" "$report_path" ||
    fail "latest.json exists but could not be read; refusing to use a missing sentinel."
  "$python_binary" "$report_helper" timestamp "$report_path" ||
    fail "The existing latest.json is invalid; refusing to start a comparison."
}

wait_for_new_report() {
  local previous_timestamp="$1"
  local destination="$2"
  local started_at
  local deadline
  local now
  local next_update
  local candidate_timestamp
  local candidate="$temporary_directory/polled-latest.json"

  started_at="$(date +%s)"
  deadline=$((started_at + timeout_minutes * 60))
  next_update=$((started_at + 60))

  while true; do
    now="$(date +%s)"
    if [[ "$now" -ge "$deadline" ]]; then
      fail "Timed out after $timeout_minutes minutes. No work was cancelled; inspect VoiceMemo on the device."
    fi
    if [[ "$(adb_command get-state 2>/dev/null || true)" != "device" ]]; then
      fail "The selected device disconnected while waiting for the report."
    fi

    if copy_private_file "files/benchmarks/latest.json" "$candidate"; then
      candidate_timestamp="$(
        "$python_binary" "$report_helper" timestamp "$candidate" 2>/dev/null || true
      )"
      if [[ -n "$candidate_timestamp" &&
        "$candidate_timestamp" != "$previous_timestamp" ]]; then
        cp "$candidate" "$destination"
        return
      fi
    fi

    if [[ "$now" -ge "$next_update" ]]; then
      echo "Still waiting for a completed benchmark report ($((now - started_at)) seconds elapsed)."
      next_update=$((now + 60))
    fi
    sleep 5
  done
}

validate_memo_identity() {
  local report_path="$1"
  local job_identity
  local job_memo_id
  local job_updated_at
  local report_timestamp
  local timestamp_delta

  job_identity="$(latest_succeeded_job_identity)"
  IFS='|' read -r job_memo_id job_updated_at <<<"$job_identity"
  if [[ -z "$job_memo_id" || ! "$job_updated_at" =~ ^[0-9]+$ ]]; then
    echo "No matching completed transcription job was found in the read-only database snapshot." >&2
    return 1
  fi

  report_timestamp="$("$python_binary" "$report_helper" timestamp "$report_path")"
  timestamp_delta=$((report_timestamp - job_updated_at))
  if [[ "$timestamp_delta" -lt 0 ]]; then
    timestamp_delta=$((-timestamp_delta))
  fi
  if [[ "$timestamp_delta" -gt 60000 ]]; then
    echo "The newest completed job is not close enough to the new benchmark timestamp." >&2
    return 1
  fi

  if [[ -z "$benchmark_memo_id" ]]; then
    benchmark_memo_id="$job_memo_id"
    echo "Pinned the first completed job's memo identity for the remaining runs."
    return 0
  fi
  if [[ "$job_memo_id" != "$benchmark_memo_id" ]]; then
    echo "The completed job belongs to a different memo than the first accepted run." >&2
    return 1
  fi
  echo "Confirmed the completed job used the same memo as the first accepted run."
}

write_device_state() {
  local config="$1"
  local destination="$result_directory/device-state-$config.txt"
  {
    echo "captured_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "serial=$device_serial"
    echo "model=$(adb_command shell getprop ro.product.manufacturer | tr -d '\r') $(adb_command shell getprop ro.product.model | tr -d '\r')"
    echo "build_fingerprint=$(adb_command shell getprop ro.build.fingerprint | tr -d '\r')"
    adb_command shell dumpsys battery |
      tr -d '\r' |
      awk '/AC powered:|USB powered:|Wireless powered:|level:|temperature:/ {print}'
    adb_command shell dumpsys thermalservice |
      tr -d '\r' |
      awk '/Thermal Status:/ {print; exit}'
    adb_command shell dumpsys power |
      tr -d '\r' |
      awk '/Display Power: state=/ {print; exit}'
  } >"$destination"
}

validate_configuration_list() {
  local config
  local seen=","
  IFS=',' read -r -a configurations <<<"$configuration_csv"
  [[ "${#configurations[@]}" -gt 0 ]] || fail "At least one configuration is required."

  for config in "${configurations[@]}"; do
    case "$config" in
      2x2 | 2x3 | 2x4 | 1x4 | 1x6) ;;
      *) fail "Unsupported CPU configuration: $config" ;;
    esac
    if [[ "$seen" == *",$config,"* ]]; then
      fail "Duplicate CPU configuration: $config"
    fi
    seen="${seen}${config},"
  done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" != "run" ]]; then
  usage >&2
  exit 2
fi
shift

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --serial)
      [[ "$#" -ge 2 ]] || fail "--serial requires a value."
      device_serial="$2"
      shift 2
      ;;
    --output)
      [[ "$#" -ge 2 ]] || fail "--output requires a value."
      result_directory="$2"
      shift 2
      ;;
    --timeout-minutes)
      [[ "$#" -ge 2 ]] || fail "--timeout-minutes requires a value."
      timeout_minutes="$2"
      shift 2
      ;;
    --configs)
      [[ "$#" -ge 2 ]] || fail "--configs requires a value."
      configuration_csv="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[[ "$timeout_minutes" =~ ^[0-9]+$ && "$timeout_minutes" -ge 1 ]] ||
  fail "--timeout-minutes must be a positive integer."
[[ -t 0 ]] ||
  fail "This workflow requires an interactive terminal; it never starts transcriptions automatically."

require_command "$adb_binary"
require_command "$python_binary"
require_command "$sqlite_binary"
[[ -f "$report_helper" ]] || fail "Missing report helper: $report_helper"
[[ -x "$project_directory/gradlew" ]] || fail "Missing executable Gradle wrapper."

validate_configuration_list
resolve_device

if [[ -z "$result_directory" ]]; then
  result_directory="$project_directory/benchmark-results/cpu-matrix-$(date '+%Y%m%d-%H%M%S')"
fi
[[ ! -e "$result_directory" ]] ||
  fail "Results path already exists; refusing to overwrite it: $result_directory"
mkdir -p "$result_directory"
temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/voicememo-cpu-matrix.XXXXXX")"

if [[ -z "${JAVA_HOME:-}" &&
  -d "/Applications/Android Studio.app/Contents/jbr/Contents/Home" ]]; then
  export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
fi

cat <<EOF
VoiceMemo CPU benchmark matrix

Device: $device_serial
Configurations: ${configuration_csv//,/ -> }
Results: $result_directory

This performs one cold-process CPU transcription per configuration and pins
the backend build property to CPU. It never retries automatically. Every APK
is installed in place with "adb install -r"; app data and recordings are
preserved. You must select the same memo each time using VoiceMemo's normal
Transcribe/Retranscribe button. The generated transcript for that memo will be
refreshed by the normal app workflow.

Keep the screen and power conditions consistent between runs. The first
accepted run pins the completed job's opaque memo identity plus its content-free
report fingerprint. Later mismatches are rejected.

Safety note: to detect queued jobs and verify memo identity without modifying
the app, this developer tool briefly copies the full Room database (which can
contain titles and transcript text, but not recording audio) into a user-only
host temporary directory. Each snapshot is deleted immediately after its
read-only query and the parent directory is removed on normal exit, INT, or
TERM. A host crash or SIGKILL can leave that temporary directory behind.
EOF

read -r -p "Type RUN to begin: " confirmation
[[ "$confirmation" == "RUN" ]] || fail "Confirmation was not RUN; nothing was installed."

assert_device_idle

for config in "${configurations[@]}"; do
  workers="${config%x*}"
  threads="${config#*x}"
  build_log="$result_directory/build-$config.log"
  install_log="$result_directory/install-$config.log"
  candidate_report="$temporary_directory/candidate-$config.json"
  accepted_report="$result_directory/$config.json"

  echo
  echo "=== $config: build ==="
  (
    cd "$project_directory"
    ./gradlew --no-daemon \
      "-PwhisperWorkerCount=$workers" \
      "-PwhisperThreadsPerWorker=$threads" \
      -PwhisperBackend=cpu \
      :app:assembleDebug
  ) 2>&1 | tee "$build_log"
  [[ -f "$apk_path" ]] || fail "Gradle succeeded but the debug APK is missing: $apk_path"

  echo "=== $config: pre-install safety check ==="
  assert_device_idle
  assert_thermal_idle
  write_device_state "$config"
  # Device-state collection takes several adb round trips. Close that window
  # with another full service/notification/database check immediately before
  # the in-place install.
  assert_device_idle

  echo "=== $config: in-place install ==="
  adb_command install -r "$apk_path" 2>&1 | tee "$install_log"
  package_is_installed || fail "VoiceMemo is not installed after adb install -r."
  run_as_is_available || fail "The installed debug APK does not allow report collection with run-as."

  previous_timestamp="$(current_report_timestamp)"
  adb_command shell am start -W -n "$package_name/.MainActivity" >/dev/null

  echo
  if [[ "${#accepted_reports[@]}" -eq 0 ]]; then
    echo "On the device, open the existing memo you want to use for every run."
  else
    echo "On the device, open the exact same memo used for the first accepted run."
  fi
  echo "Tap Transcribe or Retranscribe for configuration $config."
  read -r -p "Press Enter only after you have tapped the button: " _

  echo "Waiting for $config to finish; this script will not retry or cancel it."
  wait_for_new_report "$previous_timestamp" "$candidate_report"

  if [[ "${#accepted_reports[@]}" -eq 0 ]]; then
    report_is_valid=true
    "$python_binary" "$report_helper" validate "$candidate_report" \
      --workers "$workers" \
      --threads "$threads" ||
      report_is_valid=false
  else
    report_is_valid=true
    "$python_binary" "$report_helper" validate "$candidate_report" \
      --workers "$workers" \
      --threads "$threads" \
      --baseline "${accepted_reports[0]}" ||
      report_is_valid=false
  fi
  if [[ "$report_is_valid" != true ]]; then
    rejected_name="$result_directory/rejected-$config-$(date '+%Y%m%d-%H%M%S').json"
    cp "$candidate_report" "$rejected_name"
    fail "The new report was rejected and saved as $rejected_name. It will not be rerun automatically."
  fi
  if ! validate_memo_identity "$candidate_report"; then
    rejected_name="$result_directory/rejected-$config-$(date '+%Y%m%d-%H%M%S').json"
    cp "$candidate_report" "$rejected_name"
    fail "The memo identity check failed. The report was saved as $rejected_name and will not be rerun automatically."
  fi

  cp "$candidate_report" "$accepted_report"
  accepted_reports+=("$accepted_report")
  "$python_binary" "$report_helper" summarize \
    --output-dir "$result_directory" \
    "${accepted_reports[@]}"
done

echo
echo "CPU benchmark matrix complete."
echo "Reports: $result_directory"
echo "Summary: $result_directory/summary.md"
echo "The device still has the last benchmark build ($config) installed."
echo "The source default remains 2x3; restore it later with an explicit in-place build/install."
