#!/usr/bin/env bash
set -euo pipefail

package_name="com.example.voicememo"
activity_name="com.example.voicememo/.MainActivity"
database_name="voice-memo.db"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
node_helper="$script_directory/find-uiautomator-node.py"
parity_helper="$script_directory/compare-transcription-parity.py"
adb_binary="${ADB:-adb}"
python_binary="${PYTHON3:-python3}"
sqlite_binary="${SQLITE3:-sqlite3}"
device_serial=""
memo_id=""
timeout_minutes=45
install_apk=""
reference_database=""
reference_job=""
save_result_database_directory=""
result_directory=""
temporary_directory=""
device_ui_path=""
host_ui_path=""
adb_arguments=()
screen_width=1080
screen_height=1920

usage() {
  cat <<'EOF'
Usage:
  scripts/run-transcription-ui-test.sh run --memo-id ID [options]

Runs the real VoiceMemo retranscription flow through UI Automator:
  1. verifies that recording/transcription is idle;
  2. optionally updates the app with adb install -r;
  3. opens the memo by its stable Compose test tag;
  4. selects Transcript and taps Transcribe/Transcribe again;
  5. waits for the new Room job to finish;
  6. verifies that memo count, audio bytes, and edited text were preserved;
  7. optionally compares exact chunk tokens/text with a reference CPU job.

Options:
  --memo-id ID              Existing memo to transcribe (required).
  --serial SERIAL           Use one specific adb device.
  --install-apk PATH        Install this APK in place before the test.
  --timeout-minutes N       Test timeout (default: 45).
  --output DIRECTORY        New host result directory.
  --reference-db PATH       Read-only CPU reference database.
  --reference-job ID        CPU reference job in --reference-db.
  --save-result-db DIR      Save the completed Room snapshot for a later
                            parity comparison. The directory must not exist.
  -h, --help                Show this help.

Safety:
  The script never uninstalls VoiceMemo, clears data, deletes recordings, or
  cancels work. It refuses to begin while recording/transcription is active.
  If the timeout is reached, the app job is left untouched.
EOF
}

fail() {
  echo "VoiceMemo UI test failed: $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "$device_ui_path" ]]; then
    "$adb_binary" "${adb_arguments[@]}" shell rm -f "$device_ui_path" \
      >/dev/null 2>&1 || true
  fi
  if [[ -n "$temporary_directory" && -d "$temporary_directory" ]]; then
    rm -rf "$temporary_directory"
  fi
}

interrupted() {
  echo >&2
  echo "Interrupted. The VoiceMemo transcription was not cancelled." >&2
  exit 130
}

trap cleanup EXIT
trap interrupted INT TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    fail "Required command not found: $1"
}

adb_command() {
  "$adb_binary" "${adb_arguments[@]}" "$@"
}

resolve_device() {
  local line_serial
  local line_state
  local ignored
  local connected=()

  if [[ -n "$device_serial" ]]; then
    adb_arguments=(-s "$device_serial")
    [[ "$(adb_command get-state 2>/dev/null || true)" == "device" ]] ||
      fail "Device $device_serial is not connected and authorized."
    return
  fi

  while IFS=$'\t' read -r line_serial line_state ignored; do
    if [[ -n "$line_serial" && "$line_state" == "device" ]]; then
      connected+=("$line_serial")
    fi
  done < <("$adb_binary" devices)

  [[ "${#connected[@]}" -gt 0 ]] ||
    fail "No authorized Android device is connected."
  [[ "${#connected[@]}" -eq 1 ]] ||
    fail "More than one device is connected; pass --serial SERIAL."
  device_serial="${connected[0]}"
  adb_arguments=(-s "$device_serial")
}

copy_private_file() {
  local remote_path="$1"
  local local_path="$2"
  adb_command exec-out run-as "$package_name" cat "$remote_path" \
    >"$local_path" 2>/dev/null
}

copy_optional_private_file() {
  local remote_path="$1"
  local local_path="$2"
  if adb_command shell run-as "$package_name" test -f "$remote_path"; then
    copy_private_file "$remote_path" "$local_path" ||
      fail "Could not read app-private file $remote_path."
  fi
}

snapshot_database() {
  local destination="$1"
  mkdir -p "$destination"
  copy_private_file "databases/$database_name" "$destination/$database_name" ||
    fail "Cannot read the VoiceMemo database with run-as."
  copy_optional_private_file \
    "databases/$database_name-wal" \
    "$destination/$database_name-wal"
  copy_optional_private_file \
    "databases/$database_name-shm" \
    "$destination/$database_name-shm"
}

database_query() {
  local query="$1"
  local snapshot
  snapshot="$(mktemp -d "$temporary_directory/database.XXXXXX")"
  snapshot_database "$snapshot"
  "$sqlite_binary" -readonly "$snapshot/$database_name" \
    "PRAGMA query_only=ON; $query"
}

memo_directory_count() {
  adb_command shell run-as "$package_name" \
    find files/memos -mindepth 1 -maxdepth 1 -type d |
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
  local audio_path="$1"
  adb_command exec-out run-as "$package_name" cat "$audio_path" |
    sha256_stream
}

text_fingerprint() {
  local database="$1"
  local column="$2"
  local value_length
  local value_hash
  value_length="$(
    "$sqlite_binary" -readonly "$database" \
      "SELECT length(COALESCE($column, '')) FROM transcripts WHERE memoId='$memo_id';"
  )"
  value_hash="$(
    "$sqlite_binary" -readonly "$database" \
      "SELECT COALESCE($column, '') FROM transcripts WHERE memoId='$memo_id';" |
      sha256_stream
  )"
  printf '%s|%s\n' "${value_length:-0}" "$value_hash"
}

notification_or_service_is_active() {
  local services
  local notifications
  services="$(adb_command shell dumpsys activity services "$package_name" | tr -d '\r')"
  notifications="$(adb_command shell dumpsys notification --noredact | tr -d '\r')"
  grep -E -q \
    "RecordingService|SystemForegroundService" \
    <<<"$services" ||
    grep -E -q \
      "NotificationRecord.*pkg=${package_name}.*id=(1001|1002)([^0-9]|$)" \
      <<<"$notifications"
}

assert_idle() {
  local active_jobs
  adb_command exec-out run-as "$package_name" id >/dev/null 2>&1 ||
    fail "VoiceMemo is not installed as a debuggable build."
  active_jobs="$(
    database_query \
      "SELECT COUNT(*) FROM transcription_jobs
       WHERE status IN ('QUEUED','RUNNING','PAUSED_FOR_RECORDING','CANCEL_REQUESTED');"
  )"
  [[ "$active_jobs" =~ ^[0-9]+$ ]] ||
    fail "Could not read the current transcription state."
  [[ "$active_jobs" -eq 0 ]] ||
    fail "$active_jobs VoiceMemo transcription job(s) are active."
  if notification_or_service_is_active; then
    fail "A VoiceMemo recording or foreground transcription service is active."
  fi
}

read_screen_size() {
  local size
  size="$(
    adb_command shell wm size |
      tr -d '\r' |
      awk -F': ' '/size:/ { value=$2 } END { print value }'
  )"
  if [[ "$size" =~ ^([0-9]+)x([0-9]+)$ ]]; then
    screen_width="${BASH_REMATCH[1]}"
    screen_height="${BASH_REMATCH[2]}"
  fi
}

dump_ui() {
  adb_command shell uiautomator dump "$device_ui_path" >/dev/null
  adb_command exec-out cat "$device_ui_path" >"$host_ui_path"
}

node_center() {
  local resource_id="$1"
  "$python_binary" "$node_helper" "$host_ui_path" \
    --resource-id "$resource_id"
}

wait_for_tag() {
  local resource_id="$1"
  local seconds="$2"
  local deadline=$((SECONDS + seconds))
  local center
  while (( SECONDS < deadline )); do
    if dump_ui && center="$(node_center "$resource_id" 2>/dev/null)"; then
      printf '%s\n' "$center"
      return 0
    fi
    sleep 1
  done
  return 1
}

tap_center() {
  local center="$1"
  local x
  local y
  read -r x y <<<"$center"
  adb_command shell input tap "$x" "$y"
}

scroll_to_top() {
  local x=$((screen_width / 2))
  local start_y=$((screen_height * 35 / 100))
  local end_y=$((screen_height * 82 / 100))
  local attempt
  for attempt in {1..6}; do
    adb_command shell input swipe "$x" "$start_y" "$x" "$end_y" 250
  done
}

scroll_down_once() {
  local x=$((screen_width / 2))
  local start_y=$((screen_height * 82 / 100))
  local end_y=$((screen_height * 32 / 100))
  adb_command shell input swipe "$x" "$start_y" "$x" "$end_y" 300
}

find_tag_while_scrolling() {
  local resource_id="$1"
  local attempts="$2"
  local attempt
  local center
  for ((attempt = 0; attempt < attempts; attempt++)); do
    dump_ui
    if center="$(node_center "$resource_id" 2>/dev/null)"; then
      printf '%s\n' "$center"
      return 0
    fi
    scroll_down_once
  done
  return 1
}

navigate_to_library() {
  local attempt
  if wait_for_tag "library_screen" 3 >/dev/null; then
    return
  fi
  for attempt in {1..4}; do
    adb_command shell input keyevent KEYCODE_BACK
    if wait_for_tag "library_screen" 3 >/dev/null; then
      return
    fi
  done
  fail "Could not reach the library. Unlock the device and leave VoiceMemo visible."
}

latest_job() {
  database_query \
    "SELECT id || '|' || status || '|' || phase || '|' ||
            totalChunks || '|' || completedChunks || '|' ||
            COALESCE(errorCode, '')
     FROM transcription_jobs
     WHERE memoId='$memo_id'
     ORDER BY createdAtMillis DESC
     LIMIT 1;"
}

write_summary() {
  local status="$1"
  local job_id="$2"
  local memo_count_before="$3"
  local memo_count_after="$4"
  local audio_before="$5"
  local audio_after="$6"
  local edited_before="$7"
  local edited_after="$8"
  {
    echo "status=$status"
    echo "device_serial=$device_serial"
    echo "memo_id=$memo_id"
    echo "job_id=$job_id"
    echo "memo_count_before=$memo_count_before"
    echo "memo_count_after=$memo_count_after"
    echo "audio_sha256_before=$audio_before"
    echo "audio_sha256_after=$audio_after"
    echo "edited_fingerprint_before=$edited_before"
    echo "edited_fingerprint_after=$edited_after"
  } >"$result_directory/summary.txt"
}

[[ "${1:-}" == "run" ]] || {
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
    --serial)
      [[ "$#" -ge 2 ]] || fail "--serial requires a value."
      device_serial="$2"
      shift 2
      ;;
    --install-apk)
      [[ "$#" -ge 2 ]] || fail "--install-apk requires a path."
      install_apk="$2"
      shift 2
      ;;
    --timeout-minutes)
      [[ "$#" -ge 2 ]] || fail "--timeout-minutes requires a value."
      timeout_minutes="$2"
      shift 2
      ;;
    --output)
      [[ "$#" -ge 2 ]] || fail "--output requires a path."
      result_directory="$2"
      shift 2
      ;;
    --reference-db)
      [[ "$#" -ge 2 ]] || fail "--reference-db requires a path."
      reference_database="$2"
      shift 2
      ;;
    --reference-job)
      [[ "$#" -ge 2 ]] || fail "--reference-job requires a value."
      reference_job="$2"
      shift 2
      ;;
    --save-result-db)
      [[ "$#" -ge 2 ]] || fail "--save-result-db requires a directory."
      save_result_database_directory="$2"
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

require_command "$adb_binary"
require_command "$python_binary"
require_command "$sqlite_binary"
[[ -f "$node_helper" ]] || fail "Missing UI selector helper: $node_helper"
[[ -f "$parity_helper" ]] || fail "Missing parity helper: $parity_helper"
[[ "$memo_id" =~ ^[A-Za-z0-9-]+$ ]] ||
  fail "--memo-id may contain only letters, digits, and hyphens."
[[ "$timeout_minutes" =~ ^[1-9][0-9]*$ ]] ||
  fail "--timeout-minutes must be a positive integer."
if [[ -n "$reference_database" || -n "$reference_job" ]]; then
  [[ -f "$reference_database" ]] ||
    fail "--reference-db must be an existing file."
  [[ "$reference_job" =~ ^[A-Za-z0-9-]+$ ]] ||
    fail "--reference-job is required and must be an ID."
fi
if [[ -n "$install_apk" ]]; then
  [[ -f "$install_apk" ]] || fail "APK not found: $install_apk"
fi
if [[ -n "$save_result_database_directory" ]]; then
  [[ ! -e "$save_result_database_directory" ]] ||
    fail "--save-result-db must name a directory that does not exist."
fi

resolve_device
temporary_directory="$(mktemp -d /tmp/voicememo-ui-test.XXXXXX)"
device_ui_path="/data/local/tmp/voicememo-ui-test-$$.xml"
host_ui_path="$temporary_directory/window.xml"
if [[ -z "$result_directory" ]]; then
  result_directory="$script_directory/../build/reports/device-ui/$(date +%Y%m%d-%H%M%S)"
fi
[[ ! -e "$result_directory" ]] ||
  fail "Output already exists: $result_directory"
mkdir -p "$result_directory"

assert_idle
memo_exists="$(
  database_query \
    "SELECT COUNT(*) FROM memos WHERE id='$memo_id' AND pendingDeletion=0;"
)"
[[ "$memo_exists" == "1" ]] || fail "Memo $memo_id does not exist."
audio_path="$(
  database_query "SELECT audioPath FROM memos WHERE id='$memo_id';"
)"
[[ -n "$audio_path" ]] || fail "Memo $memo_id has no audio path."

memo_count_before="$(memo_directory_count)"
audio_before="$(audio_sha256 "$audio_path")"
pre_snapshot="$temporary_directory/pre"
snapshot_database "$pre_snapshot"
edited_before="$(text_fingerprint "$pre_snapshot/$database_name" editedText)"

if [[ -n "$install_apk" ]]; then
  echo "Installing VoiceMemo in place: $install_apk"
  adb_command install -r "$install_apk"
  assert_idle
  [[ "$(memo_directory_count)" == "$memo_count_before" ]] ||
    fail "Memo directory count changed during the in-place update."
  [[ "$(audio_sha256 "$audio_path")" == "$audio_before" ]] ||
    fail "Target memo audio changed during the in-place update."
fi

read_screen_size
adb_command shell am start -W -n "$activity_name" >/dev/null
wait_for_tag "voice_memo_app" 12 >/dev/null ||
  fail "VoiceMemo did not become visible. Unlock the device and try again."
navigate_to_library
scroll_to_top

memo_center="$(
  find_tag_while_scrolling "memo_card_$memo_id" 40
)" || fail "Could not find memo $memo_id in the library."
tap_center "$memo_center"
wait_for_tag "detail_screen" 10 >/dev/null ||
  fail "The memo detail screen did not open."

transcript_center="$(
  find_tag_while_scrolling "detail_section_transcript" 12
)" || fail "Could not find the Transcript section control."
tap_center "$transcript_center"

action_center=""
for attempt in {1..30}; do
  dump_ui
  if action_center="$(node_center "retranscribe_button" 2>/dev/null)"; then
    break
  fi
  if action_center="$(node_center "transcribe_button" 2>/dev/null)"; then
    break
  fi
  action_center=""
  scroll_down_once
done
[[ -n "$action_center" ]] ||
  fail "Could not find Transcribe or Transcribe again."

before_job_line="$(latest_job)"
before_job_id="${before_job_line%%|*}"
tap_center "$action_center"
echo "Transcription requested for memo $memo_id."

deadline=$((SECONDS + timeout_minutes * 60))
new_job_id=""
final_status=""
last_progress=""
while (( SECONDS < deadline )); do
  job_line="$(latest_job)"
  if [[ -n "$job_line" ]]; then
    IFS='|' read -r job_id status phase total completed error_code <<<"$job_line"
    if [[ "$job_id" != "$before_job_id" ]]; then
      new_job_id="$job_id"
      progress="$status/$phase $completed/$total"
      if [[ "$progress" != "$last_progress" ]]; then
        echo "$progress"
        last_progress="$progress"
      fi
      case "$status" in
        SUCCEEDED | FAILED | CANCELLED)
          final_status="$status"
          break
          ;;
      esac
    fi
  fi
  sleep 3
done

[[ -n "$new_job_id" ]] ||
  fail "No new transcription job appeared after the UI action."
if [[ -z "$final_status" ]]; then
  fail "Timed out after $timeout_minutes minutes; job $new_job_id was left running."
fi

post_snapshot="$temporary_directory/post"
snapshot_database "$post_snapshot"
memo_count_after="$(memo_directory_count)"
audio_after="$(audio_sha256 "$audio_path")"
edited_after="$(text_fingerprint "$post_snapshot/$database_name" editedText)"
write_summary \
  "$final_status" \
  "$new_job_id" \
  "$memo_count_before" \
  "$memo_count_after" \
  "$audio_before" \
  "$audio_after" \
  "$edited_before" \
  "$edited_after"

[[ "$memo_count_after" == "$memo_count_before" ]] ||
  fail "Memo directory count changed from $memo_count_before to $memo_count_after."
[[ "$audio_after" == "$audio_before" ]] ||
  fail "The target memo audio bytes changed."
[[ "$edited_after" == "$edited_before" ]] ||
  fail "The preserved edited transcript changed."
[[ "$final_status" == "SUCCEEDED" ]] || {
  error_code="$(
    "$sqlite_binary" -readonly "$post_snapshot/$database_name" \
      "SELECT COALESCE(errorCode, '') FROM transcription_jobs WHERE id='$new_job_id';"
  )"
  fail "Job $new_job_id ended as $final_status (${error_code:-no error code})."
}

copy_private_file "files/benchmarks/latest.json" \
  "$result_directory/benchmark.json" || true

if [[ -n "$save_result_database_directory" ]]; then
  mkdir -p "$save_result_database_directory"
  cp "$post_snapshot/$database_name" \
    "$save_result_database_directory/$database_name"
  for database_suffix in -wal -shm; do
    if [[ -f "$post_snapshot/$database_name$database_suffix" ]]; then
      cp "$post_snapshot/$database_name$database_suffix" \
        "$save_result_database_directory/$database_name$database_suffix"
    fi
  done
  chmod 600 "$save_result_database_directory"/"$database_name"*
  echo "Saved private Room snapshot: $save_result_database_directory"
fi

if [[ -n "$reference_database" ]]; then
  set +e
  "$python_binary" "$parity_helper" \
    --reference-db "$reference_database" \
    --reference-job "$reference_job" \
    --actual-db "$post_snapshot/$database_name" \
    --actual-job "$new_job_id" |
    tee "$result_directory/parity.txt"
  parity_status="${PIPESTATUS[0]}"
  set -e
  [[ "$parity_status" -eq 0 ]] ||
    fail "Exact CPU token/text parity failed. See $result_directory/parity.txt"
fi

echo "VoiceMemo UI test passed."
echo "Job:     $new_job_id"
echo "Results: $result_directory"
