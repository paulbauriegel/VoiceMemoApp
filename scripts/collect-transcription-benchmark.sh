#!/usr/bin/env bash
set -euo pipefail

package_name="com.example.voicememo"
command_name="${1:-latest}"

require_device() {
  if ! adb get-state >/dev/null 2>&1; then
    echo "No authorized Android device is connected." >&2
    exit 1
  fi
}

print_latest() {
  adb exec-out run-as "$package_name" \
    cat files/benchmarks/latest.json 2>/dev/null |
    python3 -m json.tool
}

case "$command_name" in
  latest)
    require_device
    print_latest
    ;;
  history)
    require_device
    adb exec-out run-as "$package_name" \
      cat files/benchmarks/history.ndjson 2>/dev/null
    ;;
  watch)
    require_device
    echo "Waiting for VoiceMemo benchmark reports. Start a transcription on the device."
    adb logcat -v time -s VoiceMemoBench:I '*:S'
    ;;
  status)
    require_device
    device_pid="$(adb shell pidof "$package_name" | tr -d '\r')"
    if [[ -z "$device_pid" ]]; then
      echo "VoiceMemo is not currently running."
      exit 0
    fi
    adb shell top -H -b -n 1 -p "$device_pid" \
      -o PID,TID,CPU,%CPU,S,RES,TIME+,CMD |
      head -40
    adb shell dumpsys thermalservice |
      awk '/Thermal Status:|Current temperatures from HAL:/{show=1} show{print} /Current cooling devices from HAL:/{exit}'
    ;;
  *)
    echo "Usage: $0 {latest|history|watch|status}" >&2
    exit 2
    ;;
esac
