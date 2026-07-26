#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
tool_dir="$project_dir/tools/whisper-decoder-split"
source_model="$project_dir/build/generated/whisper-decoder-split/whisper_large_v3_turbo_decode_shared_after_2_blocks.tflite"
output_model="$project_dir/build/generated/whisper-float-decoder-probe/whisper_large_v3_turbo_decode_float32_after_2_blocks.tflite"

usage() {
  cat <<'EOF'
Usage: scripts/generate-whisper-float-decoder-probe.sh [options]

Expands only the blockwise INT4 weights used by the first two decoder blocks
to exact FLOAT32 constants. The source and output are diagnostic artifacts
under build/; the production app model is never changed.

Options:
  --source FILE   Pinned after-two-blocks boundary model.
  --output FILE   FLOAT32 diagnostic model under a build directory.
  -h, --help      Show this help.
EOF
}

fail() {
  echo "Whisper FLOAT32 decoder control failed: $*" >&2
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --source)
      [[ "$#" -ge 2 ]] || fail "--source requires a value."
      source_model="$2"
      shift 2
      ;;
    --output)
      [[ "$#" -ge 2 ]] || fail "--output requires a value."
      output_model="$2"
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

[[ -f "$source_model" ]] || fail "Source model is missing: $source_model"
case "/$output_model/" in
  */build/*) ;;
  *) fail "Output must stay under a build directory." ;;
esac

java_home="${JAVA_HOME:-/Applications/Android Studio.app/Contents/jbr/Contents/Home}"
android_home="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
JAVA_HOME="$java_home" \
  ANDROID_HOME="$android_home" \
  PATH="$java_home/bin:$android_home/platform-tools:$PATH" \
  "$project_dir/gradlew" --no-daemon -p "$tool_dir" installDist

launcher_classpath="$tool_dir/build/install/whisper-decoder-split/lib/*"
"$java_home/bin/java" \
  -Xms128m \
  -Xmx1g \
  -cp "$launcher_classpath" \
  com.example.voicememo.tools.WhisperFloatDecoderProbeTool \
  generate "$source_model" "$output_model"
