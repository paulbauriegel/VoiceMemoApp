#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
tool_dir="$project_dir/tools/whisper-decoder-split"
source_model="$project_dir/build/generated/whisper-float-decoder-probe/whisper_large_v3_turbo_decode_float32_after_2_blocks.tflite"
output_model="$project_dir/build/generated/whisper-gpu-graph/whisper_large_v3_turbo_decode_float32_aligned_residuals.tflite"

usage() {
  cat <<'EOF'
Usage: scripts/generate-whisper-gpu-compatible-float-decoder.sh [options]

Creates an isolated FLOAT32 Whisper decoder diagnostic that preserves the
[1,128] token input and CPU GATHER, but aligns position and out-projection
residual tensors to [1,128,1280]. This removes the broadcast ADD barriers so
the remaining 155 operations can form one GPU partition.

The production app model is never changed.

Options:
  --source FILE   Pinned FLOAT32 two-block decoder control.
  --output FILE   Rewritten diagnostic model under a build directory.
  -h, --help      Show this help.
EOF
}

fail() {
  echo "Whisper GPU graph rewrite failed: $*" >&2
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
  com.example.voicememo.tools.WhisperGpuGraphRewriteTool \
  generate "$source_model" "$output_model"
