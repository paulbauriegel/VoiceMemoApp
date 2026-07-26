#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL_DIR="${PROJECT_DIR}/tools/whisper-decoder-split"
DEFAULT_INPUT="${PROJECT_DIR}/app/src/main/generatedAssets/whisper/whisper_large_v3_turbo_30s_i4.tflite"
DEFAULT_OUTPUT="${PROJECT_DIR}/build/generated/whisper-decoder-split"

usage() {
  cat <<'EOF'
Generate or verify developer-only Whisper decoder-split models.

Usage:
  scripts/generate-whisper-decoder-split.sh generate [options]
  scripts/generate-whisper-decoder-split.sh verify [options]
  scripts/generate-whisper-decoder-split.sh generate-boundary --stage STAGE [options]
  scripts/generate-whisper-decoder-split.sh verify-boundary --stage STAGE [options]
  scripts/generate-whisper-decoder-split.sh self-test

Options:
  --input PATH                 Pinned original Whisper model.
  --output DIR                 Ignored output directory under build/.
  --stage STAGE                Boundary stage: after-position-add,
                               after-cross-attention, after-1-block,
                               after-2-blocks, or after-3-blocks.
  --force                      Regenerate already-valid outputs.
  --allow-dependency-download  Allow Gradle to fetch the pinned Java tooling.
  -h, --help                   Show this help.

By default Gradle runs offline and uses the cached FlatBuffer schema/runtime.
The generated files are diagnostics only. This script never changes the
production model asset or any connected device.
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

command_name="$1"
shift

if [[ "${command_name}" == "-h" || "${command_name}" == "--help" ]]; then
  usage
  exit 0
fi

case "${command_name}" in
  generate|verify|generate-boundary|verify-boundary|self-test) ;;
  *)
    echo "Unsupported command: ${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac

input_path="${DEFAULT_INPUT}"
output_path="${DEFAULT_OUTPUT}"
force=false
allow_dependency_download=false
boundary_stage=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      [[ $# -ge 2 ]] || {
        echo "--input requires a path." >&2
        exit 2
      }
      input_path="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || {
        echo "--output requires a directory." >&2
        exit 2
      }
      output_path="$2"
      shift 2
      ;;
    --force)
      force=true
      shift
      ;;
    --stage)
      [[ $# -ge 2 ]] || {
        echo "--stage requires a value." >&2
        exit 2
      }
      boundary_stage="$2"
      shift 2
      ;;
    --allow-dependency-download)
      allow_dependency_download=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${command_name}" == "self-test" ]]; then
  if [[ "${force}" == true || -n "${boundary_stage}" ]]; then
    echo "--force and --stage are not valid with self-test." >&2
    exit 2
  fi
elif [[ "${command_name}" == "verify" || "${command_name}" == "verify-boundary" ]] \
    && [[ "${force}" == true ]]; then
  echo "--force is only valid with generate." >&2
  exit 2
fi

if [[ "${command_name}" == "generate-boundary" \
    || "${command_name}" == "verify-boundary" ]]; then
  if [[ -z "${boundary_stage}" ]]; then
    echo "--stage is required for ${command_name}." >&2
    exit 2
  fi
elif [[ -n "${boundary_stage}" ]]; then
  echo "--stage is only valid with boundary commands." >&2
  exit 2
fi

java_is_17_or_newer() {
  local java_binary="$1"
  [[ -x "${java_binary}" ]] || return 1
  local version_output
  version_output="$("${java_binary}" -version 2>&1)" || return 1
  local version_line="${version_output%%$'\n'*}"
  if [[ "${version_line}" =~ \"1\.([0-9]+) ]]; then
    [[ "${BASH_REMATCH[1]}" -ge 17 ]]
  elif [[ "${version_line}" =~ \"([0-9]+) ]]; then
    [[ "${BASH_REMATCH[1]}" -ge 17 ]]
  else
    return 1
  fi
}

java_ready=false
if [[ -n "${JAVA_HOME:-}" ]] \
    && java_is_17_or_newer "${JAVA_HOME}/bin/java"; then
  java_ready=true
else
  path_java="$(command -v java 2>/dev/null || true)"
  if [[ -n "${path_java}" ]] && java_is_17_or_newer "${path_java}"; then
    # JAVA_HOME may point at an older runtime while PATH already selects a
    # suitable cross-platform JDK. Let the Gradle wrapper follow PATH.
    unset JAVA_HOME
    java_ready=true
  fi
fi

if [[ "${java_ready}" != true ]]; then
  java_17_home=""
  android_studio_jbr="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
  if java_is_17_or_newer "${android_studio_jbr}/bin/java"; then
    java_17_home="${android_studio_jbr}"
  elif [[ -x /usr/libexec/java_home ]]; then
    mac_java_home="$(/usr/libexec/java_home -v 17 2>/dev/null || true)"
    if java_is_17_or_newer "${mac_java_home}/bin/java"; then
      java_17_home="${mac_java_home}"
    fi
  fi
  if [[ -z "${java_17_home}" ]]; then
    echo "Java 17 or newer is required. Set JAVA_HOME to a JDK 17+ installation." >&2
    exit 1
  fi
  export JAVA_HOME="${java_17_home}"
fi

gradle_arguments=(
  -p "${TOOL_DIR}"
  --quiet
)
if [[ "${allow_dependency_download}" != true ]]; then
  gradle_arguments+=(--offline)
fi
gradle_arguments+=(installDist)

"${PROJECT_DIR}/gradlew" "${gradle_arguments[@]}"

launcher="${TOOL_DIR}/build/install/whisper-decoder-split/bin/whisper-decoder-split"
if [[ ! -x "${launcher}" ]]; then
  echo "Whisper decoder split launcher was not produced: ${launcher}" >&2
  exit 1
fi

tool_arguments=("${command_name}")
if [[ "${command_name}" != "self-test" ]]; then
  tool_arguments+=(
    --project-root "${PROJECT_DIR}"
    --input "${input_path}"
    --output "${output_path}"
  )
  if [[ "${force}" == true ]]; then
    tool_arguments+=(--force)
  fi
  if [[ -n "${boundary_stage}" ]]; then
    tool_arguments+=(--stage "${boundary_stage}")
  fi
fi

"${launcher}" "${tool_arguments[@]}"
