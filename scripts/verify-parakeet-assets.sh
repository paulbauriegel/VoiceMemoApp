#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSET_DIR="${1:-${PROJECT_DIR}/app/src/main/generatedAssets/parakeet}"

declare -a FILES=(
  "parakeet_tdt_0.6b_v3_5s_i8_stateful.tflite|614261072|334745b8bc7fd372b1c213516f0b6338bb827b1a2abb3e77ad35fe6fea5cd16b"
  "tokenizer.json|1159960|bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91"
  "processor_config.json|392|8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7"
  "config.json|1153|e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e"
)

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

size_of() {
  if stat -f '%z' "$1" >/dev/null 2>&1; then stat -f '%z' "$1"; else stat -c '%s' "$1"; fi
}

for item in "${FILES[@]}"; do
  IFS='|' read -r name expected_size expected_sha <<<"${item}"
  path="${ASSET_DIR}/${name}"
  [[ -f "${path}" ]] || {
    echo "Missing ${path}. Run scripts/fetch-parakeet-assets.sh." >&2
    exit 1
  }
  actual_size="$(size_of "${path}")"
  actual_sha="$(sha256_of "${path}")"
  [[ "${actual_size}" == "${expected_size}" && "${actual_sha}" == "${expected_sha}" ]] || {
    echo "Verification failed for ${path}" >&2
    echo "Expected: ${expected_size} ${expected_sha}" >&2
    echo "Actual:   ${actual_size} ${actual_sha}" >&2
    exit 1
  }
  echo "Verified ${name}"
done
