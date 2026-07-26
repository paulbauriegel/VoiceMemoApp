#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIRECTORY="$(cd -- "$SCRIPT_DIRECTORY/.." && pwd)"
readonly GENERATED_DIRECTORY="$PROJECT_DIRECTORY/build/generated/litert-strict-math-probe"
readonly EXPECTED_SOURCE_SHA256="6bbbf3e1fedb7504d9f4ea492b9ee35b9d6e1185476601c987d7ec88a4ba31ed"
readonly ACCELERATOR_ENTRY="jni/arm64-v8a/libLiteRtClGlAccelerator.so"
readonly EXPECTED_ACCELERATOR_SHA256="4b19f18f4ba9b1bde6060def4388b74d07f939db798c8c77c4f4e5125aeabcb1"
readonly EXPECTED_FLAG="-cl-fast-relaxed-math"
readonly EXPECTED_FLAG_OFFSET_HEX="0x510c7"

gradle_user_home="${GRADLE_USER_HOME:-$HOME/.gradle}"
default_source_aar="$gradle_user_home/caches/modules-2/files-2.1/com.google.ai.edge.litert/litert/2.1.6/58482f9a401129516773d98850e9f811db151de7/litert-2.1.6.aar"
source_aar="${1:-$default_source_aar}"
output_aar="${2:-$GENERATED_DIRECTORY/litert-2.1.6-strict-math-probe.aar}"

case "$output_aar" in
    /*) ;;
    *) output_aar="$PROJECT_DIRECTORY/$output_aar" ;;
esac

fail() {
    echo "LiteRT strict-math probe preparation failed: $*" >&2
    exit 1
}

[[ -f "$source_aar" ]] || fail "source AAR not found: $source_aar"

mkdir -p "$GENERATED_DIRECTORY"

resolved_generated_directory="$(
    python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' \
        "$GENERATED_DIRECTORY"
)"
resolved_output_aar="$(
    python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' \
        "$output_aar"
)"
case "$resolved_output_aar" in
    "$resolved_generated_directory"/*) ;;
    *)
        fail "output must stay under $resolved_generated_directory"
        ;;
esac
output_aar="$resolved_output_aar"
mkdir -p "$(dirname -- "$output_aar")"

source_sha256="$(
    python3 -c \
        'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
        "$source_aar"
)"
[[ "$source_sha256" == "$EXPECTED_SOURCE_SHA256" ]] || fail \
    "source AAR SHA-256 is $source_sha256; expected $EXPECTED_SOURCE_SHA256"

work_directory="$(mktemp -d "$GENERATED_DIRECTORY/.strict-math-probe.XXXXXX")"
cleanup() {
    case "$work_directory" in
        "$GENERATED_DIRECTORY"/.strict-math-probe.*)
            rm -rf -- "$work_directory"
            ;;
        *)
            echo "Refusing to remove unexpected work directory: $work_directory" >&2
            ;;
    esac
}
trap cleanup EXIT

python3 - \
    "$source_aar" \
    "$work_directory" \
    "$output_aar" \
    "$ACCELERATOR_ENTRY" \
    "$EXPECTED_ACCELERATOR_SHA256" \
    "$EXPECTED_FLAG" \
    "$EXPECTED_FLAG_OFFSET_HEX" <<'PY'
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import zipfile

(
    source_aar_argument,
    work_directory_argument,
    output_aar_argument,
    accelerator_entry,
    expected_accelerator_sha256,
    expected_flag_text,
    expected_flag_offset_text,
) = sys.argv[1:]

source_aar = Path(source_aar_argument)
work_directory = Path(work_directory_argument)
output_aar = Path(output_aar_argument)
exploded_directory = work_directory / "exploded"
original_accelerator = work_directory / "libLiteRtClGlAccelerator.original.so"
temporary_aar = work_directory / "litert-2.1.6-strict-math-probe.aar"
expected_flag = expected_flag_text.encode("ascii")
expected_flag_offset = int(expected_flag_offset_text, 16)


def fail(message: str) -> None:
    raise SystemExit(f"LiteRT strict-math probe preparation failed: {message}")


def validate_entry_name(name: str) -> PurePosixPath:
    if "\\" in name:
        fail(f"archive entry contains a backslash: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        fail(f"unsafe archive entry: {name!r}")
    return path


with zipfile.ZipFile(source_aar, "r") as source_zip:
    source_infos = source_zip.infolist()
    source_names = [info.filename for info in source_infos]
    if len(source_names) != len(set(source_names)):
        fail("source AAR contains duplicate entries")
    if accelerator_entry not in source_names:
        fail(f"source AAR is missing {accelerator_entry}")
    if source_zip.testzip() is not None:
        fail("source AAR failed its ZIP CRC check")

    exploded_directory.mkdir()
    for info in source_infos:
        archive_path = validate_entry_name(info.filename)
        unix_mode = info.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            fail(f"source AAR contains a symbolic link: {info.filename}")
        destination = exploded_directory.joinpath(*archive_path.parts)
        if info.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source_zip.open(info, "r") as source_stream, destination.open("wb") as output_stream:
            shutil.copyfileobj(source_stream, output_stream)

accelerator_path = exploded_directory.joinpath(*PurePosixPath(accelerator_entry).parts)
accelerator_data = accelerator_path.read_bytes()
accelerator_sha256 = hashlib.sha256(accelerator_data).hexdigest()
if accelerator_sha256 != expected_accelerator_sha256:
    fail(
        f"{accelerator_entry} SHA-256 is {accelerator_sha256}; "
        f"expected {expected_accelerator_sha256}"
    )
if accelerator_data.count(expected_flag) != 1:
    fail(
        f"expected exactly one {expected_flag_text!r} byte string in "
        f"{accelerator_entry}"
    )
actual_offset = accelerator_data.find(expected_flag)
if actual_offset != expected_flag_offset:
    fail(
        f"{expected_flag_text!r} starts at {actual_offset:#x}; "
        f"expected {expected_flag_offset:#x}"
    )
if accelerator_data[expected_flag_offset] != ord("-"):
    fail(
        f"byte at {expected_flag_offset:#x} is "
        f"{accelerator_data[expected_flag_offset]:#04x}; expected 0x2d"
    )

shutil.copyfile(accelerator_path, original_accelerator)
patched_accelerator = bytearray(accelerator_data)
patched_accelerator[expected_flag_offset] = 0
accelerator_path.write_bytes(patched_accelerator)

patched_data = accelerator_path.read_bytes()
if patched_data.count(expected_flag) != 0:
    fail(f"{expected_flag_text!r} remains after patching")
if patched_data[expected_flag_offset] != 0:
    fail(f"patched byte at {expected_flag_offset:#x} is not NUL")
if patched_data[expected_flag_offset + 1 : expected_flag_offset + len(expected_flag)] != (
    expected_flag[1:]
):
    fail("bytes after the patched flag prefix changed unexpectedly")
changed_offsets = [
    index
    for index, (before, after) in enumerate(zip(accelerator_data, patched_data))
    if before != after
]
if changed_offsets != [expected_flag_offset] or len(accelerator_data) != len(patched_data):
    fail(f"patch changed unexpected bytes: {changed_offsets[:16]}")

# Use stored, sorted entries and normalized metadata. This makes the generated
# archive independent of zlib versions while preserving every AAR payload byte
# except the single intentional byte above.
with zipfile.ZipFile(temporary_aar, "w", allowZip64=True) as output_zip:
    for source_info in sorted(source_infos, key=lambda info: info.filename):
        archive_path = validate_entry_name(source_info.filename)
        source_path = exploded_directory.joinpath(*archive_path.parts)
        output_info = zipfile.ZipInfo(source_info.filename, (2010, 1, 1, 0, 0, 0))
        output_info.create_system = 3
        output_info.compress_type = zipfile.ZIP_STORED
        output_info.flag_bits = 0
        if source_info.is_dir():
            output_info.external_attr = (stat.S_IFDIR | 0o755) << 16 | 0x10
            output_zip.writestr(output_info, b"")
        else:
            output_info.external_attr = (stat.S_IFREG | 0o644) << 16
            output_zip.writestr(output_info, source_path.read_bytes())

with zipfile.ZipFile(source_aar, "r") as source_zip, zipfile.ZipFile(temporary_aar, "r") as output_zip:
    if output_zip.testzip() is not None:
        fail("generated AAR failed its ZIP CRC check")
    if sorted(source_zip.namelist()) != output_zip.namelist():
        fail("generated AAR entries differ from source entries")
    for name in output_zip.namelist():
        source_bytes = source_zip.read(name)
        output_bytes = output_zip.read(name)
        if name == accelerator_entry:
            differing = [
                index
                for index, (before, after) in enumerate(zip(source_bytes, output_bytes))
                if before != after
            ]
            if differing != [expected_flag_offset] or len(source_bytes) != len(output_bytes):
                fail(f"generated AAR changed unexpected accelerator bytes: {differing[:16]}")
        elif source_bytes != output_bytes:
            fail(f"generated AAR changed unexpected entry: {name}")

output_aar.parent.mkdir(parents=True, exist_ok=True)
os.replace(temporary_aar, output_aar)
PY

original_accelerator="$work_directory/libLiteRtClGlAccelerator.original.so"
patched_accelerator="$work_directory/exploded/$ACCELERATOR_ENTRY"

find_android_llvm_tool() {
    local tool_name="$1"
    if command -v "$tool_name" >/dev/null 2>&1; then
        command -v "$tool_name"
        return 0
    fi
    local android_sdk_root="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
    local candidate
    for candidate in \
        "$android_sdk_root"/ndk/*/toolchains/llvm/prebuilt/*/bin/"$tool_name"
    do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if llvm_readelf="$(find_android_llvm_tool llvm-readelf)"; then
    "$llvm_readelf" -d "$original_accelerator" >"$work_directory/original.dynamic"
    "$llvm_readelf" -d "$patched_accelerator" >"$work_directory/patched.dynamic"
    cmp -s "$work_directory/original.dynamic" "$work_directory/patched.dynamic" || fail \
        "ELF dynamic metadata changed"
    grep -Eq \
        '\(SONAME\).*libLiteRtClGlAccelerator\.so' \
        "$work_directory/patched.dynamic" || fail \
        "patched library does not retain the expected SONAME"
    echo "Verified ELF SONAME and dynamic metadata with $llvm_readelf."
else
    echo "Warning: llvm-readelf unavailable; skipped SONAME verification." >&2
fi

if llvm_nm="$(find_android_llvm_tool llvm-nm)"; then
    "$llvm_nm" -D --defined-only "$original_accelerator" >"$work_directory/original.symbols"
    "$llvm_nm" -D --defined-only "$patched_accelerator" >"$work_directory/patched.symbols"
    cmp -s "$work_directory/original.symbols" "$work_directory/patched.symbols" || fail \
        "exported ELF symbols changed"
    echo "Verified exported ELF symbols with $llvm_nm."
else
    echo "Warning: llvm-nm unavailable; skipped exported-symbol verification." >&2
fi

if command -v strings >/dev/null 2>&1; then
    if LC_ALL=C strings -a "$patched_accelerator" | grep -F -- "$EXPECTED_FLAG" >/dev/null; then
        fail "$EXPECTED_FLAG remains visible in patched library strings"
    fi
    echo "Verified OpenCL fast-relaxed-math flag absence with strings."
else
    echo "Warning: strings unavailable; byte-level flag verification still passed." >&2
fi

generated_sha256="$(
    python3 -c \
        'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
        "$output_aar"
)"
generated_bytes="$(wc -c <"$output_aar" | tr -d '[:space:]')"
sha_file="$output_aar.sha256"
printf '%s  %s\n' "$generated_sha256" "$(basename -- "$output_aar")" >"$sha_file"

echo "Generated strict-math GPU probe AAR: $output_aar"
echo "Generated AAR bytes: $generated_bytes"
echo "Generated AAR SHA-256: $generated_sha256"
echo "Wrote checksum: $sha_file"
