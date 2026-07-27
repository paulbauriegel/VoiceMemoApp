#!/usr/bin/env python3
"""Generate a diagnostic LiteRT AAR with mixed OpenCL precision enabled.

LiteRT 2.1.6 exposes FP16 and FP32 through its public API, but its shipped
OpenCL accelerator maps FP16 to CalculationsPrecision::F16. The underlying
accelerator also contains CalculationsPrecision::F32_F16. This script changes
only the FP16 switch arm from enum value 2 (F16) to enum value 1 (F32_F16):

    mov w10, #2  ->  mov w10, #1

The generated AAR is diagnostic-only and is accepted only when every pinned
source hash, archive entry, surrounding byte, and changed offset matches.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_AAR = (
    Path.home()
    / ".gradle/caches/modules-2/files-2.1/com.google.ai.edge.litert/"
    "litert/2.1.6/58482f9a401129516773d98850e9f811db151de7/litert-2.1.6.aar"
)
OUTPUT_AAR = (
    PROJECT_ROOT
    / "build/generated/litert-fp16-fp32-accum-probe/"
    "litert-2.1.6-fp16-fp32-accum-probe.aar"
)
SOURCE_AAR_SHA256 = "6bbbf3e1fedb7504d9f4ea492b9ee35b9d6e1185476601c987d7ec88a4ba31ed"
ACCELERATOR_ENTRY = "jni/arm64-v8a/libLiteRtClGlAccelerator.so"
SOURCE_ACCELERATOR_SHA256 = (
    "4b19f18f4ba9b1bde6060def4388b74d07f939db798c8c77c4f4e5125aeabcb1"
)
PATCH_OFFSET = 0xEB518
F16_INSTRUCTION = bytes.fromhex("4a008052")
F32_F16_INSTRUCTION = bytes.fromhex("2a008052")
EXPECTED_CONTEXT = bytes.fromhex(
    "280140f9020000144a0080522ab100b9020000143fb100b9"
)
CONTEXT_OFFSET = 0xEB510


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"LiteRT mixed-precision probe preparation failed: {message}")


def validate_name(name: str) -> PurePosixPath:
    if "\\" in name:
        fail(f"archive entry contains a backslash: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        fail(f"unsafe archive entry: {name!r}")
    return path


def main() -> None:
    source = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SOURCE_AAR
    output = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else OUTPUT_AAR
    generated_root = (
        PROJECT_ROOT / "build/generated/litert-fp16-fp32-accum-probe"
    ).resolve()
    if generated_root not in output.parents:
        fail(f"output must stay below {generated_root}")
    if not source.is_file():
        fail(f"source AAR is missing: {source}")
    actual_source_hash = digest_file(source)
    if actual_source_hash != SOURCE_AAR_SHA256:
        fail(
            f"source AAR SHA-256 is {actual_source_hash}; "
            f"expected {SOURCE_AAR_SHA256}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".mixed-probe-") as temp:
        exploded = Path(temp) / "exploded"
        temporary_output = Path(temp) / output.name
        with zipfile.ZipFile(source, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                fail("source AAR contains duplicate entries")
            if ACCELERATOR_ENTRY not in names:
                fail(f"source AAR is missing {ACCELERATOR_ENTRY}")
            if archive.testzip() is not None:
                fail("source AAR failed its ZIP CRC check")
            for info in infos:
                relative = validate_name(info.filename)
                unix_mode = info.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    fail(f"source AAR contains a symbolic link: {info.filename}")
                destination = exploded.joinpath(*relative.parts)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(info))

        accelerator = exploded.joinpath(*PurePosixPath(ACCELERATOR_ENTRY).parts)
        original = accelerator.read_bytes()
        actual_accelerator_hash = digest(original)
        if actual_accelerator_hash != SOURCE_ACCELERATOR_SHA256:
            fail(
                f"accelerator SHA-256 is {actual_accelerator_hash}; "
                f"expected {SOURCE_ACCELERATOR_SHA256}"
            )
        if original[CONTEXT_OFFSET : CONTEXT_OFFSET + len(EXPECTED_CONTEXT)] != (
            EXPECTED_CONTEXT
        ):
            fail("precision switch context does not match LiteRT 2.1.6")
        if original[PATCH_OFFSET : PATCH_OFFSET + 4] != F16_INSTRUCTION:
            fail("FP16 switch instruction is not the pinned `mov w10, #2`")

        patched = bytearray(original)
        patched[PATCH_OFFSET : PATCH_OFFSET + 4] = F32_F16_INSTRUCTION
        changed = [
            index
            for index, (before, after) in enumerate(zip(original, patched))
            if before != after
        ]
        if changed != [PATCH_OFFSET]:
            fail(f"unexpected changed offsets: {changed}")
        accelerator.write_bytes(patched)

        with zipfile.ZipFile(temporary_output, "w", allowZip64=True) as archive:
            for source_info in sorted(infos, key=lambda item: item.filename):
                relative = validate_name(source_info.filename)
                source_path = exploded.joinpath(*relative.parts)
                output_info = zipfile.ZipInfo(
                    source_info.filename,
                    (2010, 1, 1, 0, 0, 0),
                )
                output_info.create_system = 3
                output_info.compress_type = zipfile.ZIP_STORED
                output_info.flag_bits = 0
                if source_info.is_dir():
                    output_info.external_attr = (stat.S_IFDIR | 0o755) << 16 | 0x10
                    archive.writestr(output_info, b"")
                else:
                    output_info.external_attr = (stat.S_IFREG | 0o644) << 16
                    archive.writestr(output_info, source_path.read_bytes())

        with zipfile.ZipFile(temporary_output, "r") as archive:
            if archive.testzip() is not None:
                fail("generated AAR failed its ZIP CRC check")
            generated_accelerator = archive.read(ACCELERATOR_ENTRY)
            if generated_accelerator != bytes(patched):
                fail("generated AAR accelerator payload changed")
        shutil.move(temporary_output, output)

    print(f"Generated mixed-precision AAR: {output}")
    print(f"AAR bytes: {output.stat().st_size}")
    print(f"AAR SHA-256: {digest_file(output)}")
    print(f"Accelerator SHA-256: {digest(bytes(patched))}")
    print(f"Patched offset: {PATCH_OFFSET:#x}")


if __name__ == "__main__":
    main()
