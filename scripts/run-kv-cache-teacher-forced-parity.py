#!/usr/bin/env python3
"""Locate the first full-vs-cached F32_F16 decoder divergence on the failing chunk."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "scripts/run-whisper-benchmark-suite.py"
PACKAGE = "com.example.voicememo"
TEST_PACKAGE = "com.example.voicememo.test"
TEST_RUNNER = "androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = (
    "com.example.voicememo.transcription."
    "KvCacheTeacherForcedParityTest#compareFailingChunk"
)
ENCODER = ROOT / "build/experiments/fp16-safe-layernorm-encoder/whisper-large-v3-turbo-fp16-safe-layernorm-encoder.tflite"
FULL_DECODER = ROOT / "build/experiments/original-fp32-decoder/whisper-large-v3-turbo-original-fp32-decoder.tflite"
KV_ROOT = ROOT / "build/experiments/original-decoder-kv-cache"
STEP = KV_ROOT / "decoder-kv-step.tflite"
CROSS = KV_ROOT / "decoder-cross-cache.tflite"
PROJECTION = ROOT / "build/experiments/original-decoder-active-row/active-row-projection-fp32.tflite"
AUDIO = ROOT / "benchmark-inputs/sample-test/audio.wav"
REFERENCE_REPORT = ROOT / "build/reports/five-sample-precision-matrix/five-sample-fp16-f32-f16-20260806/raw/sample-test-153s/litert-fp16-safe-fp16-f32-f16.json"


def load_suite():
    spec = importlib.util.spec_from_file_location("kv_parity_suite", SUITE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SUITE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_phase(suite, adb: str, serial: str, arguments: dict[str, str]) -> dict:
    command = suite.adb(adb, serial, "shell", "am", "instrument", "-w", "-r")
    for key, value in arguments.items():
        command.extend(("-e", key, value))
    command.append(f"{TEST_PACKAGE}/{TEST_RUNNER}")
    completed = suite.run(command, capture=True, check=False)
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode or "OK (1 test)" not in completed.stdout:
        raise RuntimeError(f"Teacher-forced {arguments['phase']} phase failed")
    return json.loads(
        suite.run(
            suite.adb(
                adb,
                serial,
                "exec-out",
                "run-as",
                PACKAGE,
                "cat",
                f"cache/kv-teacher-forced/{arguments['runId']}-{arguments['phase']}.json",
            ),
            capture=True,
        ).stdout
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="2B211FDH300F4S")
    parser.add_argument("--adb", default=str(Path.home() / "Library/Android/sdk/platform-tools/adb"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build/reports/kv-cache-teacher-forced-parity",
    )
    args = parser.parse_args()
    suite = load_suite()
    serial = suite.one_device(args.adb, args.serial)
    suite.assert_device_idle(args.adb, serial)
    reference = json.loads(REFERENCE_REPORT.read_text())
    tokens = reference["report"]["candidate"]["chunks"][1]["tokens"]
    run_id = datetime.now(timezone.utc).strftime("kv-parity-%Y%m%dT%H%M%SZ")
    artifacts = [ENCODER, FULL_DECODER, STEP, CROSS, PROJECTION, AUDIO]
    for artifact in artifacts:
        if not artifact.is_file():
            raise RuntimeError(f"Missing artifact: {artifact}")
    staged: list[str] = []
    try:
        for artifact in artifacts:
            directory = "gpu-audio" if artifact == AUDIO else "gpu-models"
            relative = f"files/{directory}/{artifact.name}"
            digest = suite.sha256(artifact)
            existing = suite.run(
                suite.adb(args.adb, serial, "shell", "run-as", PACKAGE, "sha256sum", relative),
                capture=True,
                check=False,
            )
            if existing.returncode == 0:
                if existing.stdout.split()[0] != digest:
                    raise RuntimeError(f"Refusing to replace changed private file: {relative}")
            else:
                suite.stage_private_file(args.adb, serial, artifact, relative, digest)
                staged.append(relative)
        common = {
            "class": TEST_CLASS,
            "runId": run_id,
            "encoderModel": ENCODER.name,
            "audioFile": AUDIO.name,
            "language": "de",
            "chunkIndex": "1",
            "referenceTokens": ",".join(map(str, tokens)),
        }
        full = run_phase(
            suite,
            args.adb,
            serial,
            common | {"phase": "full", "decoderModel": FULL_DECODER.name},
        )
        cached = run_phase(
            suite,
            args.adb,
            serial,
            common
            | {
                "phase": "kv",
                "crossModel": CROSS.name,
                "stepModel": STEP.name,
                "projectionModel": PROJECTION.name,
            },
        )
        output = args.output.resolve() / run_id
        output.mkdir(parents=True, exist_ok=False)
        (output / "full.json").write_text(json.dumps(full, indent=2) + "\n")
        (output / "kv.json").write_text(json.dumps(cached, indent=2) + "\n")
        print(output)
        print(json.dumps({
            "first_top1_mismatch": cached["first_top1_mismatch"],
            "all_top1_match": cached["all_top1_match"],
            "first_stable_top1_mismatch": cached["first_stable_top1_mismatch"],
            "all_stable_top1_match": cached["all_stable_top1_match"],
            "full_pss_kib": full["pss_kib"],
            "kv_pss_kib": cached["pss_kib"],
        }, indent=2))
    finally:
        suite.run(
            suite.adb(
                args.adb,
                serial,
                "shell",
                "run-as",
                PACKAGE,
                "rm",
                "-f",
                f"cache/kv-teacher-forced/{run_id}-encoder-state.bin",
                f"cache/kv-teacher-forced/{run_id}-full-logits.bin",
            ),
            check=False,
        )
        for relative in reversed(staged):
            suite.remove_private_file(args.adb, serial, relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
