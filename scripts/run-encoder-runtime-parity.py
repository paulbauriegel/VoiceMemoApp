#!/usr/bin/env python3
"""Run the decoder-free LiteRT/whisper.cpp encoder-state diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "scripts/run-whisper-benchmark-suite.py"
ENCODER = ROOT / "build/experiments/original-fp32-encoder/whisper-large-v3-turbo-original-fp32-encoder.tflite"
AUDIO = ROOT / "benchmark-inputs/whisper-suite/reference.wav"
WHISPER = ROOT / "build/model-assets/whisper.cpp/ggml-large-v3-turbo.bin"
ENCODER_SHA256 = "01be8e811608cc9eb3780408f8e020509d216a1d5e280320232b2c0a13d5e353"
BLOCK_ENCODER = ROOT / "build/experiments/encoder-block-diagnostic/whisper-large-v3-turbo-encoder-late-block-boundaries-v2.tflite"
BLOCK_ENCODER_SHA256 = "66a250a6c9df22af2c5f59f0d24e8a06699a57a79238489f08b40324393796b8"
WHISPER_SHA256 = "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"
PACKAGE = "com.example.voicememo"
TEST_PACKAGE = f"{PACKAGE}.test"
RUNNER = "androidx.test.runner.AndroidJUnitRunner"
TEST = "com.example.voicememo.transcription.EncoderRuntimeParityDiagnosticTest#compareFinalEncoderStates"
DEFAULT_CANDIDATES = ",".join(
    (
        "litert-cpu-fp32",
        "litert-opencl-fp32",
        "litert-opencl-fp16",
        "whispercpp-cpu-fp16-weights",
        "whispercpp-vulkan-fp16-weights",
    )
)


def load_suite():
    spec = importlib.util.spec_from_file_location("whisper_suite", SUITE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load benchmark-suite helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def private_bytes(adb: str, serial: str, relative: str) -> bytes:
    command = [adb, "-s", serial, "exec-out", "run-as", PACKAGE, "cat", relative]
    completed = subprocess.run(command, stdout=subprocess.PIPE, check=True)
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="2B211FDH300F4S")
    parser.add_argument("--adb", default=str(Path.home() / "Library/Android/sdk/platform-tools/adb"))
    parser.add_argument("--runtime-variant", default="standard")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--blocks", action="store_true")
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--encoder-sha256")
    parser.add_argument("--encoder-name")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--thermal-timeout-seconds", type=int, default=1200)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, default=ROOT / "build/reports/encoder-runtime-parity")
    args = parser.parse_args()

    suite = load_suite()
    serial = suite.one_device(args.adb, args.serial)
    suite.assert_device_idle(args.adb, serial)
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        f"encoder-parity-{args.runtime_variant}-%Y%m%dT%H%M%SZ"
    )
    output = args.output.resolve() / run_id
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    if args.blocks and args.encoder is not None:
        raise RuntimeError("--blocks and --encoder cannot be combined")
    encoder = (
        args.encoder.resolve()
        if args.encoder is not None
        else (BLOCK_ENCODER if args.blocks else ENCODER)
    )
    encoder_sha256 = (
        args.encoder_sha256
        or (
            BLOCK_ENCODER_SHA256
            if args.blocks
            else (ENCODER_SHA256 if args.encoder is None else suite.sha256(encoder))
        )
    )
    encoder_name = args.encoder_name or (
        "encoder-late-block-boundaries-v2.tflite"
        if args.blocks
        else (
            "original-openai-fp32-encoder.tflite"
            if args.encoder is None
            else encoder.name
        )
    )

    staged = [
        (encoder, f"files/gpu-models/{encoder_name}", encoder_sha256),
        (AUDIO, f"files/gpu-audio/{AUDIO.name}", suite.sha256(AUDIO)),
    ]
    if "whispercpp" in args.candidates:
        staged.append(
            (WHISPER, f"files/whispercpp-models/{WHISPER.name}", WHISPER_SHA256)
        )
    if not args.skip_stage:
        for source, relative, digest in staged:
            if not source.is_file() or suite.sha256(source) != digest:
                raise RuntimeError(f"Missing or invalid benchmark input: {source}")
            suite.stage_private_file(args.adb, serial, source, relative, digest)

    thermal_before = suite.wait_for_nominal_thermal(
        args.adb, serial, args.thermal_timeout_seconds
    )
    arguments = {
        "class": TEST.replace(
            "#compareFinalEncoderStates",
            "#compareEncoderBlockBoundaries" if args.blocks else "#compareFinalEncoderStates",
        ),
        "runId": run_id,
        "runtimeVariant": args.runtime_variant,
        "encoderModelName": encoder_name,
        "encoderModelSha256": encoder_sha256,
        "audioName": AUDIO.name,
        "whisperModelName": WHISPER.name,
        "candidates": args.candidates,
    }
    command = suite.adb(args.adb, serial, "shell", "am", "instrument", "-w", "-r")
    for key, value in arguments.items():
        command.extend(("-e", key, value))
    command.append(f"{TEST_PACKAGE}/{RUNNER}")
    completed = suite.run(command, capture=True, check=False)
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0 or "OK (1 test)" not in completed.stdout:
        raise RuntimeError("Encoder instrumentation failed")

    report_relative = f"cache/encoder-parity/{run_id}{'-blocks' if args.blocks else ''}.json"
    report = json.loads(private_bytes(args.adb, serial, report_relative))
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for candidate in report.get("candidates", []):
        state_name = candidate.get("state_file")
        if state_name:
            (output / state_name).write_bytes(
                private_bytes(
                    args.adb,
                    serial,
                    f"cache/encoder-parity/{state_name}",
                )
            )
        for boundary in candidate.get("boundaries", []):
            state_name = boundary.get("state_file")
            if state_name:
                (output / state_name).write_bytes(
                    private_bytes(
                        args.adb,
                        serial,
                        f"cache/encoder-parity/{state_name}",
                    )
                )
    summary = {
        "run_id": run_id,
        "runtime_variant": args.runtime_variant,
        "thermal_before": thermal_before,
        "thermal_after": suite.thermal_snapshot(args.adb, serial),
        "report": report,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output / "summary.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Encoder parity failed: {error}", file=sys.stderr)
        raise SystemExit(2)
