#!/usr/bin/env python3
"""Run progressively narrower, decoder-free encoder precision diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "scripts/run-whisper-benchmark-suite.py"
PARITY_RUNNER = ROOT / "scripts/run-encoder-runtime-parity.py"
MODEL_BUILDER = ROOT / "scripts/build-encoder-stage-diagnostic-model.py"
SCHEMA_PYTHON = ROOT / "build/tools/tflite-schema-venv/bin/python"
BLOCK_ENCODER = ROOT / "build/experiments/encoder-block-diagnostic/whisper-large-v3-turbo-encoder-late-block-boundaries-v2.tflite"
BLOCK_ENCODER_SHA256 = "66a250a6c9df22af2c5f59f0d24e8a06699a57a79238489f08b40324393796b8"
LAYER_NORM_MODEL = ROOT / "build/experiments/encoder-stage-diagnostic/encoder-final-layernorm-v1.tflite"
LAYER_NORM_SHA256 = "2e1cda8a1a6cbe4312023148d96d3c6ec940d5ba5914d5e0f673de367707dcb5"
AUDIO = ROOT / "benchmark-inputs/whisper-suite/reference.wav"
PACKAGE = "com.example.voicememo"
TEST_PACKAGE = f"{PACKAGE}.test"
RUNNER = "androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "com.example.voicememo.transcription.EncoderRuntimeParityDiagnosticTest"
DEFAULT_LITERT_CANDIDATES = "litert-cpu-fp32,litert-opencl-fp32,litert-opencl-fp16"
DEFAULT_LATE_CANDIDATES = (
    "litert-cpu-fp32,litert-opencl-fp16,whispercpp-vulkan-fp16-weights"
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
    completed = subprocess.run(
        [adb, "-s", serial, "exec-out", "run-as", PACKAGE, "cat", relative],
        stdout=subprocess.PIPE,
        check=True,
    )
    return completed.stdout


def state_files(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "state_file" and isinstance(child, str):
                yield child
            else:
                yield from state_files(child)
    elif isinstance(value, list):
        for child in value:
            yield from state_files(child)


def instrument(
    suite: object,
    adb: str,
    serial: str,
    method: str,
    arguments: dict[str, str],
) -> None:
    command = suite.adb(adb, serial, "shell", "am", "instrument", "-w", "-r")
    values = {"class": f"{TEST_CLASS}#{method}", **arguments}
    for key, value in values.items():
        command.extend(("-e", key, value))
    command.append(f"{TEST_PACKAGE}/{RUNNER}")
    completed = suite.run(command, capture=True, check=False)
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0 or "OK (1 test)" not in completed.stdout:
        raise RuntimeError(f"Encoder diagnostic {method} failed")


def pull_report(
    adb: str,
    serial: str,
    run_id: str,
    suffix: str,
    output: Path,
    runtime_variant: str,
    thermal_before: dict,
    suite: object,
) -> None:
    report = json.loads(
        private_bytes(adb, serial, f"cache/encoder-parity/{run_id}-{suffix}.json")
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for name in sorted(set(state_files(report))):
        (output / name).write_bytes(
            private_bytes(adb, serial, f"cache/encoder-parity/{name}")
        )
    summary = {
        "run_id": run_id,
        "runtime_variant": runtime_variant,
        "thermal_before": thermal_before,
        "thermal_after": suite.thermal_snapshot(adb, serial),
        "report": report,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output / "summary.json")


def run_late_blocks(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(PARITY_RUNNER),
        "--blocks",
        "--serial",
        args.serial,
        "--adb",
        args.adb,
        "--runtime-variant",
        args.runtime_variant,
        "--candidates",
        args.candidates or DEFAULT_LATE_CANDIDATES,
        "--thermal-timeout-seconds",
        str(args.thermal_timeout_seconds),
        "--output",
        str(args.output),
    ]
    if args.run_id:
        command.extend(("--run-id", args.run_id))
    if args.skip_stage:
        command.append("--skip-stage")
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="2B211FDH300F4S")
    parser.add_argument("--adb", default=str(Path.home() / "Library/Android/sdk/platform-tools/adb"))
    parser.add_argument("--runtime-variant", default="standard")
    parser.add_argument("--candidates")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--thermal-timeout-seconds", type=int, default=1200)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, default=ROOT / "build/reports/encoder-precision-localization")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("late-blocks")
    stage = subparsers.add_parser("stage")
    stage.add_argument("--block", type=int, choices=range(24, 32), required=True)
    stage_choice = stage.add_mutually_exclusive_group(required=True)
    stage_choice.add_argument("--stage")
    stage_choice.add_argument("--stages", nargs="+")
    final_layernorm = subparsers.add_parser("final-layernorm")
    final_layernorm.add_argument(
        "--late-block-report",
        type=Path,
        required=True,
        help="Completed late-block report directory whose saved LiteRT tensors are cross-fed.",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode == "late-blocks":
        return run_late_blocks(args)

    suite = load_suite()
    serial = suite.one_device(args.adb, args.serial)
    suite.assert_device_idle(args.adb, serial)
    if args.mode == "stage":
        stage_names = args.stages or [args.stage]
        label = f"block{args.block}-{'-'.join(stage_names)}"
    else:
        stage_names = []
        label = "final-layernorm"
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        f"encoder-{label}-{args.runtime_variant}-%Y%m%dT%H%M%SZ"
    )
    output = args.output / run_id
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    audio_sha = suite.sha256(AUDIO)
    if not args.skip_stage:
        suite.stage_private_file(
            args.adb,
            serial,
            AUDIO,
            f"files/gpu-audio/{AUDIO.name}",
            audio_sha,
        )

    thermal_before = suite.wait_for_nominal_thermal(
        args.adb, serial, args.thermal_timeout_seconds
    )
    common = {
        "runId": run_id,
        "runtimeVariant": args.runtime_variant,
        "audioName": AUDIO.name,
    }
    if args.mode == "stage":
        with tempfile.TemporaryDirectory(prefix="encoder-stage-", dir=ROOT / "build/tmp") as temporary:
            model = Path(temporary) / "encoder-stage-diagnostic.tflite"
            python = SCHEMA_PYTHON if SCHEMA_PYTHON.is_file() else Path(sys.executable)
            build_command = [
                    str(python),
                    str(MODEL_BUILDER),
                    "--output",
                    str(model),
                    "stage-group" if len(stage_names) > 1 else "stage",
                    "--block",
                    str(args.block),
            ]
            if len(stage_names) > 1:
                build_command.extend(("--stages", *stage_names))
            else:
                build_command.extend(("--stage", stage_names[0]))
            subprocess.run(build_command, check=True)
            digest = suite.sha256(model)
            if not args.skip_stage:
                suite.stage_private_file(
                    args.adb,
                    serial,
                    model,
                    "files/gpu-models/encoder-stage-diagnostic.tflite",
                    digest,
                )
            instrument(
                suite,
                args.adb,
                serial,
                "compareEncoderStage",
                {
                    **common,
                    "encoderModelName": "encoder-stage-diagnostic.tflite",
                    "encoderModelSha256": digest,
                    "candidates": args.candidates or DEFAULT_LITERT_CANDIDATES,
                    "stageOutputs": ",".join(stage_names) if len(stage_names) > 1 else "stage",
                },
            )
        pull_report(
            args.adb,
            serial,
            run_id,
            "stage",
            output,
            args.runtime_variant,
            thermal_before,
            suite,
        )
    else:
        if not LAYER_NORM_MODEL.is_file() or suite.sha256(LAYER_NORM_MODEL) != LAYER_NORM_SHA256:
            raise RuntimeError(f"Missing or invalid diagnostic model: {LAYER_NORM_MODEL}")
        source_directory = args.late_block_report.resolve()
        source_report = json.loads((source_directory / "report.json").read_text(encoding="utf-8"))
        if source_report.get("status") != "completed":
            raise RuntimeError(f"Late-block report is not completed: {source_directory}")
        wanted = {
            ("litert-cpu-fp32", "block_31"): "cpu-block31.f32le",
            ("litert-opencl-fp16", "block_31"): "gpu-block31.f32le",
            ("litert-cpu-fp32", "output_0"): "cpu-final.f32le",
            ("litert-opencl-fp16", "output_0"): "gpu-final.f32le",
        }
        source_states: dict[str, tuple[Path, str]] = {}
        for candidate in source_report.get("candidates", []):
            for boundary in candidate.get("boundaries", []):
                staged_name = wanted.get((candidate.get("name"), boundary.get("name")))
                if staged_name is None:
                    continue
                source = source_directory / boundary["state_file"]
                if not source.is_file():
                    raise RuntimeError(f"Missing saved encoder state: {source}")
                source_states[staged_name] = (source, suite.sha256(source))
        if set(source_states) != set(wanted.values()):
            raise RuntimeError("Late-block report does not contain all required CPU/GPU states.")
        if not args.skip_stage:
            suite.stage_private_file(
                args.adb,
                serial,
                LAYER_NORM_MODEL,
                "files/gpu-models/encoder-final-layernorm-v1.tflite",
                LAYER_NORM_SHA256,
            )
            for staged_name, (source, digest) in source_states.items():
                suite.stage_private_file(
                    args.adb,
                    serial,
                    source,
                    f"files/encoder-states/{staged_name}",
                    digest,
                )
        instrument(
            suite,
            args.adb,
            serial,
            "compareFinalLayerNormCrossFeed",
            {
                **common,
                "layerNormModelName": "encoder-final-layernorm-v1.tflite",
                "layerNormModelSha256": LAYER_NORM_SHA256,
                "cpuBlock31Name": "cpu-block31.f32le",
                "cpuBlock31Sha256": source_states["cpu-block31.f32le"][1],
                "gpuBlock31Name": "gpu-block31.f32le",
                "gpuBlock31Sha256": source_states["gpu-block31.f32le"][1],
                "cpuFinalName": "cpu-final.f32le",
                "cpuFinalSha256": source_states["cpu-final.f32le"][1],
                "gpuFinalName": "gpu-final.f32le",
                "gpuFinalSha256": source_states["gpu-final.f32le"][1],
            },
        )
        pull_report(
            args.adb,
            serial,
            run_id,
            "layernorm",
            output,
            args.runtime_variant,
            thermal_before,
            suite,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Encoder precision localization failed: {error}", file=sys.stderr)
        raise SystemExit(2)
