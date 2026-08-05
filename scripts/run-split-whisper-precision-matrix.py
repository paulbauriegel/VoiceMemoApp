#!/usr/bin/env python3
"""Benchmark all FP32/FP16 encoder-decoder combinations of the split model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = PROJECT_ROOT / "scripts/run-whisper-benchmark-suite.py"
SPLIT_ROOT = PROJECT_ROOT / "build/experiments/whisper-fast-split"
ENCODER = SPLIT_ROOT / "encoder.tflite"
DECODER = SPLIT_ROOT / "decoder.tflite"
AUDIO = PROJECT_ROOT / "benchmark-inputs/whisper-suite/reference.wav"
BASELINE = PROJECT_ROOT / "benchmark-inputs/whisper-suite/baseline.json"
ENCODER_SHA256 = "a5ce67d0e53d5300c4c14509705e96f861a4d6fda7a6da3386cc3d272c54f55e"
DECODER_SHA256 = "d0e7fb4b4230fe4f41533fa649b305298b5df17a414b6af12d4eba5aa7bd2f06"
HISTORICAL_FP32_INFERENCE_MILLIS = 231_308
HISTORICAL_FP32_TOTAL_MILLIS = 247_300
HISTORICAL_FP32_WER_PERCENT = 1.8115942028985508
FP32_REPRODUCTION_TIME_TOLERANCE = 0.10
PACKAGE = "com.example.voicememo"
TEST_PACKAGE = f"{PACKAGE}.test"
TEST_RUNNER = "androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = (
    "com.example.voicememo.transcription."
    "GpuWholeMemoWerBenchmarkTest#compareWholeMemoBackends"
)

COMBINATIONS = (
    ("fp32-fp32", "fp32", "fp32"),
    ("fp32-fp16", "fp32", "fp16-native"),
    ("fp16-fp32", "fp16-native", "fp32"),
    ("fp16-fp16", "fp16-native", "fp16-native"),
)


class MatrixError(RuntimeError):
    pass


def load_suite_module():
    spec = importlib.util.spec_from_file_location("whisper_benchmark_suite", SUITE_PATH)
    if spec is None or spec.loader is None:
        raise MatrixError("Could not load the benchmark-suite helpers.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def verify_private_file(suite, adb_path: str, serial: str, relative: str, expected_sha256: str) -> None:
    completed = suite.run(
        suite.adb(
            adb_path,
            serial,
            "shell",
            "run-as",
            PACKAGE,
            "sha256sum",
            relative,
        ),
        capture=True,
        check=False,
    )
    actual = completed.stdout.split()[0] if completed.returncode == 0 else None
    if actual != expected_sha256:
        raise MatrixError(f"Staged diagnostic file is missing or changed: {relative}")


def run_candidate(suite, adb_path: str, serial: str, run_id: str, encoder_precision: str, decoder_precision: str):
    arguments = {
        "class": TEST_CLASS,
        "runId": run_id,
        "candidateOnly": "true",
        "candidateBackend": "gpu",
        "externalEncoderModelName": "split-fast-encoder.tflite",
        "externalEncoderModelSha256": ENCODER_SHA256,
        "externalDecoderModelName": "split-fast-decoder.tflite",
        "externalDecoderModelSha256": DECODER_SHA256,
        "externalAudioName": AUDIO.name,
        "forcedLanguage": "de",
        "timestampTokensEnabled": "false",
        "encoderGpuPrecision": encoder_precision,
        "decoderGpuPrecision": decoder_precision,
    }
    command = suite.adb(adb_path, serial, "shell", "am", "instrument", "-w", "-r")
    for key, value in arguments.items():
        command.extend(("-e", key, value))
    command.append(f"{TEST_PACKAGE}/{TEST_RUNNER}")
    completed = suite.run(command, capture=True, check=False)
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode != 0 or "OK (1 test)" not in completed.stdout:
        raise MatrixError(f"Instrumentation failed for {run_id}.")
    report_text = suite.run(
        suite.adb(
            adb_path,
            serial,
            "exec-out",
            "run-as",
            PACKAGE,
            "cat",
            f"cache/gpu-wer/whole-{run_id}.json",
        ),
        capture=True,
    ).stdout
    return json.loads(report_text)


def validate_report(report: dict, combination_id: str, encoder_precision: str, decoder_precision: str) -> None:
    candidate = report.get("candidate")
    expected_mode = (
        f"split-gpu-encoder-{encoder_precision.replace('-native', '')}"
        f"-decoder-{decoder_precision.replace('-native', '')}-no-source-quant"
    )
    checks = {
        "status": report.get("status") == "completed",
        "candidate": isinstance(candidate, dict),
        "encoder SHA": report.get("encoder_model_sha256") == ENCODER_SHA256,
        "decoder SHA": report.get("decoder_model_sha256") == DECODER_SHA256,
        "encoder precision": report.get("encoder_gpu_precision") == encoder_precision,
        "decoder precision": report.get("decoder_gpu_precision") == decoder_precision,
        "timestamps": report.get("timestamp_tokens_enabled") is False,
        "mode": isinstance(candidate, dict) and candidate.get("mode") == expected_mode,
        "timestamp count": isinstance(candidate, dict) and candidate.get("timestamp_token_count") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise MatrixError(f"Invalid {combination_id} report: {failed}")


def summarize(matrix_id: str, baseline: dict, results: dict[str, list[dict]]) -> dict:
    configurations = []
    audio_seconds = baseline["audio_samples_16khz"] / 16_000
    for combination_id, encoder_precision, decoder_precision in COMBINATIONS:
        rows = results[combination_id]
        inference = [row["report"]["candidate"]["inference_wall_millis"] for row in rows]
        total = [row["report"]["candidate"]["total_wall_millis"] for row in rows]
        wer = [row["comparison"]["word_error_percent"] for row in rows]
        pss = [row["report"]["candidate"]["memory_after_inference"]["total_pss_kib"] for row in rows]
        configurations.append(
            {
                "id": combination_id,
                "encoder_precision": encoder_precision,
                "decoder_precision": decoder_precision,
                "runs": len(rows),
                "inference_millis": {"median": statistics.median(inference), "min": min(inference), "max": max(inference)},
                "total_millis": {"median": statistics.median(total), "min": min(total), "max": max(total)},
                "real_time_factor": statistics.median(inference) / 1000 / audio_seconds,
                "word_error_percent": statistics.median(wer),
                "pss_kib": statistics.median(pss),
                "texts_identical": len({row["report"]["candidate"]["text"] for row in rows}) == 1,
                "tokens_identical": len({tuple(row["report"]["candidate"]["tokens"]) for row in rows}) == 1,
            }
        )
    return {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "status": "completed",
        "source_model_sha256": "a2613d64e6cb73ff40db7c548215fd086f096aa088c6cc113c833ccc87501e89",
        "encoder_model_sha256": ENCODER_SHA256,
        "decoder_model_sha256": DECODER_SHA256,
        "timestamp_tokens_enabled": False,
        "configurations": configurations,
    }


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        f"# Split Whisper precision matrix {summary['matrix_id']}",
        "",
        "| Encoder | Decoder | Inference | Total | RTF | WER | PSS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["configurations"]:
        lines.append(
            f"| {row['encoder_precision']} | {row['decoder_precision']} | "
            f"{row['inference_millis']['median'] / 1000:.3f} s | "
            f"{row['total_millis']['median'] / 1000:.3f} s | "
            f"{row['real_time_factor']:.3f} | {row['word_error_percent']:.3f}% | "
            f"{row['pss_kib'] / 1024 / 1024:.2f} GiB |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial")
    parser.add_argument("--adb", default=str(Path.home() / "Library/Android/sdk/platform-tools/adb"))
    parser.add_argument("--runs", type=int, choices=(1, 3), default=1)
    parser.add_argument("--cooldown-seconds", type=int, default=30)
    parser.add_argument("--thermal-timeout-seconds", type=int, default=900)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "build/reports/split-whisper-precision-matrix")
    parser.add_argument("--matrix-id")
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Stage and verify inputs without starting instrumentation.",
    )
    parser.add_argument(
        "--reuse-staged",
        action="store_true",
        help="Reuse and verify previously staged inputs instead of transferring them again.",
    )
    args = parser.parse_args()
    if args.stage_only and args.reuse_staged:
        raise MatrixError("--stage-only and --reuse-staged are mutually exclusive.")
    suite = load_suite_module()
    for path, expected in ((ENCODER, ENCODER_SHA256), (DECODER, DECODER_SHA256)):
        if not path.is_file() or suite.sha256(path) != expected:
            raise MatrixError(f"Split model is missing or changed: {path}")
    baseline = json.loads(BASELINE.read_text())
    serial = suite.one_device(args.adb, args.serial)
    suite.assert_device_idle(args.adb, serial)
    matrix_id = args.matrix_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output.resolve() / matrix_id
    if output.exists():
        raise MatrixError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    results = {item[0]: [] for item in COMBINATIONS}
    staged = []
    keep_staged = False
    diagnostic_files = (
        (AUDIO, f"files/gpu-audio/{AUDIO.name}", suite.sha256(AUDIO)),
        (ENCODER, "files/gpu-models/split-fast-encoder.tflite", ENCODER_SHA256),
        (DECODER, "files/gpu-models/split-fast-decoder.tflite", DECODER_SHA256),
    )
    try:
        if args.reuse_staged:
            for _, relative, digest in diagnostic_files:
                verify_private_file(suite, args.adb, serial, relative, digest)
        else:
            for source, relative, digest in diagnostic_files:
                suite.stage_private_file(args.adb, serial, source, relative, digest)
                staged.append(relative)
        if args.stage_only:
            keep_staged = True
            print(
                json.dumps(
                    {
                        "status": "staged",
                        "device_serial": serial,
                        "files": [
                            {"path": relative, "sha256": digest}
                            for _, relative, digest in diagnostic_files
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        for combination_id, encoder_precision, decoder_precision in COMBINATIONS:
            for run_number in range(1, args.runs + 1):
                if results[combination_id] or combination_id != COMBINATIONS[0][0]:
                    if args.cooldown_seconds:
                        time.sleep(args.cooldown_seconds)
                thermal_before = suite.wait_for_nominal_thermal(
                    args.adb, serial, args.thermal_timeout_seconds
                )
                suite.assert_device_idle(args.adb, serial)
                run_id = f"{matrix_id}-{combination_id}-{run_number}"
                report = run_candidate(
                    suite, args.adb, serial, run_id, encoder_precision, decoder_precision
                )
                validate_report(report, combination_id, encoder_precision, decoder_precision)
                thermal_after = suite.thermal_snapshot(args.adb, serial)
                compared = suite.comparison(baseline, report["candidate"])
                row = {
                    "thermal_before": thermal_before,
                    "thermal_after": thermal_after,
                    "comparison": compared,
                    "report": report,
                }
                results[combination_id].append(row)
                atomic_json(output / "raw" / combination_id / f"run-{run_number}.json", row)
                if combination_id == "fp32-fp32" and run_number == 1:
                    candidate = report["candidate"]
                    inference_delta = abs(
                        candidate["inference_wall_millis"] - HISTORICAL_FP32_INFERENCE_MILLIS
                    ) / HISTORICAL_FP32_INFERENCE_MILLIS
                    total_delta = abs(
                        candidate["total_wall_millis"] - HISTORICAL_FP32_TOTAL_MILLIS
                    ) / HISTORICAL_FP32_TOTAL_MILLIS
                    if (
                        compared["word_error_percent"] != HISTORICAL_FP32_WER_PERCENT
                        or inference_delta > FP32_REPRODUCTION_TIME_TOLERANCE
                        or total_delta > FP32_REPRODUCTION_TIME_TOLERANCE
                    ):
                        raise MatrixError(
                            "Split FP32/FP32 failed the original-model reproduction gate: "
                            f"inference={candidate['inference_wall_millis']} ms "
                            f"({inference_delta:.1%} delta), "
                            f"total={candidate['total_wall_millis']} ms "
                            f"({total_delta:.1%} delta), "
                            f"WER={compared['word_error_percent']:.6f}%."
                        )
        summary = summarize(matrix_id, baseline, results)
        atomic_json(output / "summary.json", summary)
        write_markdown(summary, output / "summary.md")
        print(output / "summary.md")
    finally:
        if not keep_staged:
            cleanup = staged or (
                [relative for _, relative, _ in diagnostic_files]
                if args.reuse_staged
                else []
            )
            for relative in reversed(cleanup):
                suite.remove_private_file(args.adb, serial, relative)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Split precision matrix failed: {error}", file=sys.stderr)
        raise SystemExit(2)
