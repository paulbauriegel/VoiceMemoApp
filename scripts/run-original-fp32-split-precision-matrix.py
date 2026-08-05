#!/usr/bin/env python3
"""Benchmark four GPU precisions using matched original OpenAI FP32 models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = PROJECT_ROOT / "scripts/run-whisper-benchmark-suite.py"
ENCODER = PROJECT_ROOT / "build/experiments/original-fp32-encoder/whisper-large-v3-turbo-original-fp32-encoder.tflite"
DECODER = PROJECT_ROOT / "build/experiments/original-fp32-decoder/whisper-large-v3-turbo-original-fp32-decoder.tflite"
AUDIO = PROJECT_ROOT / "benchmark-inputs/whisper-suite/reference.wav"
BASELINE = PROJECT_ROOT / "benchmark-inputs/whisper-suite/baseline.json"
ENCODER_SHA256 = "01be8e811608cc9eb3780408f8e020509d216a1d5e280320232b2c0a13d5e353"
DECODER_SHA256 = "d661702d7753165cfaf01bf5b060f46065df500959e42e5254fd2598a54dedb7"
CHECKPOINT_SHA256 = "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
PACKAGE = "com.example.voicememo"
TEST_PACKAGE = f"{PACKAGE}.test"
TEST_RUNNER = "androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "com.example.voicememo.transcription.GpuWholeMemoWerBenchmarkTest#compareWholeMemoBackends"

COMBINATIONS = (
    ("fp32-fp32", "fp32", "fp32"),
    ("fp32-fp16", "fp32", "fp16-native"),
    ("fp16-fp32", "fp16-native", "fp32"),
    ("fp16-fp16", "fp16-native", "fp16-native"),
)


class MatrixError(RuntimeError):
    pass


def load_suite():
    spec = importlib.util.spec_from_file_location("whisper_benchmark_suite", SUITE_PATH)
    if spec is None or spec.loader is None:
        raise MatrixError("Could not load benchmark-suite helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_candidate(
    suite,
    adb_path: str,
    serial: str,
    run_id: str,
    encoder_precision: str,
    decoder_precision: str,
    audio_name: str,
    forced_language: str,
) -> dict:
    arguments = {
        "class": TEST_CLASS,
        "runId": run_id,
        "candidateOnly": "true",
        "candidateBackend": "gpu",
        "externalEncoderModelName": "original-openai-fp32-encoder.tflite",
        "externalEncoderModelSha256": ENCODER_SHA256,
        "externalDecoderModelName": "original-openai-fp32-decoder.tflite",
        "externalDecoderModelSha256": DECODER_SHA256,
        "externalAudioName": audio_name,
        "forcedLanguage": forced_language,
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
        raise MatrixError(f"Instrumentation failed for {run_id}")
    text = suite.run(
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
    return json.loads(text)


def validate(report: dict, encoder_precision: str, decoder_precision: str) -> None:
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
        "timestamp mode": report.get("timestamp_tokens_enabled") is False,
        "mode": isinstance(candidate, dict) and candidate.get("mode") == expected_mode,
        "timestamp count": isinstance(candidate, dict) and candidate.get("timestamp_token_count") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise MatrixError(
            f"Invalid {encoder_precision}/{decoder_precision} report: {failed}; "
            f"error={report.get('error_type')} {report.get('error_message')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="2B211FDH300F4S")
    parser.add_argument("--adb", default=str(Path.home() / "Library/Android/sdk/platform-tools/adb"))
    parser.add_argument("--cooldown-seconds", type=int, default=30)
    parser.add_argument("--thermal-timeout-seconds", type=int, default=1200)
    parser.add_argument("--audio", type=Path, default=AUDIO)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--forced-language", default="de")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "build/reports/original-fp32-split-precision-matrix",
    )
    parser.add_argument("--matrix-id")
    args = parser.parse_args()
    args.audio = args.audio.expanduser().resolve()
    args.baseline = args.baseline.expanduser().resolve()
    suite = load_suite()
    for path, digest in ((ENCODER, ENCODER_SHA256), (DECODER, DECODER_SHA256)):
        if not path.is_file() or suite.sha256(path) != digest:
            raise MatrixError(f"Missing or changed model: {path}")
    if not args.audio.is_file() or not args.baseline.is_file():
        raise MatrixError("Audio or baseline file is missing")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("model_revision") != MODEL_REVISION:
        raise MatrixError("Baseline model revision does not match the split models")
    if baseline.get("language") != args.forced_language:
        raise MatrixError("Baseline language does not match --forced-language")
    serial = suite.one_device(args.adb, args.serial)
    suite.assert_device_idle(args.adb, serial)
    matrix_id = args.matrix_id or datetime.now(timezone.utc).strftime("original-openai-fp32-split-%Y%m%dT%H%M%SZ")
    output = args.output.resolve() / matrix_id
    if output.exists():
        raise MatrixError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    files = (
        (args.audio, f"files/gpu-audio/{args.audio.name}", suite.sha256(args.audio)),
        (ENCODER, "files/gpu-models/original-openai-fp32-encoder.tflite", ENCODER_SHA256),
        (DECODER, "files/gpu-models/original-openai-fp32-decoder.tflite", DECODER_SHA256),
    )
    staged: list[str] = []
    rows: list[dict] = []
    try:
        for source, relative, digest in files:
            suite.stage_private_file(args.adb, serial, source, relative, digest)
            staged.append(relative)
        for index, (configuration, encoder_precision, decoder_precision) in enumerate(COMBINATIONS):
            if index and args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)
            thermal_before = suite.wait_for_nominal_thermal(
                args.adb, serial, args.thermal_timeout_seconds
            )
            suite.assert_device_idle(args.adb, serial)
            run_id = f"{matrix_id}-{configuration}"
            report = run_candidate(
                suite,
                args.adb,
                serial,
                run_id,
                encoder_precision,
                decoder_precision,
                args.audio.name,
                args.forced_language,
            )
            validate(report, encoder_precision, decoder_precision)
            thermal_after = suite.thermal_snapshot(args.adb, serial)
            candidate = report["candidate"]
            comparison = suite.comparison(baseline, candidate)
            audio_seconds = baseline["audio_samples_16khz"] / 16_000
            row = {
                "configuration": configuration,
                "encoder_precision": encoder_precision,
                "decoder_precision": decoder_precision,
                "thermal_before": thermal_before,
                "thermal_after": thermal_after,
                "model_load_millis": candidate["model_load_millis"],
                "inference_millis": candidate["inference_wall_millis"],
                "total_millis": candidate["total_wall_millis"],
                "real_time_factor": candidate["inference_wall_millis"] / 1000 / audio_seconds,
                "word_error_percent": comparison["word_error_percent"],
                "word_edits": comparison["word_edits"],
                "token_error_percent": comparison["token_error_percent"],
                "token_edits": comparison["token_edits"],
                "token_count": len(candidate["tokens"]),
                "pss_kib": candidate["memory_after_inference"]["total_pss_kib"],
                "warnings": candidate.get("warnings", []),
            }
            rows.append(row)
            write_json(output / "raw" / f"{configuration}.json", {"row": row, "report": report, "comparison": comparison})
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        summary = {
            "schema_version": 1,
            "matrix_id": matrix_id,
            "status": "completed",
            "device_serial": serial,
            "model_id": "openai/whisper-large-v3-turbo",
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "audio": {
                "path": str(args.audio),
                "sha256": suite.sha256(args.audio),
                "duration_seconds": baseline["audio_samples_16khz"] / 16_000,
            },
            "baseline": {
                "path": str(args.baseline),
                "sha256": suite.sha256(args.baseline),
                "language": args.forced_language,
            },
            "encoder": {"sha256": ENCODER_SHA256, "bytes": ENCODER.stat().st_size, "stored_weights": "FLOAT32"},
            "decoder": {"sha256": DECODER_SHA256, "bytes": DECODER.stat().st_size, "stored_weights": "FLOAT32"},
            "timestamp_tokens_enabled": False,
            "source_quantization_enabled": False,
            "configurations": rows,
        }
        write_json(output / "summary.json", summary)
        lines = [
            f"# Original OpenAI FP32 split precision matrix {matrix_id}",
            "",
            "Both model halves were exported from the same pinned OpenAI checkpoint with FLOAT32 stored weights.",
            "",
            "| Encoder compute | Decoder compute | Load | Inference | Total | RTF | WER | PSS | Thermal |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                f"| {row['encoder_precision']} | {row['decoder_precision']} | "
                f"{row['model_load_millis'] / 1000:.3f} s | {row['inference_millis'] / 1000:.3f} s | "
                f"{row['total_millis'] / 1000:.3f} s | {row['real_time_factor']:.3f} | "
                f"{row['word_error_percent']:.3f}% | {row['pss_kib'] / 1024 / 1024:.2f} GiB | "
                f"{row['thermal_before']['thermal_status']}→{row['thermal_after']['thermal_status']} |"
            )
        (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(output / "summary.md", flush=True)
    finally:
        for relative in reversed(staged):
            suite.remove_private_file(args.adb, serial, relative)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MatrixError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Original FP32 split matrix failed: {error}", file=sys.stderr)
        raise SystemExit(2)
