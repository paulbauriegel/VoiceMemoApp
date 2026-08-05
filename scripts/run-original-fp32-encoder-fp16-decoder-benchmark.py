#!/usr/bin/env python3
"""Run one controlled original-FP32-encoder/native-FP16-decoder benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = PROJECT_ROOT / "scripts/run-whisper-benchmark-suite.py"
ENCODER = PROJECT_ROOT / "build/experiments/original-fp32-encoder/whisper-large-v3-turbo-original-fp32-encoder.tflite"
DECODER = PROJECT_ROOT / "build/experiments/whisper-fast-split/decoder.tflite"
AUDIO = PROJECT_ROOT / "benchmark-inputs/whisper-suite/reference.wav"
BASELINE = PROJECT_ROOT / "benchmark-inputs/whisper-suite/baseline.json"
ENCODER_SHA256 = "01be8e811608cc9eb3780408f8e020509d216a1d5e280320232b2c0a13d5e353"
DECODER_SHA256 = "d0e7fb4b4230fe4f41533fa649b305298b5df17a414b6af12d4eba5aa7bd2f06"
PACKAGE = "com.example.voicememo"
TEST_PACKAGE = f"{PACKAGE}.test"
TEST_RUNNER = "androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "com.example.voicememo.transcription.GpuWholeMemoWerBenchmarkTest#compareWholeMemoBackends"
PREVIOUS = {
    "encoder": "INT8-stored weights, FP32 GPU compute",
    "decoder": "native FP16 GPU compute",
    "inference_millis": 151_193,
    "total_millis": 167_094,
    "word_error_percent": 1.4492753623188406,
    "pss_kib": 3_558_688,
}


class BenchmarkError(RuntimeError):
    pass


def load_suite():
    spec = importlib.util.spec_from_file_location("whisper_benchmark_suite", SUITE_PATH)
    if spec is None or spec.loader is None:
        raise BenchmarkError("Could not load benchmark-suite helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_candidate(suite, adb_path: str, serial: str, run_id: str) -> dict:
    arguments = {
        "class": TEST_CLASS,
        "runId": run_id,
        "candidateOnly": "true",
        "candidateBackend": "gpu",
        "externalEncoderModelName": "original-fp32-encoder.tflite",
        "externalEncoderModelSha256": ENCODER_SHA256,
        "externalDecoderModelName": "split-fast-decoder.tflite",
        "externalDecoderModelSha256": DECODER_SHA256,
        "externalAudioName": AUDIO.name,
        "forcedLanguage": "de",
        "timestampTokensEnabled": "false",
        "encoderGpuPrecision": "fp32",
        "decoderGpuPrecision": "fp16-native",
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
        raise BenchmarkError("Instrumentation benchmark failed")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="2B211FDH300F4S")
    parser.add_argument("--adb", default=str(Path.home() / "Library/Android/sdk/platform-tools/adb"))
    parser.add_argument("--thermal-timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "build/reports/original-fp32-encoder-fp16-decoder",
    )
    args = parser.parse_args()
    suite = load_suite()
    for path, digest in ((ENCODER, ENCODER_SHA256), (DECODER, DECODER_SHA256)):
        if not path.is_file() or suite.sha256(path) != digest:
            raise BenchmarkError(f"Missing or changed model: {path}")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    serial = suite.one_device(args.adb, args.serial)
    suite.assert_device_idle(args.adb, serial)
    run_id = datetime.now(timezone.utc).strftime("original-fp32-encoder-fp16-decoder-%Y%m%dT%H%M%SZ")
    output = args.output.resolve() / run_id
    if output.exists():
        raise BenchmarkError(f"Refusing to overwrite {output}")
    staged: list[str] = []
    files = (
        (AUDIO, f"files/gpu-audio/{AUDIO.name}", suite.sha256(AUDIO)),
        (ENCODER, "files/gpu-models/original-fp32-encoder.tflite", ENCODER_SHA256),
        (DECODER, "files/gpu-models/split-fast-decoder.tflite", DECODER_SHA256),
    )
    try:
        for source, relative, digest in files:
            suite.stage_private_file(args.adb, serial, source, relative, digest)
            staged.append(relative)
        thermal_before = suite.wait_for_nominal_thermal(
            args.adb, serial, args.thermal_timeout_seconds
        )
        suite.assert_device_idle(args.adb, serial)
        report = run_candidate(suite, args.adb, serial, run_id)
        thermal_after = suite.thermal_snapshot(args.adb, serial)
        candidate = report.get("candidate")
        if report.get("status") != "completed" or not isinstance(candidate, dict):
            raise BenchmarkError(f"Benchmark did not complete: {report.get('error')}")
        expected_mode = "split-gpu-encoder-fp32-decoder-fp16-no-source-quant"
        checks = {
            "encoder SHA": report.get("encoder_model_sha256") == ENCODER_SHA256,
            "decoder SHA": report.get("decoder_model_sha256") == DECODER_SHA256,
            "mode": candidate.get("mode") == expected_mode,
            "timestamps disabled": report.get("timestamp_tokens_enabled") is False,
            "no timestamp tokens": candidate.get("timestamp_token_count") == 0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise BenchmarkError(f"Invalid benchmark provenance: {failed}")
        compared = suite.comparison(baseline, candidate)
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "device_serial": serial,
            "thermal_before": thermal_before,
            "thermal_after": thermal_after,
            "encoder": {
                "kind": "original OpenAI checkpoint FP32 weights; FP32 GPU compute",
                "sha256": ENCODER_SHA256,
                "bytes": ENCODER.stat().st_size,
            },
            "decoder": {
                "kind": "existing split FP32 weights; native FP16 GPU compute",
                "sha256": DECODER_SHA256,
                "bytes": DECODER.stat().st_size,
            },
            "timestamp_tokens_enabled": False,
            "candidate": {
                "model_load_millis": candidate["model_load_millis"],
                "inference_millis": candidate["inference_wall_millis"],
                "total_millis": candidate["total_wall_millis"],
                "real_time_factor": candidate["inference_wall_millis"] / 1000 / (baseline["audio_samples_16khz"] / 16_000),
                "word_error_percent": compared["word_error_percent"],
                "word_edits": compared["word_edits"],
                "token_error_percent": compared["token_error_percent"],
                "token_edits": compared["token_edits"],
                "token_count": len(candidate["tokens"]),
                "pss_kib": candidate["memory_after_inference"]["total_pss_kib"],
                "warnings": candidate.get("warnings", []),
            },
            "previous_int8_stored_encoder_fp32_compute": PREVIOUS,
        }
        new = summary["candidate"]
        summary["delta_vs_previous"] = {
            "inference_percent": (new["inference_millis"] / PREVIOUS["inference_millis"] - 1) * 100,
            "total_percent": (new["total_millis"] / PREVIOUS["total_millis"] - 1) * 100,
            "word_error_percentage_points": new["word_error_percent"] - PREVIOUS["word_error_percent"],
            "pss_gib": (new["pss_kib"] - PREVIOUS["pss_kib"]) / 1024 / 1024,
        }
        write_json(output / "raw.json", {"report": report, "comparison": compared})
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        for relative in reversed(staged):
            suite.remove_private_file(args.adb, serial, relative)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Original FP32 encoder benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(2)
