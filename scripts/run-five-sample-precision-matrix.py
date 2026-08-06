#!/usr/bin/env python3
"""Run five audio samples across four LiteRT precisions and whisper.cpp FP16."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = PROJECT_ROOT / "scripts/run-whisper-benchmark-suite.py"
ENCODER = PROJECT_ROOT / "build/experiments/original-fp32-encoder/whisper-large-v3-turbo-original-fp32-encoder.tflite"
DECODER = PROJECT_ROOT / "build/experiments/original-fp32-decoder/whisper-large-v3-turbo-original-fp32-decoder.tflite"
WHISPER_CPP = PROJECT_ROOT / "benchmark-inputs/whisper-suite/ggml-large-v3-turbo.bin"
ENCODER_SHA256 = "01be8e811608cc9eb3780408f8e020509d216a1d5e280320232b2c0a13d5e353"
DECODER_SHA256 = "d661702d7753165cfaf01bf5b060f46065df500959e42e5254fd2598a54dedb7"
PROJECTION: Path | None = None
PROJECTION_SHA256: str | None = None
CROSS_CACHE: Path | None = None
CROSS_CACHE_SHA256: str | None = None
WHISPER_CPP_SHA256 = "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"
CHECKPOINT_SHA256 = "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
PACKAGE = "com.example.voicememo"
TEST_PACKAGE = f"{PACKAGE}.test"
TEST_RUNNER = "androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "com.example.voicememo.transcription.GpuWholeMemoWerBenchmarkTest#compareWholeMemoBackends"


@dataclass(frozen=True)
class Sample:
    id: str
    audio: Path
    baseline: Path
    language: str


@dataclass(frozen=True)
class Configuration:
    id: str
    label: str
    runtime: str
    encoder_precision: str | None = None
    decoder_precision: str | None = None


SAMPLES = (
    Sample("controlled-german-170s", PROJECT_ROOT / "benchmark-inputs/whisper-suite/reference.wav", PROJECT_ROOT / "benchmark-inputs/whisper-suite/baseline.json", "de"),
    Sample("sample-test-153s", PROJECT_ROOT / "benchmark-inputs/sample-test/audio.wav", PROJECT_ROOT / "benchmark-inputs/sample-test/baseline.json", "de"),
    Sample("english-dialogue-short-9s", PROJECT_ROOT / "benchmark-inputs/article-samples/english-dialogue-short/audio.wav", PROJECT_ROOT / "benchmark-inputs/article-samples/english-dialogue-short/baseline.json", "en"),
    Sample("english-dialogue-proper-names-47s", PROJECT_ROOT / "benchmark-inputs/article-samples/english-dialogue-proper-names/audio.wav", PROJECT_ROOT / "benchmark-inputs/article-samples/english-dialogue-proper-names/baseline.json", "en"),
    Sample("english-dialogue-mixed-fragments-51s", PROJECT_ROOT / "benchmark-inputs/article-samples/english-dialogue-mixed-fragments/audio.wav", PROJECT_ROOT / "benchmark-inputs/article-samples/english-dialogue-mixed-fragments/baseline.json", "en"),
)

CONFIGURATIONS = (
    Configuration("litert-fp32-fp32", "LiteRT FP32 encoder + FP32 decoder", "litert", "fp32", "fp32"),
    Configuration("litert-fp32-fp16", "LiteRT FP32 encoder + native-FP16 decoder", "litert", "fp32", "fp16-native"),
    Configuration("litert-fp16-fp32", "LiteRT native-FP16 encoder + FP32 decoder", "litert", "fp16-native", "fp32"),
    Configuration("litert-fp16-fp16", "LiteRT native-FP16 encoder + native-FP16 decoder", "litert", "fp16-native", "fp16-native"),
    Configuration("whispercpp-fp16-vulkan", "whisper.cpp FP16 Vulkan", "whispercpp"),
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


def instrumentation_arguments(
    run_id: str, sample: Sample, configuration: Configuration, audio_name: str
) -> dict[str, str]:
    common = {
        "class": TEST_CLASS,
        "runId": run_id,
        "candidateOnly": "true",
        "externalAudioName": audio_name,
        "forcedLanguage": sample.language,
        "timestampTokensEnabled": "false",
    }
    if configuration.runtime == "litert":
        arguments = common | {
            "candidateBackend": "gpu",
            "externalEncoderModelName": ENCODER.name,
            "externalEncoderModelSha256": ENCODER_SHA256,
            "externalDecoderModelName": DECODER.name,
            "externalDecoderModelSha256": DECODER_SHA256,
            "encoderGpuPrecision": str(configuration.encoder_precision),
            "decoderGpuPrecision": str(configuration.decoder_precision),
        }
        if PROJECTION is not None:
            arguments |= {
                "externalProjectionModelName": PROJECTION.name,
                "externalProjectionModelSha256": str(PROJECTION_SHA256),
            }
        if CROSS_CACHE is not None:
            arguments |= {
                "externalCrossCacheModelName": CROSS_CACHE.name,
                "externalCrossCacheModelSha256": str(CROSS_CACHE_SHA256),
            }
        return arguments
    return common | {
        "candidateBackend": "whispercpp",
        "externalModelName": WHISPER_CPP.name,
        "externalModelSha256": WHISPER_CPP_SHA256,
        "whisperCppThreads": "4",
        "whisperCppBackend": "vulkan",
    }


def run_candidate(suite, adb_path: str, serial: str, arguments: dict[str, str]) -> dict:
    command = suite.adb(adb_path, serial, "shell", "am", "instrument", "-w", "-r")
    for key, value in arguments.items():
        command.extend(("-e", key, value))
    command.append(f"{TEST_PACKAGE}/{TEST_RUNNER}")
    completed = suite.run(command, capture=True, check=False)
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode != 0 or "OK (1 test)" not in completed.stdout:
        raise MatrixError(f"Instrumentation failed for {arguments['runId']}")
    report_text = suite.run(
        suite.adb(
            adb_path,
            serial,
            "exec-out",
            "run-as",
            PACKAGE,
            "cat",
            f"cache/gpu-wer/whole-{arguments['runId']}.json",
        ),
        capture=True,
    ).stdout
    return json.loads(report_text)


def existing_private_hash(suite, adb_path: str, serial: str, relative: str) -> str | None:
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
    if completed.returncode != 0:
        return None
    fields = completed.stdout.split()
    return fields[0] if fields else None


def thermal_snapshot(suite, adb_path: str, serial: str) -> dict:
    snapshot = suite.thermal_snapshot(adb_path, serial)
    thermal = suite.run(
        suite.adb(adb_path, serial, "shell", "dumpsys", "thermalservice"),
        capture=True,
    ).stdout
    skin_values = re.findall(
        r"Temperature\{mValue=([-+0-9.eE]+),\s*mType=3,\s*"
        r"mName=VIRTUAL-SKIN,",
        thermal,
    )
    snapshot["skin_celsius"] = float(skin_values[-1]) if skin_values else None
    return snapshot


def wait_for_ready(suite, adb_path: str, serial: str, timeout_seconds: int, maximum_skin_celsius: float | None) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while True:
        suite.assert_device_idle(adb_path, serial)
        snapshot = thermal_snapshot(suite, adb_path, serial)
        skin = snapshot["skin_celsius"]
        skin_ready = maximum_skin_celsius is None or (
            skin is not None and skin <= maximum_skin_celsius
        )
        if snapshot["thermal_status"] == 0 and skin_ready:
            return snapshot
        if time.monotonic() >= deadline:
            raise MatrixError(
                "Device did not reach thermal status 0 and the requested "
                f"skin limit before timeout: {snapshot}"
            )
        print(f"Waiting for benchmark thermal gate: {snapshot}", flush=True)
        time.sleep(15)


def validate(report: dict, configuration: Configuration) -> None:
    candidate = report.get("candidate")
    checks = {
        "status": report.get("status") == "completed",
        "candidate": isinstance(candidate, dict),
        "timestamp mode": report.get("timestamp_tokens_enabled") is False,
        "timestamp count": isinstance(candidate, dict) and candidate.get("timestamp_token_count") == 0,
    }
    if configuration.runtime == "litert":
        def mode_precision(value: str) -> str:
            if value == "fp16-fp32-accum":
                return "fp16"
            return value.replace("-native", "")
        expected_mode = (
            f"split-gpu-encoder-{mode_precision(configuration.encoder_precision)}"
            f"-decoder-{mode_precision(configuration.decoder_precision)}-no-source-quant"
        )
        checks |= {
            "backend": report.get("candidate_backend") == "gpu",
            "encoder SHA": report.get("encoder_model_sha256") == ENCODER_SHA256,
            "decoder SHA": report.get("decoder_model_sha256") == DECODER_SHA256,
            "encoder precision": report.get("encoder_gpu_precision") == configuration.encoder_precision,
            "decoder precision": report.get("decoder_gpu_precision") == configuration.decoder_precision,
            "mode": isinstance(candidate, dict) and candidate.get("mode") == expected_mode,
        }
        if PROJECTION is not None:
            checks["projection SHA"] = (
                report.get("projection_model_sha256") == PROJECTION_SHA256
            )
        if CROSS_CACHE is not None:
            checks["cross-cache SHA"] = (
                report.get("cross_cache_model_sha256") == CROSS_CACHE_SHA256
            )
    else:
        checks |= {
            "backend": report.get("candidate_backend") == "whispercpp",
            "model SHA": report.get("model_sha256") == WHISPER_CPP_SHA256,
            "mode": isinstance(candidate, dict)
            and candidate.get("mode") == "whisper.cpp-large-v3-turbo-fp16-vulkan",
            "threads": report.get("whisper_cpp_threads") == 4,
            "Vulkan": report.get("whisper_cpp_backend") == "vulkan",
        }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise MatrixError(f"Invalid {configuration.id} report: {failed}")


def make_row(suite, sample: Sample, configuration: Configuration, baseline: dict, report: dict, thermal_before: dict, thermal_after: dict) -> dict:
    candidate = report["candidate"]
    comparison = suite.comparison(baseline, candidate)
    audio_seconds = baseline["audio_samples_16khz"] / 16_000
    return {
        "sample_id": sample.id,
        "configuration": configuration.id,
        "label": configuration.label,
        "runtime": configuration.runtime,
        "encoder_precision": configuration.encoder_precision,
        "decoder_precision": configuration.decoder_precision,
        "language": sample.language,
        "audio_seconds": audio_seconds,
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
        "model_load_millis": candidate["model_load_millis"],
        "inference_millis": candidate["inference_wall_millis"],
        "total_millis": candidate["total_wall_millis"],
        "real_time_factor": candidate["inference_wall_millis"] / 1000 / audio_seconds,
        "word_error_percent": comparison["word_error_percent"],
        "word_edits": comparison["word_edits"],
        "reference_words": comparison["reference_words"],
        "token_error_percent": comparison["token_error_percent"],
        "token_edits": comparison["token_edits"],
        "reference_tokens": comparison["reference_tokens"],
        "candidate_tokens": comparison["candidate_tokens"],
        "pss_kib": candidate["memory_after_inference"]["total_pss_kib"],
        "warnings": candidate.get("warnings", []),
    }, comparison


def aggregate(rows: list[dict]) -> list[dict]:
    result = []
    for configuration in CONFIGURATIONS:
        selected = [row for row in rows if row["configuration"] == configuration.id]
        if len(selected) != len(SAMPLES):
            continue
        word_edits = sum(row["word_edits"] for row in selected)
        reference_words = sum(row["reference_words"] for row in selected)
        token_edits = sum(row["token_edits"] for row in selected)
        reference_tokens = sum(row["reference_tokens"] for row in selected)
        audio_seconds = sum(row["audio_seconds"] for row in selected)
        model_load_millis = sum(row["model_load_millis"] for row in selected)
        inference_millis = sum(row["inference_millis"] for row in selected)
        total_millis = sum(row["total_millis"] for row in selected)
        result.append({
            "configuration": configuration.id,
            "label": configuration.label,
            "sample_count": len(selected),
            "audio_seconds": audio_seconds,
            "model_load_millis": model_load_millis,
            "inference_millis": inference_millis,
            "total_millis": total_millis,
            "aggregate_rtf": inference_millis / 1000 / audio_seconds,
            "micro_word_error_percent": word_edits / reference_words * 100,
            "word_edits": word_edits,
            "reference_words": reference_words,
            "mean_sample_wer_percent": statistics.fmean(row["word_error_percent"] for row in selected),
            "micro_token_error_percent": token_edits / reference_tokens * 100,
            "token_edits": token_edits,
            "reference_tokens": reference_tokens,
            "maximum_pss_kib": max(row["pss_kib"] for row in selected),
            "warning_count": sum(len(row["warnings"]) for row in selected),
        })
    return result


def render_markdown(matrix_id: str, samples: list[dict], rows: list[dict], aggregates: list[dict]) -> str:
    lines = [
        f"# Five-sample Whisper precision matrix {matrix_id}",
        "",
        "All WER values are disagreements with pinned OpenAI PyTorch FLOAT32 transcripts, not human-ground-truth error rates.",
        "Every device run starts after Android reports thermal status 0. Timestamp tokens and LiteRT source quantization are disabled.",
        "",
        "## Aggregate",
        "",
        "| Configuration | Model loading | Inference | End-to-end | Aggregate RTF | Micro WER | Mean sample WER | Micro token error | Max PSS | Warnings |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['label']} | {row['model_load_millis'] / 1000:.3f} s | "
            f"{row['inference_millis'] / 1000:.3f} s | {row['total_millis'] / 1000:.3f} s | {row['aggregate_rtf']:.3f} | "
            f"{row['micro_word_error_percent']:.3f}% ({row['word_edits']}/{row['reference_words']}) | "
            f"{row['mean_sample_wer_percent']:.3f}% | {row['micro_token_error_percent']:.3f}% | "
            f"{row['maximum_pss_kib'] / 1024 / 1024:.2f} GiB | {row['warning_count']} |"
        )
    for sample in samples:
        lines += [
            "",
            f"## {sample['id']}",
            "",
            f"Duration {sample['duration_seconds']:.3f} s; language `{sample['language']}`; audio SHA-256 `{sample['audio_sha256']}`.",
            "",
            "| Configuration | Model loading | Inference | End-to-end | RTF | WER | Token error | PSS | Thermal | Warnings |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
        for row in [value for value in rows if value["sample_id"] == sample["id"]]:
            lines.append(
                f"| {row['label']} | {row['model_load_millis'] / 1000:.3f} s | "
                f"{row['inference_millis'] / 1000:.3f} s | {row['total_millis'] / 1000:.3f} s | "
                f"{row['real_time_factor']:.3f} | {row['word_error_percent']:.3f}% | {row['token_error_percent']:.3f}% | "
                f"{row['pss_kib'] / 1024 / 1024:.2f} GiB | {row['thermal_before']['thermal_status']}→{row['thermal_after']['thermal_status']} | "
                f"{len(row['warnings'])} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="2B211FDH300F4S")
    parser.add_argument("--adb", default=str(Path.home() / "Library/Android/sdk/platform-tools/adb"))
    parser.add_argument("--cooldown-seconds", type=int, default=30)
    parser.add_argument("--thermal-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--maximum-skin-celsius",
        type=float,
        help="optional VIRTUAL-SKIN ceiling required before every run",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "build/reports/five-sample-precision-matrix")
    parser.add_argument("--matrix-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    suite = load_suite()
    expected_models = [(ENCODER, ENCODER_SHA256), (DECODER, DECODER_SHA256), (WHISPER_CPP, WHISPER_CPP_SHA256)]
    if PROJECTION is not None:
        expected_models.append((PROJECTION, str(PROJECTION_SHA256)))
    if CROSS_CACHE is not None:
        expected_models.append((CROSS_CACHE, str(CROSS_CACHE_SHA256)))
    for path, digest in expected_models:
        if not path.is_file() or suite.sha256(path) != digest:
            raise MatrixError(f"Missing or changed model: {path}")
    sample_data = []
    baselines = {}
    for sample in SAMPLES:
        if not sample.audio.is_file() or not sample.baseline.is_file():
            raise MatrixError(f"Missing audio or baseline for {sample.id}")
        baseline = json.loads(sample.baseline.read_text(encoding="utf-8"))
        if baseline.get("model_revision") != MODEL_REVISION or baseline.get("language") != sample.language:
            raise MatrixError(f"Baseline provenance mismatch for {sample.id}")
        baselines[sample.id] = baseline
        sample_data.append({
            "id": sample.id,
            "language": sample.language,
            "duration_seconds": baseline["audio_samples_16khz"] / 16_000,
            "audio_path": str(sample.audio),
            "audio_sha256": suite.sha256(sample.audio),
            "baseline_path": str(sample.baseline),
            "baseline_sha256": suite.sha256(sample.baseline),
        })
    serial = suite.one_device(args.adb, args.serial)
    suite.assert_device_idle(args.adb, serial)
    matrix_id = args.matrix_id or datetime.now(timezone.utc).strftime("five-sample-%Y%m%dT%H%M%SZ")
    output = args.output.resolve() / matrix_id
    if output.exists() and not args.resume:
        raise MatrixError(f"Refusing to overwrite {output}; pass --resume to continue")
    output.mkdir(parents=True, exist_ok=True)
    audio_names = {sample.id: f"five-{sample.id}.wav" for sample in SAMPLES}
    staged_files = [
        (ENCODER, f"files/gpu-models/{ENCODER.name}", ENCODER_SHA256),
        (DECODER, f"files/gpu-models/{DECODER.name}", DECODER_SHA256),
        (WHISPER_CPP, f"files/whispercpp-models/{WHISPER_CPP.name}", WHISPER_CPP_SHA256),
    ]
    if PROJECTION is not None:
        staged_files.append(
            (PROJECTION, f"files/gpu-models/{PROJECTION.name}", str(PROJECTION_SHA256))
        )
    if CROSS_CACHE is not None:
        staged_files.append(
            (CROSS_CACHE, f"files/gpu-models/{CROSS_CACHE.name}", str(CROSS_CACHE_SHA256))
        )
    staged_files += [
        (sample.audio, f"files/gpu-audio/{audio_names[sample.id]}", suite.sha256(sample.audio))
        for sample in SAMPLES
    ]
    staged: list[str] = []
    rows: list[dict] = []
    try:
        for source, relative, digest in staged_files:
            existing_hash = existing_private_hash(
                suite, args.adb, serial, relative
            )
            if existing_hash is not None:
                if existing_hash != digest:
                    raise MatrixError(
                        f"Refusing to replace existing app-private diagnostic "
                        f"file with a different hash: {relative}"
                    )
                print(f"Reusing existing staged file {relative}", flush=True)
                continue
            suite.stage_private_file(args.adb, serial, source, relative, digest)
            staged.append(relative)
        run_index = 0
        for sample in SAMPLES:
            for configuration in CONFIGURATIONS:
                raw_path = output / "raw" / sample.id / f"{configuration.id}.json"
                if args.resume and raw_path.is_file():
                    saved = json.loads(raw_path.read_text(encoding="utf-8"))
                    rows.append(saved["row"])
                    print(f"Reusing {sample.id}/{configuration.id}", flush=True)
                    continue
                if run_index and args.cooldown_seconds:
                    time.sleep(args.cooldown_seconds)
                thermal_before = wait_for_ready(
                    suite,
                    args.adb,
                    serial,
                    args.thermal_timeout_seconds,
                    args.maximum_skin_celsius,
                )
                run_id = f"{matrix_id}-{sample.id}-{configuration.id}"
                report = run_candidate(
                    suite,
                    args.adb,
                    serial,
                    instrumentation_arguments(run_id, sample, configuration, audio_names[sample.id]),
                )
                validate(report, configuration)
                thermal_after = thermal_snapshot(suite, args.adb, serial)
                row, comparison = make_row(suite, sample, configuration, baselines[sample.id], report, thermal_before, thermal_after)
                rows.append(row)
                write_json(raw_path, {"row": row, "report": report, "comparison": comparison})
                print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
                run_index += 1
        aggregates = aggregate(rows)
        summary = {
            "schema_version": 1,
            "matrix_id": matrix_id,
            "status": "completed",
            "device_serial": serial,
            "model_id": "openai/whisper-large-v3-turbo",
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "timestamp_tokens_enabled": False,
            "source_quantization_enabled": False,
            "samples": sample_data,
            "configurations": [configuration.__dict__ for configuration in CONFIGURATIONS],
            "rows": rows,
            "aggregate": aggregates,
        }
        write_json(output / "summary.json", summary)
        (output / "summary.md").write_text(render_markdown(matrix_id, sample_data, rows, aggregates), encoding="utf-8")
        print(output / "summary.md", flush=True)
    finally:
        for relative in reversed(staged):
            suite.remove_private_file(args.adb, serial, relative)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MatrixError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Five-sample precision matrix failed: {error}", file=sys.stderr)
        raise SystemExit(2)
