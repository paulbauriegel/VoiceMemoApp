#!/usr/bin/env python3
"""Run a controlled, evidence-preserving Android Whisper benchmark suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence
import unicodedata
import wave


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSISTENT_INPUT_ROOT = PROJECT_ROOT / "benchmark-inputs/whisper-suite"
DEFAULT_AUDIO = PERSISTENT_INPUT_ROOT / "reference.wav"
DEFAULT_BASELINE = PERSISTENT_INPUT_ROOT / "baseline.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "build/reports/whisper-benchmark-suite"
TOKENIZER = PROJECT_ROOT / "app/src/main/generatedAssets/whisper/tokenizer.json"
GENERATION_CONFIG = (
    PROJECT_ROOT / "app/src/main/generatedAssets/whisper/generation_config.json"
)
TOKENIZER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
WHISPER_CPP_REVISION = "f049fff95a089aa9969deb009cdd4892b3e74916"
PACKAGE = "com.example.voicememo"
TEST_PACKAGE = f"{PACKAGE}.test"
TEST_RUNNER = "androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = (
    "com.example.voicememo.transcription."
    "GpuWholeMemoWerBenchmarkTest#compareWholeMemoBackends"
)
REFERENCE_LANGUAGE = "de"
AUTO_DETECTED_LANGUAGE = "en"
LANGUAGE_TOKEN_ID = 50261
TIMESTAMP_BEGIN_TOKEN_ID = 50365
SAMPLE_RATE = 16_000
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,127}\Z")
WORD_PATTERN = re.compile(r"[\w]+(?:['’\-][\w]+)*", re.UNICODE)


class SuiteError(RuntimeError):
    """A fail-closed benchmark-suite error."""


@dataclass(frozen=True)
class Configuration:
    id: str
    label: str
    runtime: str
    model_path: str
    model_bytes: int
    model_sha256: str
    expected_mode: str
    instrumentation: dict[str, str]

    @property
    def resolved_model(self) -> Path:
        return PROJECT_ROOT / self.model_path

    @property
    def model_directory(self) -> str:
        return (
            "files/whispercpp-models"
            if self.runtime == "whispercpp"
            else "files/gpu-models"
        )


CONFIGURATIONS = (
    Configuration(
        id="whispercpp-fp16-vulkan",
        label="whisper.cpp FP16 Vulkan",
        runtime="whispercpp",
        model_path="benchmark-inputs/whisper-suite/ggml-large-v3-turbo.bin",
        model_bytes=1_624_555_275,
        model_sha256=(
            "1fc70f774d38eb169993ac391eea357ef"
            "47c88757ef72ee5943879b7e8e2bc69"
        ),
        expected_mode="whisper.cpp-large-v3-turbo-fp16-vulkan",
        instrumentation={
            "candidateBackend": "whispercpp",
            "whisperCppThreads": "4",
            "whisperCppBackend": "vulkan",
        },
    ),
    Configuration(
        id="whispercpp-q8_0-vulkan",
        label="whisper.cpp Q8_0 Vulkan",
        runtime="whispercpp",
        model_path=(
            "benchmark-inputs/whisper-suite/"
            "ggml-large-v3-turbo-q8_0.bin"
        ),
        model_bytes=874_188_075,
        model_sha256=(
            "317eb69c11673c9de1e1f0d459b25399"
            "9804ec71ac4c23c17ecf5fbe24e259a1"
        ),
        expected_mode="whisper.cpp-large-v3-turbo-q8_0-vulkan",
        instrumentation={
            "candidateBackend": "whispercpp",
            "whisperCppThreads": "4",
            "whisperCppBackend": "vulkan",
        },
    ),
    Configuration(
        id="litert-int8-cpu-2x3",
        label="LiteRT INT8 CPU 2 workers x 3 threads",
        runtime="litert",
        model_path=(
            "benchmark-inputs/whisper-suite/"
            "whisper-large-v3-turbo-int8.tflite"
        ),
        model_bytes=1_094_571_944,
        model_sha256=(
            "8e69e43d7773b856a93cc192f6308adc"
            "1d1e6c921d48c91c684bc78bf6e91418"
        ),
        expected_mode="cpu-2x3",
        instrumentation={
            "candidateBackend": "cpu",
            "cpuWorkers": "2",
            "cpuThreads": "3",
            "gpuPrecision": "fp32",
        },
    ),
    Configuration(
        id="litert-int8-cpu-4x3",
        label="LiteRT INT8 CPU 4 workers x 3 threads",
        runtime="litert",
        model_path=(
            "benchmark-inputs/whisper-suite/"
            "whisper-large-v3-turbo-int8.tflite"
        ),
        model_bytes=1_094_571_944,
        model_sha256=(
            "8e69e43d7773b856a93cc192f6308adc"
            "1d1e6c921d48c91c684bc78bf6e91418"
        ),
        expected_mode="cpu-4x3",
        instrumentation={
            "candidateBackend": "cpu",
            "cpuWorkers": "4",
            "cpuThreads": "3",
            "gpuPrecision": "fp32",
        },
    ),
    Configuration(
        id="litert-fp16-opencl-f32_f16",
        label="LiteRT FP16 storage OpenCL with FP32 accumulation",
        runtime="litert",
        model_path=(
            "benchmark-inputs/whisper-suite/"
            "whisper-large-v3-turbo-fp16-gpu-fixed.tflite"
        ),
        model_bytes=1_617_806_560,
        model_sha256=(
            "fd50a74a3d0f47fcd8e11c3bdd167fe"
            "051b3f0f160c77c162b802881914710f3"
        ),
        expected_mode="gpu-fp16-storage-fp32-accum-no-source-quant",
        instrumentation={
            "candidateBackend": "gpu",
            "gpuPrecision": "fp16-fp32-accum",
        },
    ),
    Configuration(
        id="litert-int8-encoder-fp32-decoder-opencl",
        label="LiteRT quantized encoder plus FP32 decoder OpenCL",
        runtime="litert",
        model_path=(
            "benchmark-inputs/whisper-suite/"
            "whisper-large-v3-turbo-full-float-decoder.tflite"
        ),
        model_bytes=1_779_622_096,
        model_sha256=(
            "a2613d64e6cb73ff40db7c548215fd08"
            "6f096aa088c6cc113c833ccc87501e89"
        ),
        expected_mode="gpu-encoder-gpu-decoder-no-source-quant",
        instrumentation={
            "candidateBackend": "gpu",
            "gpuPrecision": "fp32",
        },
    ),
)


def log(message: str) -> None:
    print(message, flush=True)


def run(
    command: Sequence[str],
    *,
    capture: bool = False,
    check: bool = True,
    stdin: Any = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log("+ " + shlex.join(str(part) for part in command))
    return subprocess.run(
        [str(part) for part in command],
        cwd=PROJECT_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        stdin=stdin,
        env=env,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def adb(adb_path: str, serial: str, *arguments: str) -> list[str]:
    return [adb_path, "-s", serial, *arguments]


def instrumentation_run_id(
    suite_id: str,
    configuration_id: str,
    run_number: int,
) -> str:
    raw = f"{suite_id}-{configuration_id}-r{run_number}"
    return re.sub(r"[^A-Za-z0-9-]", "-", raw)


def one_device(adb_path: str, requested: str | None) -> str:
    output = run([adb_path, "devices"], capture=True).stdout
    devices = [
        fields[0]
        for line in output.splitlines()[1:]
        if len(fields := line.split()) >= 2 and fields[1] == "device"
    ]
    if requested:
        if requested not in devices:
            raise SuiteError(f"ADB device is unavailable: {requested}")
        return requested
    if len(devices) != 1:
        raise SuiteError(
            f"Expected exactly one authorized ADB device, found {devices}."
        )
    return devices[0]


def capture_private_file(
    adb_path: str,
    serial: str,
    relative: str,
    destination: Path,
) -> bool:
    exists = run(
        adb(adb_path, serial, "shell", "run-as", PACKAGE, "test", "-f", relative),
        capture=True,
        check=False,
    )
    if exists.returncode != 0:
        return False
    command = adb(
        adb_path,
        serial,
        "exec-out",
        "run-as",
        PACKAGE,
        "cat",
        relative,
    )
    log("+ " + shlex.join(command) + f" > {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        completed = subprocess.run(command, stdout=output, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise SuiteError(f"Could not capture app-private file {relative}.")
    return True


def assert_device_idle(adb_path: str, serial: str) -> None:
    run(
        adb(adb_path, serial, "exec-out", "run-as", PACKAGE, "id"),
        capture=True,
    )
    services = run(
        adb(
            adb_path,
            serial,
            "shell",
            "dumpsys",
            "activity",
            "services",
            PACKAGE,
        ),
        capture=True,
    ).stdout
    if "RecordingService" in services or "SystemForegroundService" in services:
        raise SuiteError("VoiceMemo is recording or transcribing.")
    with tempfile.TemporaryDirectory(prefix="voicememo-suite-idle-") as directory:
        snapshot = Path(directory)
        database = snapshot / "voice-memo.db"
        if not capture_private_file(
            adb_path,
            serial,
            "databases/voice-memo.db",
            database,
        ):
            raise SuiteError("The installed VoiceMemo database is missing.")
        for suffix in ("-wal", "-shm"):
            capture_private_file(
                adb_path,
                serial,
                f"databases/voice-memo.db{suffix}",
                snapshot / f"voice-memo.db{suffix}",
            )
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            active = connection.execute(
                """
                SELECT COUNT(*) FROM transcription_jobs
                WHERE status IN (
                    'QUEUED',
                    'RUNNING',
                    'PAUSED_FOR_RECORDING',
                    'CANCEL_REQUESTED'
                )
                """
            ).fetchone()[0]
        finally:
            connection.close()
    if active:
        raise SuiteError(f"{active} VoiceMemo transcription job(s) are active.")


def stage_private_file(
    adb_path: str,
    serial: str,
    source: Path,
    relative: str,
    expected_sha256: str,
) -> None:
    if not re.fullmatch(
        r"files/(?:gpu-audio|gpu-models|whispercpp-models|encoder-states)/"
        r"[A-Za-z0-9._-]+",
        relative,
    ):
        raise SuiteError(f"Unsafe app-private staging path: {relative}")
    run(
        adb(
            adb_path,
            serial,
            "shell",
            "run-as",
            PACKAGE,
            "mkdir",
            "-p",
            str(Path(relative).parent),
        )
    )
    command = adb(
        adb_path,
        serial,
        "shell",
        "run-as",
        PACKAGE,
        "tee",
        relative,
    )
    log("+ " + shlex.join(command) + f" < {source}")
    with source.open("rb") as stream:
        completed = subprocess.run(
            command,
            stdin=stream,
            stdout=subprocess.DEVNULL,
        )
    if completed.returncode != 0:
        raise SuiteError(f"Could not stage {source}.")
    device_hash = run(
        adb(
            adb_path,
            serial,
            "shell",
            "run-as",
            PACKAGE,
            "sha256sum",
            relative,
        ),
        capture=True,
    ).stdout.split()[0]
    if device_hash != expected_sha256:
        raise SuiteError(f"Staged SHA-256 mismatch for {source.name}.")


def remove_private_file(adb_path: str, serial: str, relative: str) -> None:
    if not re.fullmatch(
        r"files/(?:gpu-audio|gpu-models|whispercpp-models|encoder-states)/"
        r"[A-Za-z0-9._-]+",
        relative,
    ):
        raise SuiteError(f"Unsafe app-private cleanup path: {relative}")
    run(
        adb(
            adb_path,
            serial,
            "shell",
            "run-as",
            PACKAGE,
            "rm",
            "-f",
            relative,
        ),
        check=False,
    )


def thermal_snapshot(adb_path: str, serial: str) -> dict[str, Any]:
    thermal = run(
        adb(adb_path, serial, "shell", "dumpsys", "thermalservice"),
        capture=True,
    ).stdout
    battery = run(
        adb(adb_path, serial, "shell", "dumpsys", "battery"),
        capture=True,
    ).stdout
    status_match = re.search(r"Thermal Status:\s*(\d+)", thermal)
    battery_temperature = re.search(r"temperature:\s*(\d+)", battery)
    battery_level = re.search(r"level:\s*(\d+)", battery)
    return {
        "thermal_status": int(status_match.group(1)) if status_match else None,
        "battery_temperature_celsius": (
            int(battery_temperature.group(1)) / 10
            if battery_temperature
            else None
        ),
        "battery_level_percent": (
            int(battery_level.group(1)) if battery_level else None
        ),
    }


def wait_for_nominal_thermal(
    adb_path: str,
    serial: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        snapshot = thermal_snapshot(adb_path, serial)
        if snapshot["thermal_status"] == 0:
            return snapshot
        if time.monotonic() >= deadline:
            raise SuiteError(
                "Device did not return to thermal status 0 before the timeout."
            )
        log(f"Waiting for thermal status 0; current snapshot: {snapshot}")
        time.sleep(15)


def build_harness() -> tuple[Path, Path]:
    java_home = Path(
        os.environ.get(
            "JAVA_HOME",
            "/Applications/Android Studio.app/Contents/jbr/Contents/Home",
        )
    )
    environment = os.environ.copy()
    environment["JAVA_HOME"] = str(java_home)
    environment["PATH"] = str(java_home / "bin") + os.pathsep + environment["PATH"]
    run(
        [
            str(PROJECT_ROOT / "gradlew"),
            "--no-daemon",
            "-Pkotlin.compiler.execution.strategy=in-process",
            "-Pandroid.sourceset.disallowProvider=false",
            "-PincludeWhisperCppBenchmark=true",
            "-PuseFp16Fp32AccumGpuProbe=true",
            ":app:assembleDebug",
            ":app:assembleDebugAndroidTest",
        ],
        env=environment,
    )
    app_apk = PROJECT_ROOT / "app/build/outputs/apk/debug/app-debug.apk"
    test_apk = (
        PROJECT_ROOT
        / "app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
    )
    if not app_apk.is_file() or not test_apk.is_file():
        raise SuiteError("Benchmark APK output is missing.")
    return app_apk, test_apk


def installed_apk_sha256(
    adb_path: str,
    serial: str,
    package: str,
) -> str:
    package_path = run(
        adb(adb_path, serial, "shell", "pm", "path", package),
        capture=True,
    ).stdout.strip()
    if not package_path.startswith("package:"):
        raise SuiteError(f"Package is not installed: {package}")
    return run(
        adb(
            adb_path,
            serial,
            "shell",
            "sha256sum",
            package_path.removeprefix("package:"),
        ),
        capture=True,
    ).stdout.split()[0]


def normalized_words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return [match.group(0) for match in WORD_PATTERN.finditer(folded)]


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (0 if left_value == right_value else 1),
                )
            )
        previous = current
    return previous[-1]


def comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_words = normalized_words(str(reference["text"]))
    candidate_words = normalized_words(str(candidate["text"]))
    reference_tokens = [
        int(value)
        for value in reference["tokens"]
        if int(value) < TIMESTAMP_BEGIN_TOKEN_ID
    ]
    raw_candidate_tokens = [int(value) for value in candidate["tokens"]]
    candidate_tokens = [
        value for value in raw_candidate_tokens if value < TIMESTAMP_BEGIN_TOKEN_ID
    ]
    word_edits = edit_distance(reference_words, candidate_words)
    token_edits = edit_distance(reference_tokens, candidate_tokens)
    if not reference_words or not reference_tokens:
        raise SuiteError("The reference transcript must contain words and tokens.")
    return {
        "normalization": "lowercase Unicode words with internal apostrophe or hyphen",
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "candidate_words": len(candidate_words),
        "word_error_rate": word_edits / len(reference_words),
        "word_error_percent": 100 * word_edits / len(reference_words),
        "token_edits": token_edits,
        "reference_tokens": len(reference_tokens),
        "candidate_tokens": len(candidate_tokens),
        "candidate_timestamp_tokens": (
            len(raw_candidate_tokens) - len(candidate_tokens)
        ),
        "token_error_rate": token_edits / len(reference_tokens),
        "token_error_percent": 100 * token_edits / len(reference_tokens),
        "exact_text_match": reference["text"] == candidate["text"],
        "exact_token_match": reference_tokens == candidate_tokens,
    }


def device_properties(adb_path: str, serial: str) -> dict[str, str]:
    properties = {}
    for key in (
        "ro.product.manufacturer",
        "ro.product.model",
        "ro.product.device",
        "ro.build.version.release",
        "ro.build.version.sdk",
        "ro.soc.model",
    ):
        properties[key] = run(
            adb(adb_path, serial, "shell", "getprop", key),
            capture=True,
        ).stdout.strip()
    return properties


def validate_local_inputs(
    audio: Path,
    baseline_path: Path,
    requested_language: str,
) -> dict[str, Any]:
    for path in (audio, baseline_path, TOKENIZER, GENERATION_CONFIG):
        if not path.is_file():
            raise SuiteError(f"Required benchmark input is missing: {path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("model_id") != "openai/whisper-large-v3-turbo":
        raise SuiteError("Baseline model ID is not the pinned OpenAI model.")
    if baseline.get("model_revision") != TOKENIZER_REVISION:
        raise SuiteError("Baseline/tokenizer revision is not pinned as expected.")
    if baseline.get("language") != requested_language:
        raise SuiteError(
            f"Baseline language must match requested mode {requested_language}."
        )
    expected_samples = int(baseline["audio_samples_16khz"])
    if expected_samples <= 0:
        raise SuiteError("Baseline audio sample count is invalid.")
    with wave.open(str(audio), "rb") as wav:
        wav_properties = {
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "sample_rate_hz": wav.getframerate(),
            "sample_count": wav.getnframes(),
            "compression": wav.getcomptype(),
        }
    expected_wav = {
        "channels": 1,
        "sample_width_bytes": 2,
        "sample_rate_hz": SAMPLE_RATE,
        "sample_count": expected_samples,
        "compression": "NONE",
    }
    if wav_properties != expected_wav:
        raise SuiteError(
            f"Reference WAV properties differ from the baseline: {wav_properties}"
        )
    generation_config = json.loads(GENERATION_CONFIG.read_text(encoding="utf-8"))
    configured_language_token = generation_config.get("lang_to_id", {}).get(
        f"<|{REFERENCE_LANGUAGE}|>"
    )
    if configured_language_token != LANGUAGE_TOKEN_ID:
        raise SuiteError(
            "Pinned generation config does not map German to token "
            f"{LANGUAGE_TOKEN_ID}."
        )
    for configuration in CONFIGURATIONS:
        model = configuration.resolved_model
        if not model.is_file():
            raise SuiteError(f"Model is missing: {model}")
        if model.stat().st_size != configuration.model_bytes:
            raise SuiteError(f"Model size mismatch: {model}")
        if sha256(model) != configuration.model_sha256:
            raise SuiteError(f"Model SHA-256 mismatch: {model}")
    return baseline


def instrumentation_arguments(
    *,
    configuration: Configuration,
    model_name: str,
    audio_name: str,
    run_id: str,
    requested_language: str,
    timestamp_tokens_enabled: bool,
) -> dict[str, Any]:
    arguments = {
        "class": TEST_CLASS,
        "runId": run_id,
        "candidateOnly": "true",
        "externalModelName": model_name,
        "externalModelSha256": configuration.model_sha256,
        "externalAudioName": audio_name,
        "timestampTokensEnabled": str(timestamp_tokens_enabled).lower(),
        **configuration.instrumentation,
    }
    if requested_language != "auto":
        arguments["forcedLanguage"] = requested_language
    return arguments


def run_instrumentation(
    *,
    adb_path: str,
    serial: str,
    configuration: Configuration,
    model_name: str,
    audio_name: str,
    run_id: str,
    requested_language: str,
    timestamp_tokens_enabled: bool,
) -> dict[str, Any]:
    arguments = instrumentation_arguments(
        configuration=configuration,
        model_name=model_name,
        audio_name=audio_name,
        run_id=run_id,
        requested_language=requested_language,
        timestamp_tokens_enabled=timestamp_tokens_enabled,
    )
    command = adb(adb_path, serial, "shell", "am", "instrument", "-w", "-r")
    for key, value in arguments.items():
        command.extend(("-e", key, value))
    command.append(f"{TEST_PACKAGE}/{TEST_RUNNER}")
    completed = run(command, capture=True, check=False)
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode != 0 or "OK (1 test)" not in completed.stdout:
        raise SuiteError(
            f"Instrumentation failed for {configuration.id}/{run_id}."
        )
    report_relative = f"cache/gpu-wer/whole-{run_id}.json"
    report_text = run(
        adb(
            adb_path,
            serial,
            "exec-out",
            "run-as",
            PACKAGE,
            "cat",
            report_relative,
        ),
        capture=True,
    ).stdout
    try:
        return json.loads(report_text)
    except json.JSONDecodeError as error:
        raise SuiteError(f"Invalid JSON report for {run_id}.") from error


def validate_report(
    report: dict[str, Any],
    configuration: Configuration,
    expected_samples: int,
    requested_language: str,
    timestamp_tokens_enabled: bool,
) -> None:
    candidate = report.get("candidate")
    if report.get("status") != "completed" or not isinstance(candidate, dict):
        raise SuiteError(f"Incomplete report for {configuration.id}.")
    expected_detected_language = (
        AUTO_DETECTED_LANGUAGE
        if requested_language == "auto"
        else requested_language
    )
    checks = {
        "requested language": (
            report.get("requested_language") == requested_language
        ),
        "detected language": (
            candidate.get("detected_language") == expected_detected_language
        ),
        "model SHA-256": report.get("model_sha256") == configuration.model_sha256,
        "mode": candidate.get("mode") == configuration.expected_mode,
        "sample count": (
            report.get("audio", {}).get("whisper_sample_count") == expected_samples
        ),
        "text": isinstance(candidate.get("text"), str),
        "tokens": isinstance(candidate.get("tokens"), list),
        "chunks": isinstance(candidate.get("chunks"), list),
        "inference timing": int(candidate.get("inference_wall_millis", 0)) > 0,
        "total timing": int(candidate.get("total_wall_millis", 0)) > 0,
        "timestamp tokens enabled": (
            report.get("timestamp_tokens_enabled") is timestamp_tokens_enabled
        ),
        "timestamp token count": (
            int(candidate.get("timestamp_token_count", 0)) > 0
            if timestamp_tokens_enabled
            else int(candidate.get("timestamp_token_count", 0)) == 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SuiteError(
            f"Report validation failed for {configuration.id}: {failed}"
        )
    if configuration.runtime == "whispercpp":
        warnings = " ".join(str(value) for value in candidate.get("warnings", []))
        if "Vulkan=Vulkan0 (Mali-G710)" not in warnings:
            raise SuiteError(
                f"whisper.cpp Vulkan device evidence is missing for {configuration.id}."
            )


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "population_standard_deviation": statistics.pstdev(values),
    }


def summarize(
    suite_id: str,
    manifest: dict[str, Any],
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    configurations = []
    audio_seconds = manifest["audio"]["samples_16khz"] / SAMPLE_RATE
    for configuration in CONFIGURATIONS:
        runs = results[configuration.id]
        inference = [
            float(item["raw_report"]["candidate"]["inference_wall_millis"])
            for item in runs
        ]
        total = [
            float(item["raw_report"]["candidate"]["total_wall_millis"])
            for item in runs
        ]
        pss = [
            float(
                item["raw_report"]["candidate"]["memory_after_inference"][
                    "total_pss_kib"
                ]
            )
            for item in runs
        ]
        wers = [
            float(item["comparison"]["word_error_percent"])
            for item in runs
        ]
        token_counts = [
            float(item["raw_report"]["candidate"]["token_count"])
            for item in runs
        ]
        timestamp_token_counts = [
            float(item["raw_report"]["candidate"]["timestamp_token_count"])
            for item in runs
        ]
        text_hashes = [
            hashlib.sha256(
                item["raw_report"]["candidate"]["text"].encode()
            ).hexdigest()
            for item in runs
        ]
        token_hashes = [
            sha256_json(item["raw_report"]["candidate"]["tokens"])
            for item in runs
        ]
        configurations.append(
            {
                "id": configuration.id,
                "label": configuration.label,
                "model_sha256": configuration.model_sha256,
                "runs": len(runs),
                "inference_millis": aggregate(inference),
                "total_millis": aggregate(total),
                "real_time_factor": aggregate(
                    [value / 1000 / audio_seconds for value in inference]
                ),
                "word_error_percent": aggregate(wers),
                "token_count": aggregate(token_counts),
                "timestamp_token_count": aggregate(timestamp_token_counts),
                "pss_kib": aggregate(pss),
                "deterministic_text": len(set(text_hashes)) == 1,
                "deterministic_tokens": len(set(token_hashes)) == 1,
                "text_sha256_per_run": text_hashes,
                "tokens_sha256_per_run": token_hashes,
            }
        )
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "status": "completed",
        "language": manifest["language"],
        "timestamp_tokens_enabled": manifest["timestamp_tokens_enabled"],
        "runs_per_configuration": manifest["runs_per_configuration"],
        "audio": manifest["audio"],
        "baseline": manifest["baseline"],
        "configurations": configurations,
    }


def write_markdown(summary: dict[str, Any], destination: Path) -> None:
    lines = [
        f"# Whisper Android benchmark suite {summary['suite_id']}",
        "",
        f"- Requested language: `{summary['language']}`",
        f"- Timestamp tokens enabled: `{str(summary['timestamp_tokens_enabled']).lower()}`",
        f"- Runs per configuration: {summary['runs_per_configuration']}",
        f"- Audio SHA-256: `{summary['audio']['sha256']}`",
        f"- Baseline SHA-256: `{summary['baseline']['sha256']}`",
        "",
        "| Configuration | Median inference | Range | Median RTF | "
        "Median WER | Median timestamp tokens | Median PSS | Deterministic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary["configurations"]:
        inference = item["inference_millis"]
        lines.append(
            "| {label} | {median:.3f} s | {minimum:.3f}–{maximum:.3f} s | "
            "{rtf:.3f} | {wer:.3f}% | {timestamps:.0f} | {pss:.2f} GiB | "
            "{deterministic} |".format(
                label=item["label"],
                median=inference["median"] / 1000,
                minimum=inference["min"] / 1000,
                maximum=inference["max"] / 1000,
                rtf=item["real_time_factor"]["median"],
                wer=item["word_error_percent"]["median"],
                timestamps=item["timestamp_token_count"]["median"],
                pss=item["pss_kib"]["median"] / 1024 / 1024,
                deterministic=(
                    "yes"
                    if item["deterministic_text"] and item["deterministic_tokens"]
                    else "no"
                ),
            )
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(args: argparse.Namespace) -> Path:
    audio = args.audio.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    baseline = validate_local_inputs(audio, baseline_path, args.language)
    adb_path = args.adb
    serial = one_device(adb_path, args.serial)
    assert_device_idle(adb_path, serial)
    suite_id = args.suite_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    if not SAFE_RUN_ID.fullmatch(suite_id):
        raise SuiteError("--suite-id contains unsupported characters.")
    output = args.output.expanduser().resolve() / suite_id
    if output.exists() and not args.resume:
        raise SuiteError(f"Refusing to overwrite suite directory: {output}")
    if args.resume and not output.is_dir():
        raise SuiteError(f"Cannot resume missing suite directory: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)

    audio_sha = sha256(audio)
    baseline_sha = sha256(baseline_path)
    tokenizer_sha = sha256(TOKENIZER)
    generation_config_sha = sha256(GENERATION_CONFIG)
    proposed_manifest = {
        "schema_version": 1,
        "suite_id": suite_id,
        "status": "preparing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_per_configuration": args.runs,
        "language": args.language,
        "timestamp_tokens_enabled": args.timestamp_tokens_enabled,
        "audio": {
            "path": str(audio),
            "bytes": audio.stat().st_size,
            "sha256": audio_sha,
            "samples_16khz": int(baseline["audio_samples_16khz"]),
        },
        "baseline": {
            "path": str(baseline_path),
            "bytes": baseline_path.stat().st_size,
            "sha256": baseline_sha,
            "model_id": baseline["model_id"],
            "model_revision": baseline["model_revision"],
            "language": baseline["language"],
            "token_count": len(baseline["tokens"]),
            "text_sha256": hashlib.sha256(baseline["text"].encode()).hexdigest(),
        },
        "tokenizer": {
            "path": str(TOKENIZER),
            "revision": TOKENIZER_REVISION,
            "sha256": tokenizer_sha,
            "generation_config_sha256": generation_config_sha,
            "forced_language": (
                None if args.language == "auto" else args.language
            ),
            "forced_language_token_id": (
                None if args.language == "auto" else LANGUAGE_TOKEN_ID
            ),
        },
        "whisper_cpp_revision": WHISPER_CPP_REVISION,
        "configurations": [asdict(value) for value in CONFIGURATIONS],
        "device_serial": serial,
        "device": device_properties(adb_path, serial),
    }
    manifest_path = output / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stable_fields = (
            "suite_id",
            "runs_per_configuration",
            "language",
            "timestamp_tokens_enabled",
            "audio",
            "baseline",
            "tokenizer",
            "whisper_cpp_revision",
            "configurations",
            "device_serial",
            "device",
        )
        changed = [
            field
            for field in stable_fields
            if manifest.get(field) != proposed_manifest.get(field)
        ]
        if changed:
            raise SuiteError(
                f"Cannot resume because manifest fields changed: {changed}"
            )
        if manifest.get("status") == "completed":
            raise SuiteError("The requested suite is already completed.")
    else:
        manifest = proposed_manifest
        atomic_json(manifest_path, manifest)

    app_apk = PROJECT_ROOT / "app/build/outputs/apk/debug/app-debug.apk"
    test_apk = (
        PROJECT_ROOT
        / "app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
    )
    reuse_installed_harness = (
        args.resume
        and app_apk.is_file()
        and test_apk.is_file()
        and manifest.get("app_apk_sha256") == sha256(app_apk)
        and manifest.get("test_apk_sha256") == sha256(test_apk)
        and installed_apk_sha256(adb_path, serial, PACKAGE)
        == manifest.get("app_apk_sha256")
        and installed_apk_sha256(adb_path, serial, TEST_PACKAGE)
        == manifest.get("test_apk_sha256")
    )
    if not reuse_installed_harness:
        app_apk, test_apk = build_harness()
        assert_device_idle(adb_path, serial)
        run(adb(adb_path, serial, "install", "-r", str(app_apk)))
        run(adb(adb_path, serial, "install", "-r", str(test_apk)))
    app_sha = sha256(app_apk)
    test_sha = sha256(test_apk)
    if installed_apk_sha256(adb_path, serial, PACKAGE) != app_sha:
        raise SuiteError("Installed app APK SHA-256 mismatch.")
    if installed_apk_sha256(adb_path, serial, TEST_PACKAGE) != test_sha:
        raise SuiteError("Installed test APK SHA-256 mismatch.")
    manifest["app_apk_sha256"] = app_sha
    manifest["test_apk_sha256"] = test_sha
    manifest["status"] = "running"
    atomic_json(manifest_path, manifest)

    audio_name = audio.name
    if not SAFE_NAME.fullmatch(audio_name):
        raise SuiteError("Audio file name is unsafe for app-private staging.")
    audio_relative = f"files/gpu-audio/{audio_name}"
    results: dict[str, list[dict[str, Any]]] = {
        configuration.id: [] for configuration in CONFIGURATIONS
    }
    if args.resume:
        for configuration in CONFIGURATIONS:
            for run_number in range(1, args.runs + 1):
                raw_path = (
                    output
                    / "raw"
                    / configuration.id
                    / f"run-{run_number}.json"
                )
                comparison_path = (
                    output
                    / "comparisons"
                    / configuration.id
                    / f"run-{run_number}.json"
                )
                if not raw_path.exists() and not comparison_path.exists():
                    break
                if not raw_path.is_file() or not comparison_path.is_file():
                    raise SuiteError(
                        f"Incomplete saved evidence for {configuration.id} "
                        f"run {run_number}."
                    )
                raw_report = json.loads(raw_path.read_text(encoding="utf-8"))
                evidence = json.loads(comparison_path.read_text(encoding="utf-8"))
                validate_report(
                    raw_report,
                    configuration,
                    int(baseline["audio_samples_16khz"]),
                    args.language,
                    args.timestamp_tokens_enabled,
                )
                compared = comparison(baseline, raw_report["candidate"])
                if evidence.get("raw_report_sha256") != sha256(raw_path):
                    raise SuiteError(
                        f"Saved raw-report hash mismatch for {configuration.id} "
                        f"run {run_number}."
                    )
                if evidence.get("comparison") != compared:
                    raise SuiteError(
                        f"Saved comparison mismatch for {configuration.id} "
                        f"run {run_number}."
                    )
                results[configuration.id].append(
                    {
                        "raw_report": raw_report,
                        "comparison": compared,
                        "evidence_path": str(comparison_path.relative_to(output)),
                    }
                )
    staged_model: str | None = None
    audio_staged = False
    try:
        stage_private_file(
            adb_path,
            serial,
            audio,
            audio_relative,
            audio_sha,
        )
        audio_staged = True
        for configuration in CONFIGURATIONS:
            completed_runs = len(results[configuration.id])
            if completed_runs == args.runs:
                log(f"Reusing {args.runs} completed runs for {configuration.id}.")
                continue
            assert_device_idle(adb_path, serial)
            model = configuration.resolved_model
            model_name = model.name
            if not SAFE_NAME.fullmatch(model_name):
                raise SuiteError(f"Unsafe model file name: {model_name}")
            staged_model = f"{configuration.model_directory}/{model_name}"
            stage_private_file(
                adb_path,
                serial,
                model,
                staged_model,
                configuration.model_sha256,
            )
            try:
                for run_number in range(completed_runs + 1, args.runs + 1):
                    if completed_runs > 0 or run_number > 1:
                        if not args.cooldown_seconds:
                            pass
                        else:
                            log(
                                f"Cooling down for {args.cooldown_seconds}s before "
                                f"{configuration.id} run {run_number}."
                            )
                            time.sleep(args.cooldown_seconds)
                    pre_thermal = wait_for_nominal_thermal(
                        adb_path,
                        serial,
                        args.thermal_timeout_seconds,
                    )
                    assert_device_idle(adb_path, serial)
                    run_id = instrumentation_run_id(
                        suite_id,
                        configuration.id,
                        run_number,
                    )
                    raw_report = run_instrumentation(
                        adb_path=adb_path,
                        serial=serial,
                        configuration=configuration,
                        model_name=model_name,
                        audio_name=audio_name,
                        run_id=run_id,
                        requested_language=args.language,
                        timestamp_tokens_enabled=args.timestamp_tokens_enabled,
                    )
                    validate_report(
                        raw_report,
                        configuration,
                        int(baseline["audio_samples_16khz"]),
                        args.language,
                        args.timestamp_tokens_enabled,
                    )
                    post_thermal = thermal_snapshot(adb_path, serial)
                    raw_path = (
                        output
                        / "raw"
                        / configuration.id
                        / f"run-{run_number}.json"
                    )
                    atomic_json(raw_path, raw_report)
                    compared = comparison(baseline, raw_report["candidate"])
                    evidence = {
                        "schema_version": 1,
                        "suite_id": suite_id,
                        "configuration_id": configuration.id,
                        "run_number": run_number,
                        "raw_report_path": str(raw_path.relative_to(output)),
                        "raw_report_sha256": sha256(raw_path),
                        "audio_sha256": audio_sha,
                        "baseline_sha256": baseline_sha,
                        "tokenizer_revision": TOKENIZER_REVISION,
                        "tokenizer_sha256": tokenizer_sha,
                        "generation_config_sha256": generation_config_sha,
                        "model_sha256": configuration.model_sha256,
                        "app_apk_sha256": app_sha,
                        "test_apk_sha256": test_sha,
                        "requested_language": args.language,
                        "expected_detected_language": (
                            AUTO_DETECTED_LANGUAGE
                            if args.language == "auto"
                            else args.language
                        ),
                        "forced_language_token_id": (
                            None if args.language == "auto" else LANGUAGE_TOKEN_ID
                        ),
                        "thermal_before": pre_thermal,
                        "thermal_after": post_thermal,
                        "comparison": compared,
                    }
                    comparison_path = (
                        output
                        / "comparisons"
                        / configuration.id
                        / f"run-{run_number}.json"
                    )
                    atomic_json(comparison_path, evidence)
                    results[configuration.id].append(
                        {
                            "raw_report": raw_report,
                            "comparison": compared,
                            "evidence_path": str(comparison_path.relative_to(output)),
                        }
                    )
            finally:
                remove_private_file(adb_path, serial, staged_model)
                staged_model = None
    finally:
        if staged_model is not None:
            remove_private_file(adb_path, serial, staged_model)
        if audio_staged:
            remove_private_file(adb_path, serial, audio_relative)

    summary = summarize(suite_id, manifest, results)
    atomic_json(output / "summary.json", summary)
    write_markdown(summary, output / "summary.md")
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_sha256"] = sha256(output / "summary.json")
    atomic_json(output / "manifest.json", manifest)
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
        child.add_argument(
            "--baseline",
            type=Path,
            help=(
                "Defaults to baseline.json for forced German and "
                "baseline-auto.json for auto language."
            ),
        )
        child.add_argument("--runs", type=int, default=3)
        child.add_argument(
            "--language",
            choices=("de", "auto"),
            default="de",
            help="Force German (default) or let each backend auto-detect language.",
        )
        child.add_argument(
            "--timestamp-tokens-enabled",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Enable timestamp-token decoding. Use "
                "--no-timestamp-tokens-enabled to reproduce older text-only "
                "benchmark suites."
            ),
        )
        child.add_argument(
            "--configuration",
            action="append",
            choices=[value.id for value in CONFIGURATIONS],
            help=(
                "Run only the selected configuration; repeat for multiple "
                "configurations. Defaults to the complete suite."
            ),
        )
        child.add_argument("--serial")
        child.add_argument(
            "--adb",
            default=os.environ.get(
                "ADB",
                str(Path.home() / "Library/Android/sdk/platform-tools/adb"),
            ),
        )
        child.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
        child.add_argument("--suite-id")
        child.add_argument(
            "--resume",
            action="store_true",
            help="Resume and verify an incomplete suite without rerunning saved runs.",
        )
        child.add_argument("--cooldown-seconds", type=int, default=30)
        child.add_argument("--thermal-timeout-seconds", type=int, default=900)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    global CONFIGURATIONS
    args = parser().parse_args(argv)
    if args.baseline is None:
        args.baseline = (
            PERSISTENT_INPUT_ROOT / "baseline-auto.json"
            if args.language == "auto"
            else DEFAULT_BASELINE
        )
    if args.configuration:
        requested = set(args.configuration)
        CONFIGURATIONS = tuple(
            value for value in CONFIGURATIONS if value.id in requested
        )
    if args.runs != 3:
        raise SuiteError("This controlled suite requires exactly three runs.")
    if args.cooldown_seconds < 0 or args.cooldown_seconds > 600:
        raise SuiteError("--cooldown-seconds must be between 0 and 600.")
    if args.command == "plan":
        baseline = validate_local_inputs(
            args.audio.expanduser().resolve(),
            args.baseline.expanduser().resolve(),
            args.language,
        )
        print(
            json.dumps(
                {
                    "runs": args.runs,
                    "language": args.language,
                    "timestamp_tokens_enabled": args.timestamp_tokens_enabled,
                    "audio_samples_16khz": baseline["audio_samples_16khz"],
                    "configurations": [asdict(value) for value in CONFIGURATIONS],
                },
                indent=2,
            )
        )
        return 0
    output = run_suite(args)
    log(f"Suite report: {output / 'summary.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SuiteError, subprocess.CalledProcessError) as error:
        print(f"Whisper benchmark suite failed: {error}", file=sys.stderr)
        raise SystemExit(2)
