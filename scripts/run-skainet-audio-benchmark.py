#!/usr/bin/env python3
"""Run the isolated VoiceMemo versus SKaiNET-audio benchmark on Pixel 7 Pro."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import textwrap
import time
import unicodedata
import wave


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "com.example.voicememo.benchmark"
TEST_PACKAGE = f"{PACKAGE}.test"
REAL_PACKAGE = "com.example.voicememo"
RUNNER = "androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = (
    "com.example.voicememo.transcription."
    "SkainetAudioBenchmarkTest#runBenchmark"
)
REPORT_DIRECTORY = "skainet-audio-benchmark"
SKAINET_VERSION = "0.2.1"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
MINIMUM_FREE_KIB = 8 * 1024 * 1024
REPEATS = 3
FRONTENDS = ("voicememo", "skainet")
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9-]{1,128}")
SAFE_AUDIO_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.wav")


@dataclass(frozen=True)
class Sample:
    id: str
    audio: Path
    baseline: Path
    language: str
    audio_sha256: str
    baseline_sha256: str


SAMPLES = (
    Sample(
        "english-dialogue-short-9s",
        ROOT / "benchmark-inputs/article-samples/english-dialogue-short/audio.wav",
        ROOT / "benchmark-inputs/article-samples/english-dialogue-short/baseline.json",
        "en",
        "ef4e4b07a08679a2c708b5fa159ef036c8ae7415a56a69c34cd5142832ef4b48",
        "a81c2a5935c844b40d02b5856197bf5772b28934603856308bbc1eb6971debd3",
    ),
    Sample(
        "english-dialogue-proper-names-47s",
        ROOT / "benchmark-inputs/article-samples/english-dialogue-proper-names/audio.wav",
        ROOT / "benchmark-inputs/article-samples/english-dialogue-proper-names/baseline.json",
        "en",
        "2c25dda8131ed747340566a5c67312a88325563cb66d2b18cfe4d81f45c2aeb1",
        "b08b489bd5867e92529601bfb4bb2c0f21dc05cb53807db0497b5f76cb85eb6f",
    ),
    Sample(
        "english-dialogue-mixed-fragments-51s",
        ROOT / "benchmark-inputs/article-samples/english-dialogue-mixed-fragments/audio.wav",
        ROOT / "benchmark-inputs/article-samples/english-dialogue-mixed-fragments/baseline.json",
        "en",
        "49fbb2b621336e0ebff646074f77c6875fdc82066615ca20e2cb37b078133ae9",
        "5b063ffd9580d3b67c83c3a963ba793cc666367b389ea5f599f641a227e6cb71",
    ),
    Sample(
        "sample-test-153s",
        ROOT / "benchmark-inputs/sample-test/audio.wav",
        ROOT / "benchmark-inputs/sample-test/baseline.json",
        "de",
        "6d2b2be2e010d7c139071ac5d4283b17dbe6834bb07251ccbde83d280e5ace4e",
        "17e0625e3ad04a84cb7842b3156ccfccd104c0f6f1905e0f020d48779ec9252e",
    ),
    Sample(
        "controlled-german-170s",
        ROOT / "benchmark-inputs/whisper-suite/reference.wav",
        ROOT / "benchmark-inputs/whisper-suite/baseline.json",
        "de",
        "138fc04d96018548265f9d729f20913033cfb902cfc590da5f5559c68bae6ce6",
        "7dcd842c1d9eff5791273259176022555bdc1b2f46a890c574d5d13173bb9367",
    ),
)


class BenchmarkError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def run(
    command: list[str],
    *,
    capture: bool = False,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log("+ " + " ".join(command))
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise BenchmarkError(
            f"Command failed ({completed.returncode}): {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return completed


def adb(adb_path: str, serial: str, *arguments: str) -> list[str]:
    return [adb_path, "-s", serial, *arguments]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def validate_samples() -> dict[str, dict]:
    result: dict[str, dict] = {}
    if len(SAMPLES) != 5 or len({sample.id for sample in SAMPLES}) != 5:
        raise BenchmarkError("The benchmark must contain exactly five unique samples.")
    for sample in SAMPLES:
        for path in (sample.audio, sample.baseline):
            if not path.is_file():
                raise BenchmarkError(f"Benchmark input is missing: {path}")
        if sha256(sample.audio) != sample.audio_sha256:
            raise BenchmarkError(f"Audio SHA-256 mismatch: {sample.audio}")
        if sha256(sample.baseline) != sample.baseline_sha256:
            raise BenchmarkError(f"Baseline SHA-256 mismatch: {sample.baseline}")
        baseline = json.loads(sample.baseline.read_text(encoding="utf-8"))
        expected = {
            "model_id": "openai/whisper-large-v3-turbo",
            "model_revision": MODEL_REVISION,
            "language": sample.language,
            "window_seconds": 28,
            "overlap_seconds": 2,
        }
        mismatches = [key for key, value in expected.items() if baseline.get(key) != value]
        if mismatches:
            raise BenchmarkError(f"Baseline metadata mismatch for {sample.id}: {mismatches}")
        if not isinstance(baseline.get("text"), str) or not baseline["text"].strip():
            raise BenchmarkError(f"Baseline text is missing for {sample.id}.")
        with wave.open(str(sample.audio), "rb") as audio:
            properties = {
                "channels": audio.getnchannels(),
                "sample_width_bytes": audio.getsampwidth(),
                "sample_rate_hz": audio.getframerate(),
                "sample_count": audio.getnframes(),
                "compression": audio.getcomptype(),
            }
        required = {
            "channels": 1,
            "sample_width_bytes": 2,
            "sample_rate_hz": 16_000,
            "sample_count": int(baseline["audio_samples_16khz"]),
            "compression": "NONE",
        }
        if properties != required:
            raise BenchmarkError(
                f"WAV properties differ from baseline for {sample.id}: {properties}"
            )
        result[sample.id] = {
            "audio": str(sample.audio.relative_to(ROOT)),
            "baseline_path": str(sample.baseline.relative_to(ROOT)),
            "language": sample.language,
            "audio_sha256": sample.audio_sha256,
            "baseline_sha256": sample.baseline_sha256,
            "sample_count": properties["sample_count"],
            "duration_seconds": properties["sample_count"] / 16_000,
            "baseline": baseline,
        }
    return result


def resolve_serial(adb_path: str, requested: str | None) -> str:
    output = run([adb_path, "devices"], capture=True).stdout
    devices = [
        line.split()[0]
        for line in output.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    if requested:
        if requested not in devices:
            raise BenchmarkError(f"Requested ADB device is not authorized: {requested}")
        return requested
    if len(devices) != 1:
        raise BenchmarkError(f"Expected exactly one authorized ADB device, found {devices}.")
    return devices[0]


def getprop(adb_path: str, serial: str, name: str) -> str:
    return run(adb(adb_path, serial, "shell", "getprop", name), capture=True).stdout.strip()


def validate_device(adb_path: str, serial: str) -> dict[str, object]:
    properties = {
        "serial": serial,
        "manufacturer": getprop(adb_path, serial, "ro.product.manufacturer"),
        "model": getprop(adb_path, serial, "ro.product.model"),
        "device": getprop(adb_path, serial, "ro.product.device"),
        "android_release": getprop(adb_path, serial, "ro.build.version.release"),
        "sdk": int(getprop(adb_path, serial, "ro.build.version.sdk")),
        "build_fingerprint": getprop(adb_path, serial, "ro.build.fingerprint"),
        "security_patch": getprop(adb_path, serial, "ro.build.version.security_patch"),
        "soc": getprop(adb_path, serial, "ro.soc.model"),
        "abis": getprop(adb_path, serial, "ro.product.cpu.abilist").split(","),
    }
    if properties["model"] != "Pixel 7 Pro" or properties["device"] != "cheetah":
        raise BenchmarkError(f"Benchmark target must be Pixel 7 Pro/cheetah: {properties}")
    if properties["sdk"] < 24 or "arm64-v8a" not in properties["abis"]:
        raise BenchmarkError(f"Device SDK/ABI is incompatible: {properties}")
    df = run(adb(adb_path, serial, "shell", "df", "-Pk", "/data"), capture=True).stdout
    fields = df.strip().splitlines()[-1].split()
    if len(fields) < 6:
        raise BenchmarkError(f"Could not parse /data free space: {df}")
    free_kib = int(fields[3])
    properties["data_free_kib"] = free_kib
    if free_kib < MINIMUM_FREE_KIB:
        raise BenchmarkError(
            f"Pixel has {free_kib / 1024 / 1024:.2f} GiB free; 8 GiB is required."
        )
    return properties


def package_snapshot(adb_path: str, serial: str, package: str) -> dict[str, object]:
    paths = run(
        adb(adb_path, serial, "shell", "pm", "path", package),
        capture=True,
        check=False,
    )
    if paths.returncode != 0 or "package:" not in paths.stdout:
        return {"installed": False}
    details = run(
        adb(adb_path, serial, "shell", "dumpsys", "package", package),
        capture=True,
    ).stdout

    def field(pattern: str) -> str | None:
        match = re.search(pattern, details, re.MULTILINE)
        return match.group(1).strip() if match else None

    return {
        "installed": True,
        "paths": sorted(
            line.removeprefix("package:").strip()
            for line in paths.stdout.splitlines()
            if line.startswith("package:")
        ),
        "version_code": field(r"^\s*versionCode=(\d+)"),
        "first_install_time": field(r"^\s*firstInstallTime=(.+)$"),
        "last_update_time": field(r"^\s*lastUpdateTime=(.+)$"),
        "uid": field(r"^\s*userId=(\d+)"),
    }


def assert_real_package_idle(adb_path: str, serial: str) -> None:
    services = run(
        adb(adb_path, serial, "shell", "dumpsys", "activity", "services", REAL_PACKAGE),
        capture=True,
    ).stdout
    if "RecordingService" in services or "SystemForegroundService" in services:
        raise BenchmarkError("VoiceMemo is recording or transcribing.")
    jobs = run(
        adb(adb_path, serial, "shell", "dumpsys", "jobscheduler", REAL_PACKAGE),
        capture=True,
        check=False,
    ).stdout
    if re.search(r"\bRUNNING\b", jobs):
        raise BenchmarkError("A VoiceMemo JobScheduler job is running.")


def assert_package_unchanged(before: dict[str, object], after: dict[str, object]) -> None:
    if after != before:
        raise BenchmarkError("Real VoiceMemo package changed during isolated benchmarking.")


def thermal_snapshot(adb_path: str, serial: str) -> dict[str, float | int | None]:
    thermal = run(
        adb(adb_path, serial, "shell", "dumpsys", "thermalservice"),
        capture=True,
    ).stdout
    battery = run(
        adb(adb_path, serial, "shell", "dumpsys", "battery"),
        capture=True,
    ).stdout
    status = re.search(r"Thermal Status:\s*(\d+)", thermal)
    temperature = re.search(r"temperature:\s*(\d+)", battery)
    level = re.search(r"level:\s*(\d+)", battery)
    skin = re.findall(
        r"Temperature\{mValue=([-+0-9.eE]+),\s*mType=3,\s*mName=VIRTUAL-SKIN,",
        thermal,
    )
    return {
        "thermal_status": int(status.group(1)) if status else None,
        "battery_temperature_celsius": int(temperature.group(1)) / 10 if temperature else None,
        "battery_level_percent": int(level.group(1)) if level else None,
        "skin_celsius": float(skin[-1]) if skin else None,
    }


def wait_for_thermal(adb_path: str, serial: str, timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while True:
        assert_real_package_idle(adb_path, serial)
        snapshot = thermal_snapshot(adb_path, serial)
        if snapshot["thermal_status"] == 0:
            return snapshot
        if time.monotonic() >= deadline:
            raise BenchmarkError(f"Thermal status did not return to NONE: {snapshot}")
        log(f"Waiting for thermal status NONE: {snapshot}")
        time.sleep(15)


def find_aapt() -> Path:
    sdk_root = Path(
        os.environ.get("ANDROID_SDK_ROOT")
        or os.environ.get("ANDROID_HOME")
        or Path.home() / "Library/Android/sdk"
    )
    candidates = [path for path in (sdk_root / "build-tools").glob("*/aapt") if path.is_file()]
    if not candidates:
        raise BenchmarkError(f"Android aapt is missing under {sdk_root / 'build-tools'}.")

    def version(path: Path) -> tuple[int, ...]:
        return tuple(int(value) for value in re.findall(r"\d+", path.parent.name))

    return max(candidates, key=version)


def parse_apk_application_id(aapt_badging: str) -> str:
    match = re.search(r"^package: name='([^']+)'", aapt_badging, re.MULTILINE)
    if not match:
        raise BenchmarkError("Could not read application ID from APK manifest.")
    return match.group(1)


def assert_apk_package(aapt: Path, apk: Path, expected: str) -> None:
    output = run([str(aapt), "dump", "badging", str(apk)], capture=True).stdout
    actual = parse_apk_application_id(output)
    if actual != expected or actual == REAL_PACKAGE:
        raise BenchmarkError(
            f"Refusing to install {apk}: expected package {expected}, found {actual}."
        )


def build_apks() -> tuple[Path, Path]:
    java_home = Path(
        os.environ.get("JAVA_HOME", "/Applications/Android Studio.app/Contents/jbr/Contents/Home")
    )
    if not (java_home / "bin/java").is_file():
        raise BenchmarkError(f"Java 17 runtime is missing: {java_home}")
    environment = os.environ.copy()
    environment["JAVA_HOME"] = str(java_home)
    environment["PATH"] = str(java_home / "bin") + os.pathsep + environment["PATH"]
    run(
        [
            str(ROOT / "gradlew"),
            "--no-daemon",
            "--no-build-cache",
            "-Pkotlin.compiler.execution.strategy=in-process",
            "-PisolatedBenchmarkPackage=true",
            ":app:assembleDebug",
            ":app:assembleDebugAndroidTest",
        ],
        env=environment,
    )
    app = ROOT / "app/build/outputs/apk/debug/app-debug.apk"
    test = ROOT / "app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
    if not app.is_file() or not test.is_file():
        raise BenchmarkError("Isolated benchmark APK outputs are missing.")
    aapt = find_aapt()
    assert_apk_package(aapt, app, PACKAGE)
    assert_apk_package(aapt, test, TEST_PACKAGE)
    return app, test


def install_apks(adb_path: str, serial: str, app: Path, test: Path) -> None:
    for apk in (app, test):
        run(adb(adb_path, serial, "install", "-r", str(apk)))
    target = package_snapshot(adb_path, serial, PACKAGE)
    instrumentation = package_snapshot(adb_path, serial, TEST_PACKAGE)
    if not target.get("installed") or not instrumentation.get("installed"):
        raise BenchmarkError("The isolated benchmark packages were not installed.")


def stage_audio(adb_path: str, serial: str, sample: Sample) -> str:
    name = f"{sample.id}.wav"
    if not SAFE_AUDIO_NAME.fullmatch(name):
        raise BenchmarkError(f"Unsafe staged audio name: {name}")
    relative = f"files/{REPORT_DIRECTORY}/{name}"
    run(adb(adb_path, serial, "shell", "run-as", PACKAGE, "mkdir", "-p", f"files/{REPORT_DIRECTORY}"))
    command = adb(adb_path, serial, "shell", "run-as", PACKAGE, "tee", relative)
    log("+ " + " ".join(command) + f" < {sample.audio}")
    with sample.audio.open("rb") as stream:
        completed = subprocess.run(command, stdin=stream, stdout=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise BenchmarkError(f"Could not stage {sample.audio}.")
    device_hash = run(
        adb(adb_path, serial, "shell", "run-as", PACKAGE, "sha256sum", relative),
        capture=True,
    ).stdout.split()[0]
    if device_hash != sample.audio_sha256:
        raise BenchmarkError(f"Staged audio hash differs for {sample.id}.")
    return name


def safe_remove(adb_path: str, serial: str, relative: str) -> None:
    allowed = re.fullmatch(
        rf"(?:files|cache)/{REPORT_DIRECTORY}/[A-Za-z0-9._-]+",
        relative,
    )
    if not allowed:
        raise BenchmarkError(f"Unsafe benchmark cleanup path: {relative}")
    run(
        adb(adb_path, serial, "shell", "run-as", PACKAGE, "rm", "-f", relative),
        check=False,
    )


def current_pss_kib(adb_path: str, serial: str) -> int | None:
    completed = subprocess.run(
        adb(adb_path, serial, "shell", "dumpsys", "meminfo", PACKAGE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return None
    match = re.search(r"TOTAL PSS:\s*([0-9,]+)", completed.stdout)
    if not match:
        match = re.search(r"^\s*TOTAL\s+([0-9,]+)", completed.stdout, re.MULTILINE)
    return int(match.group(1).replace(",", "")) if match else None


def instrumentation_command(
    adb_path: str,
    serial: str,
    run_id: str,
    mode: str,
    audio_name: str,
    language: str,
) -> list[str]:
    if not SAFE_RUN_ID.fullmatch(run_id) or mode not in (*FRONTENDS, "parity"):
        raise BenchmarkError("Unsafe instrumentation arguments.")
    arguments = {
        "class": TEST_CLASS,
        "runId": run_id,
        "mode": mode,
        "externalAudioName": audio_name,
        "forcedLanguage": language,
    }
    command = adb(adb_path, serial, "shell", "am", "instrument", "-w", "-r")
    for key, value in arguments.items():
        command.extend(("-e", key, value))
    command.append(f"{TEST_PACKAGE}/{RUNNER}")
    return command


def pull_report(adb_path: str, serial: str, run_id: str) -> dict:
    relative = f"cache/{REPORT_DIRECTORY}/{run_id}.json"
    completed = run(
        adb(adb_path, serial, "exec-out", "run-as", PACKAGE, "cat", relative),
        capture=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BenchmarkError(f"Could not read device report for {run_id}.")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"Device report is invalid JSON for {run_id}.") from error
    safe_remove(adb_path, serial, relative)
    return report


def execute_instrumentation(
    adb_path: str,
    serial: str,
    run_id: str,
    mode: str,
    audio_name: str,
    language: str,
    timeout_seconds: int,
) -> tuple[dict, list[dict[str, int]], str]:
    run(adb(adb_path, serial, "shell", "am", "force-stop", PACKAGE))
    command = instrumentation_command(adb_path, serial, run_id, mode, audio_name, language)
    log("+ " + " ".join(command))
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    started = time.monotonic()
    samples: list[dict[str, int]] = []
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            process.terminate()
            run(adb(adb_path, serial, "shell", "am", "force-stop", PACKAGE), check=False)
            raise BenchmarkError(f"Instrumentation timed out after {timeout_seconds}s: {run_id}")
        pss = current_pss_kib(adb_path, serial)
        if pss is not None:
            samples.append({"elapsed_millis": int(elapsed * 1000), "pss_kib": pss})
        time.sleep(1)
    output = process.communicate()[0] or ""
    print(output, end="", flush=True)
    report = pull_report(adb_path, serial, run_id)
    if process.returncode != 0 or "OK (1 test)" not in output:
        raise BenchmarkError(f"Instrumentation failed for {run_id}; report={report}")
    if report.get("status") != "completed" or report.get("package_name") != PACKAGE:
        raise BenchmarkError(f"Incomplete or non-isolated report for {run_id}.")
    return report, samples, output


def normalized_words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return [match.group(0) for match in WORD_PATTERN.finditer(folded)]


def word_error(reference: str, candidate: str) -> dict[str, int | float | str]:
    left = normalized_words(reference)
    right = normalized_words(candidate)
    if not left:
        raise BenchmarkError("WER reference contains no words.")
    rows = [[(0, 0, 0, 0) for _ in range(len(right) + 1)] for _ in range(len(left) + 1)]
    for index in range(1, len(left) + 1):
        rows[index][0] = (index, 0, index, 0)
    for index in range(1, len(right) + 1):
        rows[0][index] = (index, 0, 0, index)
    for row in range(1, len(left) + 1):
        for column in range(1, len(right) + 1):
            diagonal = rows[row - 1][column - 1]
            substitution = diagonal if left[row - 1] == right[column - 1] else (
                diagonal[0] + 1,
                diagonal[1] + 1,
                diagonal[2],
                diagonal[3],
            )
            deletion_base = rows[row - 1][column]
            deletion = (deletion_base[0] + 1, deletion_base[1], deletion_base[2] + 1, deletion_base[3])
            insertion_base = rows[row][column - 1]
            insertion = (insertion_base[0] + 1, insertion_base[1], insertion_base[2], insertion_base[3] + 1)
            rows[row][column] = min(substitution, deletion, insertion)
    edits, substitutions, deletions, insertions = rows[-1][-1]
    return {
        "reference": "pinned OpenAI Whisper large-v3-turbo FP32 baseline",
        "normalization": "NFKC casefolded Unicode words with internal apostrophe or hyphen",
        "reference_words": len(left),
        "candidate_words": len(right),
        "word_edits": edits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "word_error_rate": edits / len(left),
        "word_error_percent": 100 * edits / len(left),
    }


def run_file_valid(path: Path, sample: Sample, mode: str, repeat: int) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    raw_report = value.get("raw_report")
    if not isinstance(raw_report, dict):
        return False
    result = raw_report.get("result")
    if not isinstance(result, dict):
        return False
    return (
        value.get("sample_id") == sample.id
        and value.get("frontend") == mode
        and value.get("repeat") == repeat
        and value.get("audio_sha256") == sample.audio_sha256
        and raw_report.get("status") == "completed"
        and raw_report.get("package_name") == PACKAGE
        and result.get("frontend") == mode
    )


def parity_file_valid(path: Path, sample: Sample) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    raw_report = value.get("raw_report")
    if not isinstance(raw_report, dict):
        return False
    return (
        value.get("sample_id") == sample.id
        and value.get("audio_sha256") == sample.audio_sha256
        and raw_report.get("status") == "completed"
        and raw_report.get("package_name") == PACKAGE
        and isinstance(raw_report.get("parity"), dict)
    )


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "population_standard_deviation": statistics.pstdev(values),
    }


def load_runs(output: Path) -> list[dict]:
    values = []
    for path in sorted((output / "raw").glob("*/*.json")):
        values.append(json.loads(path.read_text(encoding="utf-8")))
    return values


def summarize(
    suite_id: str,
    manifest: dict[str, dict],
    runs: list[dict],
    parity: dict[str, dict],
    device: dict,
    package_before: dict,
    package_after: dict,
) -> dict:
    expected_parity_ids = {sample.id for sample in SAMPLES}
    if set(parity) != expected_parity_ids:
        raise BenchmarkError(
            f"Missing parity results: {sorted(expected_parity_ids - set(parity))}."
        )
    incomplete_parity = [
        sample_id
        for sample_id, value in parity.items()
        if value.get("raw_report", {}).get("status") != "completed"
    ]
    if incomplete_parity:
        raise BenchmarkError(f"Incomplete parity results: {sorted(incomplete_parity)}.")

    rows = []
    for sample in SAMPLES:
        for frontend in FRONTENDS:
            selected = sorted(
                (item for item in runs if item["sample_id"] == sample.id and item["frontend"] == frontend),
                key=lambda item: item["repeat"],
            )
            if len(selected) != REPEATS:
                raise BenchmarkError(f"Missing runs for {sample.id}/{frontend}.")
            results = [item["raw_report"]["result"] for item in selected]
            text_hashes = [hashlib.sha256(result["text"].encode()).hexdigest() for result in results]
            rows.append(
                {
                    "sample_id": sample.id,
                    "frontend": frontend,
                    "runs": len(selected),
                    "duration_seconds": manifest[sample.id]["duration_seconds"],
                    "pipeline_rtf": aggregate([float(result["pipeline_rtf"]) for result in results]),
                    "feature_extraction_rtf": aggregate(
                        [float(result["feature_extraction_rtf"]) for result in results]
                    ),
                    "frontend_initialization_wall_millis": aggregate(
                        [float(result["frontend_initialization_millis"]) for result in results]
                    ),
                    "model_load_wall_millis": aggregate(
                        [float(result["model_load_wall_millis"]) for result in results]
                    ),
                    "model_load_cpu_millis": aggregate(
                        [float(result["model_load_cpu_millis"]) for result in results]
                    ),
                    "input_preparation_wall_millis": aggregate(
                        [float(result["input_preparation_wall_millis"]) for result in results]
                    ),
                    "input_preparation_cpu_millis": aggregate(
                        [float(result["input_preparation_cpu_millis"]) for result in results]
                    ),
                    "feature_extraction_wall_millis": aggregate(
                        [float(result["feature_extraction_wall_millis"]) for result in results]
                    ),
                    "feature_extraction_cpu_millis": aggregate(
                        [float(result["feature_extraction_cpu_millis"]) for result in results]
                    ),
                    "inference_wall_millis": aggregate(
                        [float(result["inference_wall_millis"]) for result in results]
                    ),
                    "inference_cpu_millis": aggregate(
                        [float(result["inference_cpu_millis"]) for result in results]
                    ),
                    "pipeline_wall_millis": aggregate(
                        [float(result["pipeline_wall_millis"]) for result in results]
                    ),
                    "pipeline_cpu_millis": aggregate(
                        [float(result["pipeline_cpu_millis"]) for result in results]
                    ),
                    "peak_pss_kib": aggregate([float(item["peak_pss_kib"]) for item in selected]),
                    "word_error_percent": aggregate(
                        [float(item["word_error"]["word_error_percent"]) for item in selected]
                    ),
                    "word_edits": aggregate(
                        [float(item["word_error"]["word_edits"]) for item in selected]
                    ),
                    "substitutions": aggregate(
                        [float(item["word_error"]["substitutions"]) for item in selected]
                    ),
                    "deletions": aggregate(
                        [float(item["word_error"]["deletions"]) for item in selected]
                    ),
                    "insertions": aggregate(
                        [float(item["word_error"]["insertions"]) for item in selected]
                    ),
                    "deterministic_text": len(set(text_hashes)) == 1,
                    "text_sha256_per_run": text_hashes,
                }
            )
    corpus = []
    for frontend in FRONTENDS:
        repeat_values = []
        for repeat in range(1, REPEATS + 1):
            selected = [
                item for item in runs if item["frontend"] == frontend and item["repeat"] == repeat
            ]
            if len(selected) != len(SAMPLES):
                raise BenchmarkError(f"Incomplete corpus repeat {frontend}/{repeat}.")
            reference_words = sum(int(item["word_error"]["reference_words"]) for item in selected)
            edits = sum(int(item["word_error"]["word_edits"]) for item in selected)
            pipeline = sum(float(item["raw_report"]["result"]["pipeline_wall_millis"]) for item in selected)
            features = sum(float(item["raw_report"]["result"]["feature_extraction_wall_millis"]) for item in selected)
            duration = sum(float(manifest[item["sample_id"]]["duration_seconds"]) for item in selected)
            repeat_values.append(
                {
                    "repeat": repeat,
                    "reference_words": reference_words,
                    "word_edits": edits,
                    "word_error_percent": 100 * edits / reference_words,
                    "pipeline_rtf": pipeline / 1000 / duration,
                    "feature_extraction_rtf": features / 1000 / duration,
                    "substitutions": sum(int(item["word_error"]["substitutions"]) for item in selected),
                    "deletions": sum(int(item["word_error"]["deletions"]) for item in selected),
                    "insertions": sum(int(item["word_error"]["insertions"]) for item in selected),
                }
            )
        corpus.append(
            {
                "frontend": frontend,
                "repeats": repeat_values,
                "word_error_percent": aggregate([item["word_error_percent"] for item in repeat_values]),
                "pipeline_rtf": aggregate([item["pipeline_rtf"] for item in repeat_values]),
                "feature_extraction_rtf": aggregate(
                    [item["feature_extraction_rtf"] for item in repeat_values]
                ),
                "substitutions": aggregate(
                    [float(item["substitutions"]) for item in repeat_values]
                ),
                "deletions": aggregate(
                    [float(item["deletions"]) for item in repeat_values]
                ),
                "insertions": aggregate(
                    [float(item["insertions"]) for item in repeat_values]
                ),
            }
        )
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wer_reference": "pinned OpenAI Whisper large-v3-turbo FP32 baselines; not human ground truth",
        "skainet_audio_version": SKAINET_VERSION,
        "device": device,
        "real_package_proof": {
            "before": package_before,
            "after": package_after,
            "unchanged": package_before == package_after,
        },
        "samples": [{key: value for key, value in manifest[sample.id].items() if key != "baseline"} for sample in SAMPLES],
        "per_sample": rows,
        "corpus": corpus,
        "parity": parity,
    }


def write_outputs(output: Path, summary: dict, runs: list[dict]) -> None:
    atomic_json(output / "summary.json", summary)
    with (output / "summary.csv.tmp").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample_id",
                "frontend",
                "duration_seconds",
                "median_pipeline_rtf",
                "min_pipeline_rtf",
                "max_pipeline_rtf",
                "median_feature_rtf",
                "median_model_load_ms",
                "median_model_load_cpu_ms",
                "median_inference_ms",
                "median_inference_cpu_ms",
                "median_pipeline_ms",
                "median_pipeline_cpu_ms",
                "median_peak_pss_mib",
                "median_wer_percent",
                "median_substitutions",
                "median_deletions",
                "median_insertions",
                "deterministic_text",
            ]
        )
        for row in summary["per_sample"]:
            writer.writerow(
                [
                    row["sample_id"],
                    row["frontend"],
                    row["duration_seconds"],
                    row["pipeline_rtf"]["median"],
                    row["pipeline_rtf"]["min"],
                    row["pipeline_rtf"]["max"],
                    row["feature_extraction_rtf"]["median"],
                    row["model_load_wall_millis"]["median"],
                    row["model_load_cpu_millis"]["median"],
                    row["inference_wall_millis"]["median"],
                    row["inference_cpu_millis"]["median"],
                    row["pipeline_wall_millis"]["median"],
                    row["pipeline_cpu_millis"]["median"],
                    row["peak_pss_kib"]["median"] / 1024,
                    row["word_error_percent"]["median"],
                    row["substitutions"]["median"],
                    row["deletions"]["median"],
                    row["insertions"]["median"],
                    row["deterministic_text"],
                ]
            )
    (output / "summary.csv.tmp").replace(output / "summary.csv")

    lines = [
        "# SKaiNET-audio Pixel 7 Pro benchmark",
        "",
        "WER is measured against the pinned OpenAI Whisper large-v3-turbo FP32 outputs, not human ground truth.",
        "",
        f"Real VoiceMemo package unchanged: **{summary['real_package_proof']['unchanged']}**",
        "",
        "| Sample | Frontend | Median RTF | Feature RTF | WER | S/D/I | Peak PSS | Deterministic |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["per_sample"]:
        lines.append(
            "| {sample_id} | {frontend} | {rtf:.3f} | {feature:.3f} | "
            "{wer:.3f}% | {substitutions:.0f}/{deletions:.0f}/{insertions:.0f} | "
            "{pss:.1f} MiB | {deterministic} |".format(
                sample_id=row["sample_id"],
                frontend=row["frontend"],
                rtf=row["pipeline_rtf"]["median"],
                feature=row["feature_extraction_rtf"]["median"],
                wer=row["word_error_percent"]["median"],
                substitutions=row["substitutions"]["median"],
                deletions=row["deletions"]["median"],
                insertions=row["insertions"]["median"],
                pss=row["peak_pss_kib"]["median"] / 1024,
                deterministic="yes" if row["deterministic_text"] else "no",
            )
        )
    lines += [
        "",
        "## Corpus",
        "",
        "| Frontend | Median RTF | Feature RTF | Weighted WER | S/D/I |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["corpus"]:
        lines.append(
            f"| {row['frontend']} | {row['pipeline_rtf']['median']:.3f} | "
            f"{row['feature_extraction_rtf']['median']:.3f} | "
            f"{row['word_error_percent']['median']:.3f}% | "
            f"{row['substitutions']['median']:.0f}/"
            f"{row['deletions']['median']:.0f}/"
            f"{row['insertions']['median']:.0f} |"
        )
    atomic_text(output / "summary.md", "\n".join(lines) + "\n")

    transcript_lines = [
        "# Raw benchmark transcripts",
        "",
        "Text below is unedited device output. WER uses the pinned FP32 baseline.",
        "",
    ]
    for item in sorted(runs, key=lambda value: (value["sample_id"], value["frontend"], value["repeat"])):
        atomic_text(
            output
            / "raw"
            / item["sample_id"]
            / f"{item['frontend']}-r{item['repeat']}.txt",
            item["raw_report"]["result"]["text"],
        )
        transcript_lines += [
            f"## {item['sample_id']} — {item['frontend']} — repeat {item['repeat']}",
            "",
            f"WER: {item['word_error']['word_error_percent']:.6f}% "
            f"({item['word_error']['word_edits']}/{item['word_error']['reference_words']})",
            "",
            textwrap.indent(item["raw_report"]["result"]["text"], "    "),
            "",
        ]
    atomic_text(output / "transcripts.md", "\n".join(transcript_lines))


def execute(args: argparse.Namespace) -> int:
    manifest = validate_samples()
    if args.command == "plan":
        printable = {
            "target_package": PACKAGE,
            "test_package": TEST_PACKAGE,
            "real_package": REAL_PACKAGE,
            "frontends": FRONTENDS,
            "repeats": REPEATS,
            "timed_runs": len(SAMPLES) * len(FRONTENDS) * REPEATS,
            "parity_runs": len(SAMPLES),
            "samples": [{key: value for key, value in manifest[item.id].items() if key != "baseline"} for item in SAMPLES],
        }
        print(json.dumps(printable, indent=2))
        return 0

    adb_path = args.adb
    serial = resolve_serial(adb_path, args.serial)
    device = validate_device(adb_path, serial)
    assert_real_package_idle(adb_path, serial)
    package_before = package_snapshot(adb_path, serial, REAL_PACKAGE)
    if not package_before.get("installed"):
        raise BenchmarkError(
            "The real VoiceMemo package is not installed; path/update-time proof is required."
        )
    app_apk, test_apk = build_apks()
    install_apks(adb_path, serial, app_apk, test_apk)
    assert_package_unchanged(
        package_before,
        package_snapshot(adb_path, serial, REAL_PACKAGE),
    )

    suite_id = args.suite_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    if not SAFE_RUN_ID.fullmatch(suite_id):
        raise BenchmarkError("suite-id must contain only letters, numbers, and hyphens.")
    output = ROOT / "build/reports/skainet-audio-benchmark" / suite_id
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "manifest.json",
        {
            "suite_id": suite_id,
            "skainet_audio_version": SKAINET_VERSION,
            "device": device,
            "real_package_before": package_before,
            "samples": [{key: value for key, value in manifest[item.id].items() if key != "baseline"} for item in SAMPLES],
        },
    )

    parity_results: dict[str, dict] = {}
    for sample in SAMPLES:
        audio_name = stage_audio(adb_path, serial, sample)
        staged_relative = f"files/{REPORT_DIRECTORY}/{audio_name}"
        try:
            for repeat in range(1, REPEATS + 1):
                order = FRONTENDS if repeat % 2 == 1 else tuple(reversed(FRONTENDS))
                for frontend in order:
                    destination = output / "raw" / sample.id / f"{frontend}-r{repeat}.json"
                    if args.resume and run_file_valid(destination, sample, frontend, repeat):
                        log(f"Resume: keeping {destination.relative_to(output)}")
                        continue
                    thermal_before = wait_for_thermal(
                        adb_path,
                        serial,
                        args.thermal_timeout_minutes * 60,
                    )
                    run_id = f"{sample.id}-{frontend}-r{repeat}"
                    raw, memory_samples, _ = execute_instrumentation(
                        adb_path,
                        serial,
                        run_id,
                        frontend,
                        audio_name,
                        sample.language,
                        args.run_timeout_minutes * 60,
                    )
                    result = raw.get("result")
                    if not isinstance(result, dict) or result.get("audio_sample_count") != manifest[sample.id]["sample_count"]:
                        raise BenchmarkError(f"Invalid timed result for {run_id}.")
                    record = {
                        "sample_id": sample.id,
                        "frontend": frontend,
                        "repeat": repeat,
                        "audio_sha256": sample.audio_sha256,
                        "baseline_sha256": sample.baseline_sha256,
                        "thermal_before": thermal_before,
                        "thermal_after": thermal_snapshot(adb_path, serial),
                        "peak_pss_kib": max((item["pss_kib"] for item in memory_samples), default=int(result["memory_after"]["total_pss_kib"])),
                        "memory_samples": memory_samples,
                        "word_error": word_error(manifest[sample.id]["baseline"]["text"], result["text"]),
                        "raw_report": raw,
                    }
                    atomic_json(destination, record)
                    atomic_text(destination.with_suffix(".txt"), result["text"])

            parity_path = output / "parity" / f"{sample.id}.json"
            if args.resume and parity_file_valid(parity_path, sample):
                parity_value = json.loads(parity_path.read_text(encoding="utf-8"))
                parity_results[sample.id] = parity_value
                continue
            thermal_before = wait_for_thermal(
                adb_path,
                serial,
                args.thermal_timeout_minutes * 60,
            )
            raw, memory_samples, _ = execute_instrumentation(
                adb_path,
                serial,
                f"{sample.id}-parity",
                "parity",
                audio_name,
                sample.language,
                args.run_timeout_minutes * 60,
            )
            parity_value = {
                "sample_id": sample.id,
                "audio_sha256": sample.audio_sha256,
                "thermal_before": thermal_before,
                "thermal_after": thermal_snapshot(adb_path, serial),
                "peak_pss_kib": max((item["pss_kib"] for item in memory_samples), default=0),
                "raw_report": raw,
            }
            atomic_json(parity_path, parity_value)
            parity_results[sample.id] = parity_value
        finally:
            safe_remove(adb_path, serial, staged_relative)

    package_after = package_snapshot(adb_path, serial, REAL_PACKAGE)
    assert_package_unchanged(package_before, package_after)
    runs = load_runs(output)
    summary = summarize(
        suite_id,
        manifest,
        runs,
        parity_results,
        device,
        package_before,
        package_after,
    )
    write_outputs(output, summary, runs)
    log(f"Benchmark complete: {output}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Validate and print the 35-run benchmark plan.")
    run_parser = subparsers.add_parser("run", help="Build and run the isolated Pixel benchmark.")
    run_parser.add_argument("--adb", default="adb")
    run_parser.add_argument("--serial")
    run_parser.add_argument("--suite-id")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--thermal-timeout-minutes", type=int, default=30)
    run_parser.add_argument("--run-timeout-minutes", type=int, default=90)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        return execute(args)
    except (BenchmarkError, OSError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
