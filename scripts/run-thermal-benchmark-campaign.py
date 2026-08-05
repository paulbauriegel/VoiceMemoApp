#!/usr/bin/env python3
"""Run the VoiceMemo audio-impact and throttling thermal protocols.

This host-side runner reuses the installed instrumentation APK and a verified,
already-staged LiteRT INT8 model. It never installs, uninstalls, clears app
data, or touches recordings.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import pathlib
import random
import signal
import statistics
import subprocess
import sys
import time
import wave
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "scripts/run-whisper-benchmark-suite.py"
SAMPLER = ROOT / "scripts/sample-android-thermal.py"
PACKAGE = "com.example.voicememo"
MODEL_NAME = "whisper-large-v3-turbo-int8.tflite"
MODEL_RELATIVE = f"files/gpu-models/{MODEL_NAME}"
MODEL_SHA256 = "8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418"
CONFIGURATION_ID = "litert-int8-cpu-4x3"
UNSAFE_STATUS = 4


class CampaignError(RuntimeError):
    pass


def load_suite():
    spec = importlib.util.spec_from_file_location("thermal_campaign_suite", SUITE_PATH)
    if spec is None or spec.loader is None:
        raise CampaignError("Could not load benchmark-suite helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def audio_samples(path: pathlib.Path) -> int:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != 16_000:
            raise CampaignError(f"Expected mono 16 kHz WAV: {path}")
        return audio.getnframes()


def live_sensor(snapshot: dict[str, Any], name: str) -> float:
    for item in snapshot["temperatures"]:
        if item["name"] == name:
            return float(item["value_celsius"])
    raise CampaignError(f"Live HAL sensor is missing: {name}")


def thermal_snapshot(suite, adb_path: str, serial: str) -> dict[str, Any]:
    thermal_text = suite.run(
        suite.adb(adb_path, serial, "shell", "dumpsys", "thermalservice"),
        capture=True,
    ).stdout
    status, temperatures = parse_sampler().parse_thermalservice(thermal_text)
    return {
        "captured_at_utc": utc_now(),
        "thermal_status": status,
        "temperatures": [item.as_json() for item in temperatures],
    }


def parse_sampler():
    spec = importlib.util.spec_from_file_location("thermal_sampler", SAMPLER)
    if spec is None or spec.loader is None:
        raise CampaignError("Could not load thermal sampler")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def baseline_skin(path: pathlib.Path) -> tuple[float, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    points: list[tuple[float, float, int]] = []
    for row in rows:
        if row.get("kind") != "sample":
            continue
        temperatures = {item["name"]: item["value_celsius"] for item in row["temperatures"]}
        if "VIRTUAL-SKIN" in temperatures:
            points.append(
                (row["elapsed_seconds"], temperatures["VIRTUAL-SKIN"], row["thermal_status"])
            )
    if not points or points[-1][0] < 170:
        raise CampaignError("Stable baseline must contain at least 170 seconds")
    if max(item[2] for item in points) != 0:
        raise CampaignError("Stable baseline did not remain at thermal status NONE")
    final = [item for item in points if item[0] >= points[-1][0] - 120]
    mean_x = statistics.mean(item[0] for item in final)
    mean_y = statistics.mean(item[1] for item in final)
    denominator = sum((item[0] - mean_x) ** 2 for item in final)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y, _ in final)
        / denominator
        * 60
    )
    last_ten = [item[1] for item in points if item[0] >= points[-1][0] - 10]
    return statistics.median(last_ten), slope


def wait_until_ready(
    suite,
    adb_path: str,
    serial: str,
    baseline_celsius: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        suite.assert_device_idle(adb_path, serial)
        snapshot = thermal_snapshot(suite, adb_path, serial)
        skin = live_sensor(snapshot, "VIRTUAL-SKIN")
        if snapshot["thermal_status"] == 0 and skin <= baseline_celsius + 0.5:
            return snapshot
        if snapshot["thermal_status"] >= UNSAFE_STATUS:
            raise CampaignError(f"Unsafe thermal status while waiting: {snapshot}")
        if time.monotonic() >= deadline:
            raise CampaignError(
                f"Phone did not recover to NONE and <= {baseline_celsius + 0.5:.2f} C skin"
            )
        print(
            f"Waiting for recovery: status={snapshot['thermal_status']} skin={skin:.2f} C",
            flush=True,
        )
        time.sleep(15)


def start_sampler(
    adb_path: str,
    serial: str,
    output: pathlib.Path,
    label: str,
    metadata: dict[str, str],
) -> tuple[subprocess.Popen[str], float]:
    command = [
        str(SAMPLER),
        "--adb",
        adb_path,
        "--serial",
        serial,
        "--output",
        str(output),
        "--interval-seconds",
        "1",
        "--label",
        label,
    ]
    for key, value in metadata.items():
        command.extend(("--metadata", f"{key}={value}"))
    print("+ " + " ".join(command), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, text=True)
    time.sleep(10)
    if process.poll() is not None:
        raise CampaignError(f"Thermal sampler exited early with {process.returncode}")
    return process, started


def stop_sampler(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        return_code = process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        process.terminate()
        raise CampaignError("Thermal sampler did not stop cleanly") from error
    if return_code != 0:
        raise CampaignError(f"Thermal sampler failed with exit code {return_code}")


def run_unit(
    suite,
    configuration,
    adb_path: str,
    serial: str,
    audio_name: str,
    samples: int,
    run_id: str,
) -> dict[str, Any]:
    report = suite.run_instrumentation(
        adb_path=adb_path,
        serial=serial,
        configuration=configuration,
        model_name=MODEL_NAME,
        audio_name=audio_name,
        run_id=run_id,
        requested_language="de",
        timestamp_tokens_enabled=False,
    )
    suite.validate_report(report, configuration, samples, "de", False)
    return report


def stage_inputs(suite, adb_path: str, serial: str, audios: list[dict[str, Any]]) -> list[str]:
    device_hash = suite.run(
        suite.adb(adb_path, serial, "shell", "run-as", PACKAGE, "sha256sum", MODEL_RELATIVE),
        capture=True,
    ).stdout.split()[0]
    if device_hash != MODEL_SHA256:
        raise CampaignError("Already-staged LiteRT INT8 model hash mismatch")
    staged: list[str] = []
    for audio in audios:
        relative = f"files/gpu-audio/{audio['device_name']}"
        suite.stage_private_file(adb_path, serial, audio["path"], relative, audio["sha256"])
        staged.append(relative)
    return staged


def run_audio_phase(args, suite, configuration, audios, baseline_celsius: float) -> None:
    output = args.output / "audio-impact"
    if output.exists():
        raise CampaignError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    schedule = []
    labels = [audio["label"] for audio in audios]
    random.Random(args.seed).shuffle(labels)
    for repeat in range(1, args.audio_repeats + 1):
        schedule.extend((label, repeat) for label in labels)
        labels = labels[1:] + labels[:1]
    by_label = {audio["label"]: audio for audio in audios}
    write_json(output / "schedule.json", schedule)

    for index, (label, repeat) in enumerate(schedule, start=1):
        audio = by_label[label]
        wait_until_ready(
            suite, args.adb, args.serial, baseline_celsius, args.recovery_timeout_seconds
        )
        run_id = f"{args.campaign_id}-audio-{index:02d}-{label}-r{repeat}"
        run_output = output / run_id
        run_output.mkdir()
        sampler, sampler_started = start_sampler(
            args.adb,
            args.serial,
            run_output / "thermal.ndjson",
            run_id,
            {
                "protocol": "audio-impact",
                "audio_sha256": audio["sha256"],
                "audio_samples": str(audio["samples"]),
                "configuration": CONFIGURATION_ID,
            },
        )
        manifest: dict[str, Any] = {
            "run_id": run_id,
            "audio": {key: value for key, value in audio.items() if key != "path"},
            "configuration": CONFIGURATION_ID,
            "workload_started_at_utc": utc_now(),
            "workload_start_offset_seconds": time.monotonic() - sampler_started,
        }
        try:
            report = run_unit(
                suite,
                configuration,
                args.adb,
                args.serial,
                audio["device_name"],
                audio["samples"],
                run_id,
            )
            manifest["workload_finished_at_utc"] = utc_now()
            manifest["workload_end_offset_seconds"] = time.monotonic() - sampler_started
            manifest["thermal_after_workload"] = thermal_snapshot(
                suite, args.adb, args.serial
            )
            if manifest["thermal_after_workload"]["thermal_status"] >= UNSAFE_STATUS:
                raise CampaignError("Unsafe thermal status after audio workload")
            write_json(run_output / "report.json", report)
            time.sleep(args.recovery_capture_seconds)
            manifest["status"] = "completed"
        finally:
            stop_sampler(sampler)
            write_json(run_output / "manifest.json", manifest)


def run_throttle_phase(args, suite, configuration, audio, baseline_celsius: float) -> None:
    output = args.output / "throttle"
    if output.exists():
        raise CampaignError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    wait_until_ready(
        suite, args.adb, args.serial, baseline_celsius, args.recovery_timeout_seconds
    )
    sampler, sampler_started = start_sampler(
        args.adb,
        args.serial,
        output / "thermal.ndjson",
        f"{args.campaign_id}-throttle",
        {
            "protocol": "time-to-throttle",
            "audio_sha256": audio["sha256"],
            "configuration": CONFIGURATION_ID,
        },
    )
    manifest: dict[str, Any] = {
        "campaign_id": args.campaign_id,
        "configuration": CONFIGURATION_ID,
        "audio": {key: value for key, value in audio.items() if key != "path"},
        "maximum_heating_seconds": args.maximum_heating_seconds,
        "units": [],
        "started_at_utc": utc_now(),
    }
    heating_started = time.monotonic()
    try:
        unit = 0
        while time.monotonic() - heating_started < args.maximum_heating_seconds:
            unit += 1
            before = thermal_snapshot(suite, args.adb, args.serial)
            if before["thermal_status"] >= UNSAFE_STATUS:
                raise CampaignError("Unsafe thermal status before throttle unit")
            run_id = f"{args.campaign_id}-throttle-u{unit:02d}"
            entry: dict[str, Any] = {
                "unit": unit,
                "run_id": run_id,
                "start_offset_seconds": time.monotonic() - sampler_started,
                "thermal_before": before,
            }
            report = run_unit(
                suite,
                configuration,
                args.adb,
                args.serial,
                audio["device_name"],
                audio["samples"],
                run_id,
            )
            entry["end_offset_seconds"] = time.monotonic() - sampler_started
            entry["thermal_after"] = thermal_snapshot(suite, args.adb, args.serial)
            entry["candidate"] = report["candidate"]
            write_json(output / f"unit-{unit:02d}-report.json", report)
            manifest["units"].append(entry)
            write_json(output / "manifest.json", manifest)
            if entry["thermal_after"]["thermal_status"] >= UNSAFE_STATUS:
                raise CampaignError("Unsafe thermal status after throttle unit")
        manifest["stop_reason"] = "maximum_heating_duration"
    finally:
        manifest["finished_at_utc"] = utc_now()
        stop_sampler(sampler)
        write_json(output / "manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="2B211FDH300F4S")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--stable-baseline", required=True, type=pathlib.Path)
    parser.add_argument("--audio-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--recovery-capture-seconds", type=int, default=120)
    parser.add_argument("--recovery-timeout-seconds", type=int, default=1200)
    parser.add_argument("--maximum-heating-seconds", type=int, default=1800)
    parser.add_argument("--phase", choices=("audio", "throttle", "all"), default="all")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.audio_repeats < 1 or args.maximum_heating_seconds <= 0:
        raise CampaignError("Repeat count and maximum heating time must be positive")

    baseline_celsius, slope = baseline_skin(args.stable_baseline.resolve())
    if not math.isfinite(slope) or abs(slope) > 0.1:
        raise CampaignError(f"Baseline skin slope is not stable: {slope:.3f} C/min")
    suite = load_suite()
    serial = suite.one_device(args.adb, args.serial)
    args.serial = serial
    suite.assert_device_idle(args.adb, serial)
    configuration = next(item for item in suite.CONFIGURATIONS if item.id == CONFIGURATION_ID)
    audio_root = args.output.parent / "audio"
    sources = (
        ("28s", audio_root / "reference-28s.wav", "reference-28s.wav"),
        ("84s", audio_root / "reference-84s.wav", "reference-84s.wav"),
        ("170s", ROOT / "benchmark-inputs/whisper-suite/reference.wav", "reference-170s.wav"),
    )
    audios = []
    for label, path, device_name in sources:
        if not path.is_file():
            raise CampaignError(f"Missing audio input: {path}")
        audios.append(
            {
                "label": label,
                "path": path,
                "device_name": device_name,
                "sha256": suite.sha256(path),
                "samples": audio_samples(path),
            }
        )

    staged: list[str] = []
    try:
        staged = stage_inputs(suite, args.adb, serial, audios)
        write_json(
            args.output / "campaign.json",
            {
                "campaign_id": args.campaign_id,
                "baseline_skin_celsius": baseline_celsius,
                "baseline_skin_slope_celsius_per_minute": slope,
                "configuration": CONFIGURATION_ID,
                "model_sha256": MODEL_SHA256,
                "audios": [{key: value for key, value in item.items() if key != "path"} for item in audios],
                "power_control": "AC powered, battery 100 percent at campaign start",
            },
        )
        if args.phase in ("audio", "all"):
            run_audio_phase(args, suite, configuration, audios, baseline_celsius)
        if args.phase in ("throttle", "all"):
            run_throttle_phase(args, suite, configuration, audios[0], baseline_celsius)
    finally:
        for relative in reversed(staged):
            suite.remove_private_file(args.adb, serial, relative)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CampaignError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Thermal benchmark campaign failed: {error}", file=sys.stderr)
        raise SystemExit(2)
