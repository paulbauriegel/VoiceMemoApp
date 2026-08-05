#!/usr/bin/env python3
"""Sustain LiteRT OpenCL transcription load while sampling Android thermals.

The runner reuses the installed instrumentation APK and staged INT8 model. It
stages only the benchmark audio, never installs/uninstalls the app, and aborts
the instrumentation process immediately when global CRITICAL is observed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "scripts/run-whisper-benchmark-suite.py"
SAMPLER_PATH = ROOT / "scripts/sample-android-thermal.py"
PACKAGE = "com.example.voicememo"
MODEL_NAME = "whisper-large-v3-turbo-int8.tflite"
MODEL_SHA256 = "8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418"
MODEL_RELATIVE = f"files/gpu-models/{MODEL_NAME}"
AUDIO_PATH = ROOT / "benchmark-inputs/whisper-suite/reference.wav"
AUDIO_NAME = "reference-170s.wav"
AUDIO_SHA256 = "138fc04d96018548265f9d729f20913033cfb902cfc590da5f5559c68bae6ce6"
AUDIO_SAMPLES = 2_729_598
CRITICAL_STATUS = 4


class GpuThermalError(RuntimeError):
    pass


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GpuThermalError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sensor_value(sample: dict[str, Any], name: str) -> float:
    for item in sample.get("temperatures", []):
        if item.get("name") == name:
            return float(item["value_celsius"])
    raise GpuThermalError(f"Thermal sample is missing {name}")


def sensor_status(sample: dict[str, Any], name: str) -> int:
    for item in sample.get("temperatures", []):
        if item.get("name") == name:
            return int(item["status"])
    raise GpuThermalError(f"Thermal sample is missing {name}")


def start_sampler(args, output: pathlib.Path, run_id: str) -> subprocess.Popen[str]:
    command = [
        str(SAMPLER_PATH),
        "--adb",
        args.adb,
        "--serial",
        args.serial,
        "--output",
        str(output),
        "--interval-seconds",
        "1",
        "--label",
        run_id,
        "--metadata",
        "protocol=gpu-time-to-critical",
        "--metadata",
        f"model_sha256={MODEL_SHA256}",
        "--metadata",
        f"audio_sha256={AUDIO_SHA256}",
        "--metadata",
        "backend=litert-opencl-mali",
    ]
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, text=True)
    time.sleep(5)
    if process.poll() is not None:
        raise GpuThermalError(f"Thermal sampler exited early: {process.returncode}")
    return process


def stop_sampler(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        code = process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        process.terminate()
        raise GpuThermalError("Thermal sampler did not stop") from error
    if code != 0:
        raise GpuThermalError(f"Thermal sampler failed: {code}")


def drain_samples(path: pathlib.Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        while line := stream.readline():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                break
            if value.get("kind") == "sample":
                samples.append(value)
        return samples, stream.tell()


def abort_instrumentation(suite, args, process: subprocess.Popen[str]) -> None:
    suite.run(
        suite.adb(args.adb, args.serial, "shell", "am", "force-stop", PACKAGE),
        check=False,
    )
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def fetch_report(suite, args, run_id: str) -> dict[str, Any]:
    text = suite.run(
        suite.adb(
            args.adb,
            args.serial,
            "exec-out",
            "run-as",
            PACKAGE,
            "cat",
            f"cache/gpu-wer/whole-{run_id}.json",
        ),
        capture=True,
    ).stdout
    return json.loads(text)


def assert_mali_opencl_coverage(report: dict[str, Any]) -> None:
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise GpuThermalError("GPU report has no candidate")
    coverage = candidate.get("coverage")
    if not isinstance(coverage, dict):
        raise GpuThermalError("GPU report has no coverage")
    encoder = str(coverage.get("encoder", ""))
    decoder = str(coverage.get("decoder", ""))
    if not encoder.startswith("GPU_OPENCL") or not decoder.startswith("GPU_OPENCL"):
        raise GpuThermalError(f"Neural graph did not use OpenCL GPU: {coverage}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-units", type=int, default=4)
    parser.add_argument("--maximum-heating-seconds", type=int, default=1200)
    args = parser.parse_args()

    if args.output.exists():
        raise GpuThermalError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    suite = load_module("gpu_thermal_suite", SUITE_PATH)
    configuration = suite.Configuration(
        id="litert-int8-opencl-mali",
        label="LiteRT INT8 model via OpenCL GPU",
        runtime="litert",
        model_path="benchmark-inputs/whisper-suite/whisper-large-v3-turbo-int8.tflite",
        model_bytes=1_094_571_944,
        model_sha256=MODEL_SHA256,
        expected_mode="gpu-encoder-gpu-decoder-no-source-quant",
        instrumentation={"candidateBackend": "gpu", "gpuPrecision": "fp32"},
    )

    suite.assert_device_idle(args.adb, args.serial)
    device_model_hash = suite.run(
        suite.adb(
            args.adb,
            args.serial,
            "shell",
            "run-as",
            PACKAGE,
            "sha256sum",
            MODEL_RELATIVE,
        ),
        capture=True,
    ).stdout.split()[0]
    if device_model_hash != MODEL_SHA256:
        raise GpuThermalError("Staged model hash mismatch")
    suite.stage_private_file(
        args.adb,
        args.serial,
        AUDIO_PATH,
        f"files/gpu-audio/{AUDIO_NAME}",
        AUDIO_SHA256,
    )

    thermal_path = args.output / "thermal.ndjson"
    sampler = start_sampler(args, thermal_path, args.run_id)
    campaign_started = time.monotonic()
    sample_offset = 0
    last_sample: dict[str, Any] | None = None
    highest_status = 0
    stop_reason = "maximum-units"
    units: list[dict[str, Any]] = []
    critical_observed = False
    time_limit_observed = False

    try:
        for unit in range(1, args.max_units + 1):
            elapsed = time.monotonic() - campaign_started
            if elapsed >= args.maximum_heating_seconds:
                stop_reason = "maximum-heating-seconds"
                break
            run_id = f"{args.run_id}-u{unit:02d}"
            arguments = suite.instrumentation_arguments(
                configuration=configuration,
                model_name=MODEL_NAME,
                audio_name=AUDIO_NAME,
                run_id=run_id,
                requested_language="de",
                timestamp_tokens_enabled=False,
            )
            command = suite.adb(
                args.adb,
                args.serial,
                "shell",
                "am",
                "instrument",
                "-w",
                "-r",
            )
            for key, value in arguments.items():
                command.extend(("-e", key, value))
            command.append(f"{suite.TEST_PACKAGE}/{suite.TEST_RUNNER}")

            log_path = args.output / f"unit-{unit:02d}-instrumentation.log"
            unit_started = time.monotonic()
            print("+ " + " ".join(command), flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
                next_update = 0.0
                while process.poll() is None:
                    time.sleep(1)
                    new_samples, sample_offset = drain_samples(thermal_path, sample_offset)
                    for sample in new_samples:
                        last_sample = sample
                        highest_status = max(highest_status, int(sample["thermal_status"]))
                        skin_status = sensor_status(sample, "VIRTUAL-SKIN")
                        if int(sample["thermal_status"]) >= CRITICAL_STATUS or skin_status >= CRITICAL_STATUS:
                            critical_observed = True
                            stop_reason = "critical"
                            abort_instrumentation(suite, args, process)
                            break
                    elapsed = time.monotonic() - campaign_started
                    if elapsed >= args.maximum_heating_seconds and not critical_observed:
                        time_limit_observed = True
                        stop_reason = "maximum-heating-seconds"
                        abort_instrumentation(suite, args, process)
                        break
                    if last_sample is not None and elapsed >= next_update:
                        print(
                            "THERMAL "
                            f"elapsed={elapsed:.1f}s unit={unit} "
                            f"global={last_sample['thermal_status']} "
                            f"skin={sensor_value(last_sample, 'VIRTUAL-SKIN'):.2f}C "
                            f"g3d={sensor_value(last_sample, 'G3D'):.1f}C "
                            f"battery={sensor_value(last_sample, 'battery'):.1f}C",
                            flush=True,
                        )
                        next_update = elapsed + 30
                    if critical_observed:
                        break

            if critical_observed or time_limit_observed:
                units.append(
                    {
                        "unit": unit,
                        "run_id": run_id,
                        "status": f"aborted-at-{stop_reason}",
                        "elapsed_seconds": time.monotonic() - unit_started,
                    }
                )
                break

            instrumentation_text = log_path.read_text(encoding="utf-8")
            if process.returncode != 0 or "OK (1 test)" not in instrumentation_text:
                raise GpuThermalError(f"Instrumentation failed for {run_id}; see {log_path}")
            report = fetch_report(suite, args, run_id)
            suite.validate_report(report, configuration, AUDIO_SAMPLES, "de", False)
            assert_mali_opencl_coverage(report)
            write_json(args.output / f"unit-{unit:02d}-report.json", report)
            new_samples, sample_offset = drain_samples(thermal_path, sample_offset)
            for sample in new_samples:
                last_sample = sample
                highest_status = max(highest_status, int(sample["thermal_status"]))
                if (
                    int(sample["thermal_status"]) >= CRITICAL_STATUS
                    or sensor_status(sample, "VIRTUAL-SKIN") >= CRITICAL_STATUS
                ):
                    critical_observed = True
                    stop_reason = "critical-after-completed-unit"
            candidate = report["candidate"]
            units.append(
                {
                    "unit": unit,
                    "run_id": run_id,
                    "status": "completed",
                    "total_wall_millis": candidate["total_wall_millis"],
                    "inference_wall_millis": candidate["inference_wall_millis"],
                    "coverage": candidate["coverage"],
                    "thermal_status_after_inference": candidate["memory_after_inference"][
                        "thermal_status"
                    ],
                }
            )
            write_json(
                args.output / "manifest.json",
                {
                    "run_id": args.run_id,
                    "started_at_utc": utc_now(),
                    "model_sha256": MODEL_SHA256,
                    "audio_sha256": AUDIO_SHA256,
                    "backend": "LiteRT OpenCL on Pixel 7 Pro Mali GPU",
                    "critical_abort_status": CRITICAL_STATUS,
                    "highest_global_status": highest_status,
                    "stop_reason": "running",
                    "units": units,
                },
            )
            if critical_observed:
                break
    finally:
        stop_sampler(sampler)
        suite.run(
            suite.adb(
                args.adb,
                args.serial,
                "shell",
                "run-as",
                PACKAGE,
                "rm",
                "-f",
                f"files/gpu-audio/{AUDIO_NAME}",
            ),
            check=False,
        )

    write_json(
        args.output / "manifest.json",
        {
            "run_id": args.run_id,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - campaign_started,
            "model_sha256": MODEL_SHA256,
            "audio_sha256": AUDIO_SHA256,
            "backend": "LiteRT OpenCL on Pixel 7 Pro Mali GPU",
            "critical_abort_status": CRITICAL_STATUS,
            "highest_global_status": highest_status,
            "stop_reason": stop_reason,
            "units": units,
        },
    )
    print(f"GPU thermal manifest: {args.output / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
