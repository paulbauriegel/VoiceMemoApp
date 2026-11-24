#!/usr/bin/env python3
"""Run pyannote segmentation-3.0 on the phone (LiteRT and ONNX Runtime) and compare.

Everything happens in the isolated package com.example.voicememo.benchmark; the real
VoiceMemo package is snapshotted before and after and must be unchanged. Nothing is
uninstalled or cleared. Audio is the five public benchmark samples only.

Needs numpy and onnxruntime for the host reference, e.g.:
  build/speaker-spike/venv/bin/python scripts/run-pyannote-segmentation-benchmark.py

Models come from scripts/export-pyannote-segmentation-litert.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("skainet_runner", ROOT / "scripts/run-skainet-audio-benchmark.py")
base = importlib.util.module_from_spec(_spec)
sys.modules["skainet_runner"] = base
_spec.loader.exec_module(base)

PACKAGE, TEST_PACKAGE, REAL_PACKAGE = base.PACKAGE, base.TEST_PACKAGE, base.REAL_PACKAGE
TEST_CLASS = "com.example.voicememo.transcription.PyannoteSegmentationBenchmarkTest#runBenchmark"
DIRECTORY = "pyannote-seg-benchmark"
MODEL_DIR = ROOT / "build/speaker-spike/models/seg"
MODELS = {
    "unrolled": MODEL_DIR / "pyannote-segmentation-3-0-litert/segmentation.tflite",
    "fused": MODEL_DIR / "pyannote-segmentation-3-0-litert/segmentation-fused.tflite",
    "onnx": MODEL_DIR / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
}
STAGED = {"unrolled": "segmentation.tflite", "fused": "segmentation-fused.tflite", "onnx": "segmentation.onnx"}
SR, WINDOW, FRAMES, CLASSES = 16000, 160000, 589, 7
HOP_MILLIS = 2500
TIMING_SAMPLE = "controlled-german-170s"
# (label, model, engine): the three candidates, then two diagnostics.
CANDIDATES = [
    ("onnxruntime", "onnx", "onnx"),
    ("litert-unrolled", "unrolled", "litert-compiled"),
    ("litert-fused", "fused", "litert-interpreter"),
]
DIAGNOSTICS = [
    ("litert-fused-compiledmodel", "fused", "litert-compiled"),
    ("litert-fused-no-state-reset", "fused", "litert-interpreter-keep-state"),
    ("litert-unrolled-interpreter", "unrolled", "litert-interpreter"),
]
GPU_RUNS = [
    ("litert-unrolled-gpu-fp16", "unrolled", "litert-gpu-fp16"),
    ("litert-unrolled-gpu-fp32", "unrolled", "litert-gpu-fp32"),
    ("litert-unrolled-gpu-fp16-cpu-fallback", "unrolled", "litert-gpu-fp16-cpu-fallback"),
    ("litert-fused-gpu-fp16", "fused", "litert-gpu-fp16"),
]
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_ABS_LIMIT, MIN_ARGMAX = 1e-2, 0.999


def stage(adb_path: str, serial: str, source: Path, name: str) -> None:
    if not SAFE_NAME.fullmatch(name):
        raise base.BenchmarkError(f"Unsafe staged name: {name}")
    relative = f"files/{DIRECTORY}/{name}"
    expected = base.sha256(source)
    existing = base.run(base.adb(adb_path, serial, "shell", "run-as", PACKAGE, "sha256sum", relative),
                        capture=True, check=False)
    if existing.returncode == 0 and existing.stdout.split()[:1] == [expected]:
        return
    base.run(base.adb(adb_path, serial, "shell", "run-as", PACKAGE, "mkdir", "-p", f"files/{DIRECTORY}"))
    command = base.adb(adb_path, serial, "shell", "run-as", PACKAGE, "tee", relative)
    base.log("+ " + " ".join(command) + f" < {source}")
    with source.open("rb") as stream:
        if subprocess.run(command, stdin=stream, stdout=subprocess.DEVNULL).returncode != 0:
            raise base.BenchmarkError(f"Could not stage {source}.")
    actual = base.run(base.adb(adb_path, serial, "shell", "run-as", PACKAGE, "sha256sum", relative),
                      capture=True).stdout.split()[0]
    if actual != expected:
        raise base.BenchmarkError(f"Staged hash differs for {name}.")


def remove(adb_path: str, serial: str, relative: str) -> None:
    if not re.fullmatch(rf"(?:files|cache)/{DIRECTORY}/[A-Za-z0-9._-]+", relative):
        raise base.BenchmarkError(f"Unsafe cleanup path: {relative}")
    base.run(base.adb(adb_path, serial, "shell", "run-as", PACKAGE, "rm", "-f", relative), check=False)


def pull(adb_path: str, serial: str, relative: str) -> bytes:
    completed = subprocess.run(base.adb(adb_path, serial, "exec-out", "run-as", PACKAGE, "cat", relative),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise base.BenchmarkError(f"Could not read {relative} from the device.")
    remove(adb_path, serial, relative)
    return completed.stdout


def run_on_device(adb_path, serial, run_id, engine, model_name, audio_name, threads, timeout):
    base.run(base.adb(adb_path, serial, "shell", "am", "force-stop", PACKAGE))
    command = base.adb(adb_path, serial, "shell", "am", "instrument", "-w", "-r")
    for key, value in {"class": TEST_CLASS, "runId": run_id, "engine": engine, "modelName": model_name,
                       "audioName": audio_name, "threads": str(threads), "hopMillis": str(HOP_MILLIS)}.items():
        command.extend(("-e", key, value))
    command.append(f"{TEST_PACKAGE}/{base.RUNNER}")
    base.log("+ " + " ".join(command))
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    try:
        report = json.loads(pull(adb_path, serial, f"cache/{DIRECTORY}/{run_id}.json"))
    except (base.BenchmarkError, json.JSONDecodeError) as error:
        print(completed.stdout)
        raise base.BenchmarkError(f"Run {run_id} wrote no report.") from error
    if completed.returncode != 0 or "OK (1 test)" not in completed.stdout or report.get("status") != "completed":
        print(completed.stdout)
        raise base.BenchmarkError(f"Run {run_id} failed: {report.get('error_type')}: {report.get('error_message')}")
    if report.get("package_name") != PACKAGE:
        raise base.BenchmarkError("Report came from a non-isolated package.")
    output = np.frombuffer(pull(adb_path, serial, f"cache/{DIRECTORY}/{run_id}.f32"), "<f4")
    return report, output.reshape(-1, FRAMES, CLASSES)


def load_audio(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == SR and handle.getnchannels() == 1 and handle.getsampwidth() == 2
        return np.frombuffer(handle.readframes(handle.getnframes()), "<i2").astype(np.float32) / 32768


def host_reference(path: Path) -> np.ndarray:
    import onnxruntime as ort
    audio = load_audio(path)
    session = ort.InferenceSession(str(MODELS["onnx"]), providers=["CPUExecutionProvider"])
    hop = HOP_MILLIS * SR // 1000
    outputs = []
    for start in range(0, max(1, len(audio) - WINDOW + hop), hop):
        chunk = np.zeros(WINDOW, np.float32)
        piece = audio[start:start + WINDOW]
        chunk[:len(piece)] = piece
        outputs.append(session.run(None, {"x": chunk[None, None, :]})[0][0])
    return np.stack(outputs)


def summarise(report: dict, output: np.ndarray, reference: np.ndarray) -> dict:
    result = report["result"]
    chunk_ms = result["chunk_ms"]
    steady = chunk_ms[1:] or chunk_ms
    median = statistics.median(steady)
    same_shape = output.shape == reference.shape
    difference = np.abs(output - reference) if same_shape else None
    argmax = float((output.argmax(-1) == reference.argmax(-1)).mean()) if same_shape else 0.0
    max_abs = float(difference.max()) if same_shape else float("inf")
    return {
        "chunks": result["chunks"], "audio_seconds": round(result["audio_seconds"], 1),
        "model_load_ms": round(result["model_load_ms"], 1), "first_chunk_ms": round(chunk_ms[0], 1),
        "median_chunk_ms": round(median, 1),
        "p90_chunk_ms": round(float(np.percentile(steady, 90)), 1),
        "real_time_factor": round(result["real_time_factor"], 4),
        "cpu_ms_per_wall_ms": round(result["inference_cpu_ms"] / result["inference_wall_ms"], 2),
        "minutes_for_60min_recording": {"hop_2.5s": round(1440 * median / 60000, 2), "hop_5s": round(720 * median / 60000, 2)},
        "max_abs_vs_host": max_abs, "mean_abs_vs_host": float(difference.mean()) if same_shape else None,
        "argmax_agreement_vs_host": argmax,
        "parity_passed": bool(same_shape and max_abs <= MAX_ABS_LIMIT and argmax >= MIN_ARGMAX),
        "native_heap_after_run_mib": round(result["memory_after_run"]["native_heap_allocated_bytes"] / 2**20, 1),
        "total_pss_after_run_mib": round(result["memory_after_run"]["total_pss_kib"] / 1024, 1),
        "thermal_status": [result["thermal_status_before"], result["thermal_status_after"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--skip-build", action="store_true", help="reuse the APKs from the last isolated build")
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--gpu-only", action="store_true", help="Run only the LiteRT GPU attempts.")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    for name, path in MODELS.items():
        if not path.is_file():
            raise base.BenchmarkError(f"Missing model {name}: {path}")
    base.validate_samples()
    serial = base.resolve_serial(args.adb, args.serial)
    threads = [int(value) for value in args.threads.split(",")]

    real_before = base.package_snapshot(args.adb, serial, REAL_PACKAGE)
    base.assert_real_package_idle(args.adb, serial)
    if args.skip_build:
        app = ROOT / "app/build/outputs/apk/debug/app-debug.apk"
        test = ROOT / "app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
        aapt = base.find_aapt()
        base.assert_apk_package(aapt, app, PACKAGE)
        base.assert_apk_package(aapt, test, TEST_PACKAGE)
    else:
        app, test = base.build_apks()
    base.install_apks(args.adb, serial, app, test)

    suite_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "build/reports/pyannote-segmentation-benchmark" / suite_id
    out.mkdir(parents=True)
    samples = {sample.id: sample for sample in base.SAMPLES}
    for key in MODELS:
        stage(args.adb, serial, MODELS[key], STAGED[key])
    for sample in samples.values():
        stage(args.adb, serial, sample.audio, f"{sample.id}.wav")
    references = {sample_id: host_reference(sample.audio) for sample_id, sample in samples.items()}

    rows, counter = [], 0

    def one(label, model, engine, sample_id, thread_count, kind):
        nonlocal counter
        counter += 1
        base.assert_real_package_idle(args.adb, serial)
        run_id = f"seg-{suite_id}-{counter:03d}"
        report, output = run_on_device(args.adb, serial, run_id, engine, STAGED[model],
                                       f"{sample_id}.wav", thread_count, args.timeout)
        row = {"label": label, "kind": kind, "model": model, "engine": engine, "sample": sample_id,
               "threads": thread_count} | summarise(report, output, references[sample_id])
        rows.append(row)
        base.atomic_json(out / f"{run_id}.json", report | {"summary": row})
        base.log(f"   {label} {sample_id} t{thread_count}: {row['median_chunk_ms']} ms/chunk, "
                 f"arg-max {row['argmax_agreement_vs_host']:.5f}, max abs {row['max_abs_vs_host']:.2e}")
        return report

    try:
        device = None
        # Timing: alternate the candidates per thread count to spread thermal drift.
        for label, model, engine in GPU_RUNS if args.gpu_only else []:
            try:
                device = one(label, model, engine, TIMING_SAMPLE, threads[-1], "diagnostic")["device"]
            except (base.BenchmarkError, subprocess.TimeoutExpired) as error:
                base.log(f"   gpu {label} failed: {error}")
        candidates = [] if args.gpu_only else CANDIDATES
        for thread_count in threads:
            for label, model, engine in candidates:
                device = one(label, model, engine, TIMING_SAMPLE, thread_count, "timing")["device"]
        # Parity on the other samples at the largest thread count.
        for sample_id in samples:
            if sample_id == TIMING_SAMPLE:
                continue
            for label, model, engine in candidates:
                one(label, model, engine, sample_id, threads[-1], "parity")
        if not args.no_diagnostics and not args.gpu_only:
            for label, model, engine in DIAGNOSTICS:
                try:
                    one(label, model, engine, TIMING_SAMPLE, threads[-1], "diagnostic")
                except base.BenchmarkError as error:  # informational only, e.g. unsupported model/API pair
                    base.log(f"   diagnostic {label} failed: {error}")
    finally:
        real_after = base.package_snapshot(args.adb, serial, REAL_PACKAGE)
        base.atomic_json(out / "summary.json", {
            "suite_id": suite_id, "device": device, "hop_millis": HOP_MILLIS,
            "models": {key: {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                             "sha256": base.sha256(path)} for key, path in MODELS.items()},
            "real_package_before": real_before, "real_package_after": real_after,
            "real_package_unchanged": real_before == real_after, "runs": rows,
        })
    base.assert_package_unchanged(real_before, real_after)

    print(f"\n{'configuration':30s} {'thr':>3s} {'load ms':>8s} {'ms/chunk':>9s} {'p90':>7s} {'60 min @2.5s':>13s} {'arg-max':>8s} {'max abs':>9s}")
    for row in rows:
        if row["kind"] == "parity":
            continue
        print(f"{row['label']:30s} {row['threads']:3d} {row['model_load_ms']:8.0f} {row['median_chunk_ms']:9.1f} "
              f"{row['p90_chunk_ms']:7.1f} {row['minutes_for_60min_recording']['hop_2.5s']:10.2f} min "
              f"{row['argmax_agreement_vs_host']:8.5f} {row['max_abs_vs_host']:9.2e}")
    for label, _, _ in [] if args.gpu_only else CANDIDATES:
        parity = [row for row in rows if row["label"] == label and row["kind"] != "diagnostic"]
        print(f"parity {label}: {sum(row['parity_passed'] for row in parity)}/{len(parity)} runs pass "
              f"(worst arg-max {min(row['argmax_agreement_vs_host'] for row in parity):.5f}, "
              f"worst max abs {max(row['max_abs_vs_host'] for row in parity):.2e})")
    print(f"real VoiceMemo package unchanged: {real_before == real_after}")
    print(f"report: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except base.BenchmarkError as error:
        sys.exit(f"error: {error}")
