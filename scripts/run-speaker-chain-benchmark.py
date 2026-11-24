#!/usr/bin/env python3
"""Run the app's speaker-label engines on the phone and compare with the host.

Runs the production PyannoteSegmenter and CamPlusPlusEmbedder (bundled assets, Kotlin
Kaldi fbank, feature padding) in the isolated package com.example.voicememo.benchmark on the five public
benchmark samples. The host reference is ONNX Runtime with kaldi-native-fbank features.
The real VoiceMemo package is snapshotted before and after and must be unchanged;
nothing is uninstalled or cleared.

Needs numpy, onnxruntime and kaldi-native-fbank, e.g.:
  build/speaker-spike/venv/bin/python scripts/run-speaker-chain-benchmark.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pyannote_runner", ROOT / "scripts/run-pyannote-segmentation-benchmark.py")
seg = importlib.util.module_from_spec(_spec)
sys.modules["pyannote_runner"] = seg
_spec.loader.exec_module(seg)
base = seg.base

PACKAGE, TEST_PACKAGE, REAL_PACKAGE = base.PACKAGE, base.TEST_PACKAGE, base.REAL_PACKAGE
TEST_CLASS = "com.example.voicememo.speaker.SpeakerChainBenchmarkTest#runBenchmark"
DIRECTORY = "speaker-chain-benchmark"
ASSETS = ROOT / "app/src/main/generatedSpeakerAssets/speaker"
SEGMENTATION = ASSETS / "pyannote_segmentation_3_0.onnx"
VOICE = ASSETS / "3dspeaker_campplus_zh_en_advanced.onnx"
# The bundled 3D-Speaker model: samples in -1..1, feature frames repeated up to a multiple of 200.
SAMPLE_SCALE, FRAME_PADDING = 1.0, 200
SR, WINDOW, FRAMES, CLASSES = 16000, 160000, 589, 7
SEG_HOP, EMB_WINDOW, EMB_HOP, DIMENSIONS = 40000, 24000, 12000, 256
TIMING_SAMPLE = "controlled-german-170s"
MAX_ABS_LIMIT, MIN_ARGMAX, MIN_COSINE = 1e-2, 0.999, 0.999


def host_reference(path: Path, voice_model: Path = VOICE, sample_scale: float = SAMPLE_SCALE,
                   frame_padding: int = FRAME_PADDING):
    import kaldi_native_fbank as knf
    import onnxruntime as ort
    audio = seg.load_audio(path)
    session = ort.InferenceSession(str(SEGMENTATION), providers=["CPUExecutionProvider"])
    outputs = []
    for start in range(0, max(1, len(audio) - WINDOW + SEG_HOP), SEG_HOP):
        chunk = np.zeros(WINDOW, np.float32)
        piece = audio[start:start + WINDOW]
        chunk[:len(piece)] = piece
        outputs.append(session.run(None, {"x": chunk[None, None, :]})[0][0])
    voice = ort.InferenceSession(str(voice_model), providers=["CPUExecutionProvider"])
    options = knf.FbankOptions()
    options.frame_opts.samp_freq = SR
    options.frame_opts.dither = 0.0
    options.frame_opts.snip_edges = True
    options.mel_opts.num_bins = 80
    vectors = []
    for start in range(0, len(audio) - EMB_WINDOW + 1, EMB_HOP):
        fbank = knf.OnlineFbank(options)
        fbank.accept_waveform(SR, (audio[start:start + EMB_WINDOW] * sample_scale).tolist())
        fbank.input_finished()
        features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        features = (features - features.mean(0, keepdims=True)).astype(np.float32)
        if frame_padding:
            features = np.resize(features, (-(-len(features) // frame_padding) * frame_padding, features.shape[1]))
        vector = voice.run(None, {voice.get_inputs()[0].name: features[None]})[0][0]
        vectors.append(vector / np.linalg.norm(vector))
    return np.stack(outputs), np.stack(vectors) if vectors else np.zeros((0, 0), np.float32)


def remove(adb_path, serial, relative):
    if not relative.startswith(f"cache/{DIRECTORY}/"):
        raise base.BenchmarkError(f"Unsafe cleanup path: {relative}")
    base.run(base.adb(adb_path, serial, "shell", "run-as", PACKAGE, "rm", "-f", relative), check=False)


def pull(adb_path, serial, relative) -> bytes:
    completed = subprocess.run(base.adb(adb_path, serial, "exec-out", "run-as", PACKAGE, "cat", relative),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise base.BenchmarkError(f"Could not read {relative} from the device.")
    remove(adb_path, serial, relative)
    return completed.stdout


def run_on_device(adb_path, serial, run_id, audio_name, threads, timeout, voice_name=None, voice_optimisation=False,
                  sample_scale=None, frame_padding=None):
    base.run(base.adb(adb_path, serial, "shell", "am", "force-stop", PACKAGE))
    command = base.adb(adb_path, serial, "shell", "am", "instrument", "-w", "-r")
    for key, value in {"class": TEST_CLASS, "runId": run_id, "audioName": audio_name, "threads": str(threads)}.items():
        command.extend(("-e", key, value))
    if voice_name:
        command.extend(("-e", "voiceModelName", voice_name))
        if voice_optimisation:
            command.extend(("-e", "voiceGraphOptimisation", "on"))
        if sample_scale is not None:
            command.extend(("-e", "voiceSampleScale", str(sample_scale)))
        if frame_padding is not None:
            command.extend(("-e", "voiceFramePadding", str(frame_padding)))
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
    segmentation = np.frombuffer(pull(adb_path, serial, f"cache/{DIRECTORY}/{run_id}.seg.f32"), "<f4")
    vectors = np.frombuffer(pull(adb_path, serial, f"cache/{DIRECTORY}/{run_id}.emb.f32"), "<f4")
    dimensions = report["result"]["embedding_dimensions"]
    return report, segmentation.reshape(-1, FRAMES, CLASSES), vectors.reshape(-1, dimensions)


def summarise(report, segmentation, vectors, reference):
    result = report["result"]
    ref_seg, ref_vec = reference
    seg_ok = segmentation.shape == ref_seg.shape
    vec_ok = vectors.shape == ref_vec.shape
    argmax = float((segmentation.argmax(-1) == ref_seg.argmax(-1)).mean()) if seg_ok else 0.0
    max_abs = float(np.abs(segmentation - ref_seg).max()) if seg_ok else float("inf")
    cosine = (vectors * ref_vec).sum(1) if vec_ok and len(ref_vec) else np.ones(1)
    chunk = result["segmentation_chunk_ms"][1:] or result["segmentation_chunk_ms"]
    window = result["embedding_window_ms"][1:] or result["embedding_window_ms"] or [0.0]
    seg_ms, emb_ms = statistics.median(chunk), statistics.median(window)
    return {
        "audio_seconds": round(result["audio_seconds"], 1),
        "segmentation_load_ms": round(result["segmentation_load_ms"], 1),
        "embedding_load_ms": round(result["embedding_load_ms"], 1),
        "segmentation_ms_per_10s_window": round(seg_ms, 1),
        "embedding_ms_per_1.5s_window": round(emb_ms, 1),
        "embedding_ms_long": {k: round(v, 1) for k, v in result["embedding_long_ms"].items()},
        # 60 min: 1440 segmentation windows (2.5 s hop); voiceprints for all 3600 s of speech at 1.5 s each.
        "minutes_for_60min_recording": {
            "segmentation": round(1440 * seg_ms / 60000, 2),
            "voiceprints_upper_bound": round(2400 * emb_ms / 60000, 2),
        },
        "segmentation_argmax_agreement": argmax, "segmentation_max_abs": max_abs,
        "voiceprint_min_cosine": float(cosine.min()), "voiceprint_mean_cosine": float(cosine.mean()),
        "voiceprints": int(len(vectors)),
        "parity_passed": bool(seg_ok and vec_ok and max_abs <= MAX_ABS_LIMIT and argmax >= MIN_ARGMAX
                              and cosine.min() >= MIN_COSINE),
        "total_pss_after_run_mib": round(result["memory_after_run"]["total_pss_kib"] / 1024, 1),
        "thermal_status": [result["thermal_status_before"], result["thermal_status_after"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--skip-build", action="store_true", help="reuse the APKs from the last isolated build")
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--voice-model", type=Path,
                        help="voice model .onnx (one fbank input, one voiceprint output) to stage and use "
                             "instead of the bundled one")
    parser.add_argument("--voice-sample-scale", type=float, default=SAMPLE_SCALE,
                        help="sample multiplier for --voice-model: 1 for 3D-Speaker (default), 32768 for WeSpeaker")
    parser.add_argument("--voice-graph-optimisation", action="store_true",
                        help="load the alternative voice model with ONNX Runtime graph optimisation on "
                             "(wrong for WeSpeaker CAM++ on ONNX Runtime 1.26)")
    parser.add_argument("--voice-frame-padding", type=int, default=FRAME_PADDING,
                        help="for --voice-model: repeat the feature frames up to a multiple of this; 0 for none. "
                             "3D-Speaker CAM++ needs 200 on ONNX Runtime 1.26 (default)")
    parser.add_argument("--timing-only", action="store_true", help="skip the parity runs on the other samples")
    args = parser.parse_args()
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
    out = ROOT / "build/reports/speaker-chain-benchmark" / suite_id
    out.mkdir(parents=True)
    samples = {sample.id: sample for sample in base.SAMPLES}
    seg.DIRECTORY = DIRECTORY  # stage() writes below files/<DIRECTORY>
    for sample in samples.values():
        seg.stage(args.adb, serial, sample.audio, f"{sample.id}.wav")
    voice_name = None
    if args.voice_model:
        voice_name = "alternative-voice-model.onnx"
        seg.stage(args.adb, serial, args.voice_model, voice_name)
    references = {sample_id: host_reference(sample.audio, args.voice_model or VOICE, args.voice_sample_scale,
                                            args.voice_frame_padding)
                  for sample_id, sample in samples.items()}

    rows, counter, device = [], 0, None
    try:
        plan = [(TIMING_SAMPLE, count, "timing") for count in threads]
        if not args.timing_only:
            plan += [(sample_id, 2, "parity") for sample_id in samples if sample_id != TIMING_SAMPLE]
        for sample_id, thread_count, kind in plan:
            counter += 1
            run_id = f"chain-{suite_id}-{counter:03d}"
            report, segmentation, vectors = run_on_device(args.adb, serial, run_id, f"{sample_id}.wav",
                                                          thread_count, args.timeout, voice_name,
                                                          args.voice_graph_optimisation,
                                                          args.voice_sample_scale if voice_name else None,
                                                          args.voice_frame_padding if voice_name else None)
            device = report["device"]
            row = {"kind": kind, "sample": sample_id, "threads": thread_count} | summarise(
                report, segmentation, vectors, references[sample_id])
            rows.append(row)
            base.atomic_json(out / f"{run_id}.json", report | {"summary": row})
            base.log(f"   {sample_id} t{thread_count}: seg {row['segmentation_ms_per_10s_window']} ms, "
                     f"voice {row['embedding_ms_per_1.5s_window']} ms, arg-max {row['segmentation_argmax_agreement']:.5f}, "
                     f"min cosine {row['voiceprint_min_cosine']:.6f}")
    finally:
        real_after = base.package_snapshot(args.adb, serial, REAL_PACKAGE)
        base.atomic_json(out / "summary.json", {
            "suite_id": suite_id, "device": device, "voice_model": str(args.voice_model or VOICE),
            "real_package_before": real_before, "real_package_after": real_after,
            "real_package_unchanged": real_before == real_after, "runs": rows,
        })
    base.assert_package_unchanged(real_before, real_after)

    print(f"\n{'sample':38s} {'thr':>3s} {'seg ms':>7s} {'voice ms':>8s} {'3/5/10 s':>18s} {'arg-max':>8s} {'min cos':>9s} pass")
    for row in rows:
        long = "/".join(str(row["embedding_ms_long"].get(k, "-")) for k in ("3", "5", "10"))
        print(f"{row['sample']:38s} {row['threads']:3d} {row['segmentation_ms_per_10s_window']:7.1f} "
              f"{row['embedding_ms_per_1.5s_window']:8.1f} {long:>18s} {row['segmentation_argmax_agreement']:8.5f} "
              f"{row['voiceprint_min_cosine']:9.6f} {row['parity_passed']}")
    print(f"real VoiceMemo package unchanged: {real_before == real_after}")
    print(f"report: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except base.BenchmarkError as error:
        sys.exit(f"error: {error}")
