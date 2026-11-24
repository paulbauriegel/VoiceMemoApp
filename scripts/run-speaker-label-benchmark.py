#!/usr/bin/env python3
"""End-to-end check of the app's speaker labelling on the phone against a host reference.

Runs SpeakerLabelIndex (analyse, pause/resume, enrol, label) in the isolated package
com.example.voicememo.benchmark on public benchmark samples with an in-memory database,
and compares the resulting 10 ms timeline with a numpy/scipy reference of the same chain.
The "voice profile" is simply the first voiceprint found, on both sides; this checks that
phone and host agree, not how accurate the labels are. The real VoiceMemo package is
snapshotted before and after and must be unchanged.

  build/speaker-spike/venv/bin/python scripts/run-speaker-label-benchmark.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


chain = _load("speaker_chain_runner", "run-speaker-chain-benchmark.py")
reference = _load("speaker_labeling_reference", "generate-speaker-labeling-fixture.py")
seg, base = chain.seg, chain.base
PACKAGE, TEST_PACKAGE, REAL_PACKAGE = base.PACKAGE, base.TEST_PACKAGE, base.REAL_PACKAGE
TEST_CLASS = "com.example.voicememo.speaker.SpeakerLabelIndexBenchmarkTest#runBenchmark"
DIRECTORY = chain.DIRECTORY
SR, WINDOW, HOP, SHIFT, CENTRE = 16000, 160000, 40000, 270, 495
POWERSET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], bool)
DEFAULT_SAMPLES = ["controlled-german-170s", "sample-test-153s", "english-dialogue-proper-names-47s"]


def host_timeline(path: Path):
    """numpy/scipy version of SpeakerVoiceprintCollector + SpeakerDecision + SpeakerTimeline."""
    import kaldi_native_fbank as knf
    import onnxruntime as ort
    audio = seg.load_audio(path)
    segmenter = ort.InferenceSession(str(chain.SEGMENTATION), providers=["CPUExecutionProvider"])
    voice = ort.InferenceSession(str(chain.VOICE), providers=["CPUExecutionProvider"])
    options = knf.FbankOptions()
    options.frame_opts.samp_freq = SR
    options.frame_opts.dither = 0.0
    options.frame_opts.snip_edges = True
    options.mel_opts.num_bins = 80

    def embed(samples):
        fbank = knf.OnlineFbank(options)
        fbank.accept_waveform(SR, (samples * chain.SAMPLE_SCALE).tolist())
        fbank.input_finished()
        features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        features = (features - features.mean(0, keepdims=True)).astype(np.float32)
        padded = -(-len(features) // chain.FRAME_PADDING) * chain.FRAME_PADDING
        features = np.resize(features, (padded, features.shape[1]))
        vector = voice.run(None, {voice.get_inputs()[0].name: features[None]})[0][0]
        return vector / np.linalg.norm(vector)

    count = 1 if len(audio) <= WINDOW else 1 + -(-(len(audio) - WINDOW) // HOP)
    speakers, vectors = [], []
    for index in range(count):
        start = index * HOP
        piece = audio[start:start + WINDOW]
        padded = np.zeros(WINDOW, np.float32)
        padded[:len(piece)] = piece
        mask = POWERSET[segmenter.run(None, {"x": padded[None, None, :]})[0][0].argmax(1)].copy()
        mask[(np.arange(len(mask)) * SHIFT + CENTRE) >= len(piece)] = False
        alone = mask.sum(1) == 1
        for speaker in range(3):
            active = mask[:, speaker]
            if active.sum() < 15:
                continue
            use = active & alone
            if use.sum() < 24:
                use = active
            frames = np.flatnonzero(use)
            if len(frames) > 237:
                first = (len(frames) - 237) // 2
                frames = frames[first:first + 237]
            samples = np.concatenate([piece[f * SHIFT + CENTRE - SHIFT // 2: f * SHIFT + CENTRE + SHIFT // 2 + 1] for f in frames])
            if len(samples) < 4000:
                continue
            vectors.append(embed(samples))
            speakers.append((start, active))
    vectors = np.stack(vectors).astype(np.float32)
    _, is_me = reference.adapt(vectors, vectors[:1])
    labels = reference.timeline([(start, active, me) for (start, active), me in zip(speakers, is_me)], len(audio))
    return labels, len(vectors), count


def decode(encoded: bytes) -> np.ndarray:
    labels, index = [], 0
    while index < len(encoded):
        label = encoded[index]; index += 1
        run, shift = 0, 0
        while True:
            part = encoded[index]; index += 1
            run |= (part & 0x7F) << shift
            if part < 0x80:
                break
            shift += 7
        labels.extend([label] * run)
    return np.array(labels, np.int8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--skip-build", action="store_true", help="reuse the APKs from the last isolated build")
    parser.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    base.validate_samples()
    serial = base.resolve_serial(args.adb, args.serial)
    samples = {sample.id: sample for sample in base.SAMPLES}
    chosen = [samples[name] for name in args.samples.split(",")]

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
    out = ROOT / "build/reports/speaker-label-benchmark" / suite_id
    out.mkdir(parents=True)
    seg.DIRECTORY = DIRECTORY
    rows = []
    try:
        for number, sample in enumerate(chosen, 1):
            seg.stage(args.adb, serial, sample.audio, f"{sample.id}.wav")
            host_labels, host_voiceprints, host_windows = host_timeline(sample.audio)
            run_id = f"label-{suite_id}-{number:03d}"
            base.run(base.adb(args.adb, serial, "shell", "am", "force-stop", PACKAGE))
            command = base.adb(args.adb, serial, "shell", "am", "instrument", "-w", "-r", "-e", "class", TEST_CLASS,
                               "-e", "runId", run_id, "-e", "audioName", f"{sample.id}.wav",
                               f"{TEST_PACKAGE}/{base.RUNNER}")
            base.log("+ " + " ".join(command))
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       timeout=args.timeout)
            try:
                report = json.loads(chain.pull(args.adb, serial, f"cache/{DIRECTORY}/{run_id}.json"))
            except (base.BenchmarkError, json.JSONDecodeError) as error:
                print(completed.stdout)
                raise base.BenchmarkError(f"Run {run_id} wrote no report.") from error
            if "OK (1 test)" not in completed.stdout or report.get("status") != "completed":
                print(completed.stdout)
                raise base.BenchmarkError(f"Run {run_id} failed: {report.get('error_type')}: {report.get('error_message')}")
            if report.get("package_name") != PACKAGE:
                raise base.BenchmarkError("Report came from a non-isolated package.")
            device_labels = decode(chain.pull(args.adb, serial, f"cache/{DIRECTORY}/{run_id}.timeline"))
            result = report["result"]
            same_length = len(device_labels) == len(host_labels)
            agreement = float((device_labels == host_labels).mean()) if same_length else 0.0
            audio_seconds = result["sample_count"] / SR
            row = {
                "sample": sample.id, "audio_seconds": round(audio_seconds, 1),
                "windows": result["windows"], "host_windows": host_windows,
                "voiceprints": result["voiceprints"], "host_voiceprints": host_voiceprints,
                "analysis_ms": result["analysis_ms"],
                "minutes_for_60min_recording": round(result["analysis_ms"] / audio_seconds * 3600 / 60000, 2),
                "relabel_ms": round(result["relabel_ms"], 1), "timeline_bytes": result["timeline_bytes"],
                "timeline_agreement_vs_host": agreement,
                "share_me": round(float((device_labels == 1).mean()), 3),
                "share_other": round(float((device_labels == 2).mean()), 3),
                "pause_resume_ok": bool(result["paused_returned_null"]
                                        and result["voiceprints_after_resume"] == result["voiceprints"]
                                        and result["voiceprints_stored_at_pause"] <= result["voiceprints"]),
                "total_pss_mib": round(result["memory"]["total_pss_kib"] / 1024, 1),
            }
            row["passed"] = bool(same_length and agreement >= 0.999 and row["voiceprints"] == host_voiceprints
                                 and row["windows"] == host_windows and row["pause_resume_ok"])
            rows.append(row)
            base.atomic_json(out / f"{run_id}.json", report | {"summary": row})
            base.log("   " + json.dumps(row))
    finally:
        real_after = base.package_snapshot(args.adb, serial, REAL_PACKAGE)
        base.atomic_json(out / "summary.json", {
            "suite_id": suite_id, "real_package_unchanged": real_before == real_after, "runs": rows})
    base.assert_package_unchanged(real_before, real_after)
    print(f"\n{'sample':36s} {'audio s':>7s} {'windows':>7s} {'prints':>6s} {'analysis s':>10s} {'60 min':>8s} {'relabel':>8s} {'agree':>7s} {'me/other':>11s} pass")
    for row in rows:
        print(f"{row['sample']:36s} {row['audio_seconds']:7.1f} {row['windows']:7d} {row['voiceprints']:6d} "
              f"{row['analysis_ms'] / 1000:10.1f} {row['minutes_for_60min_recording']:6.2f} m {row['relabel_ms']:6.1f}ms "
              f"{row['timeline_agreement_vs_host']:7.4f} {row['share_me']:.2f}/{row['share_other']:.2f} {row['passed']}")
    print(f"real VoiceMemo package unchanged: {real_before == real_after}")
    print(f"report: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except base.BenchmarkError as error:
        sys.exit(f"error: {error}")
