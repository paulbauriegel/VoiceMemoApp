#!/usr/bin/env python3
"""Host-side spike for on-device speaker identification ("me vs. others").

Pipeline (mirrors the planned Android implementation):
  Silero VAD speech regions -> 80-bin Kaldi fbank -> speaker embedding per
  sliding window -> two-cluster agglomerative clustering -> centroid
  refinement + temporal smoothing -> speaker turns.

For every audio file and every candidate embedding model it writes
  <out>/<sample>/<model>.turns.json   speaker turns (draft labels to correct)
  <out>/<sample>/<model>.speaker{A,B}.wav   audio per cluster, for listening
and a summary.json with cluster separation, cross-model agreement and timing.

Optional inputs:
  <audio>.speakers.json  hand-labelled turns [{"start":s,"end":s,"speaker":"me"|"other"}]
                         -> diarization error rate per model
  --enroll CHANNEL=WAV   solo sample(s) of the user -> which cluster is "me",
                         cosine scores and precision/recall against labels

Dependencies: onnxruntime numpy soundfile scipy kaldi-native-fbank
Models: sherpa-onnx speaker-recognition release (see MODELS below), expected in
build/speaker-spike/models/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parent.parent
SR = 16_000
FRAME_S = 0.01  # scoring resolution

MODELS = {
    "wespeaker-resnet34-lm": (
        "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
    ),
    "wespeaker-campplus": (
        "wespeaker_en_voxceleb_CAM++.onnx",
        "c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef",
    ),
    "3dspeaker-campplus": (
        "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b",
    ),
}


def load_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SR:
        audio = resample_poly(audio, SR, rate).astype(np.float32)
    return audio


# --------------------------------------------------------------------------- VAD
class SileroVad:
    """Frame-level Silero probabilities, same framing as the app (512 + 64 context)."""

    FRAME = 512
    CONTEXT = 64

    def __init__(self, model: Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(self.CONTEXT, dtype=np.float32)
        sr = np.array(SR, dtype=np.int64)
        probs = []
        for offset in range(0, len(audio), self.FRAME):
            frame = np.zeros(self.FRAME, dtype=np.float32)
            chunk = audio[offset : offset + self.FRAME]
            frame[: len(chunk)] = chunk
            x = np.concatenate([context, frame])[None, :]
            out, state = self.session.run(
                ["output", "stateN"], {"input": x, "state": state, "sr": sr}
            )
            probs.append(float(out.reshape(-1)[0]))
            context = frame[-self.CONTEXT :]
        return np.asarray(probs, dtype=np.float32)

    def regions(
        self,
        audio: np.ndarray,
        threshold: float = 0.5,
        min_speech_s: float = 0.25,
        min_silence_s: float = 0.30,
        pad_s: float = 0.05,
    ) -> list[tuple[float, float]]:
        probs = self.probabilities(audio)
        hop = self.FRAME / SR
        raw: list[list[float]] = []
        start = None
        for i, p in enumerate(probs):
            if p >= threshold and start is None:
                start = i * hop
            elif p < threshold and start is not None:
                raw.append([start, i * hop])
                start = None
        if start is not None:
            raw.append([start, len(audio) / SR])
        merged: list[list[float]] = []
        for region in raw:
            if merged and region[0] - merged[-1][1] < min_silence_s:
                merged[-1][1] = region[1]
            else:
                merged.append(region)
        total = len(audio) / SR
        return [
            (max(0.0, s - pad_s), min(total, e + pad_s))
            for s, e in merged
            if e - s >= min_speech_s
        ]


# --------------------------------------------------------------------- embedding
class Embedder:
    def __init__(self, name: str, model: Path, threads: int):
        self.name = name
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )
        meta = self.session.get_modelmeta().custom_metadata_map
        self.normalize_samples = meta.get("normalize_samples", "0") == "1"
        self.input_name = self.session.get_inputs()[0].name
        self.dim = int(meta.get("output_dim", "0"))
        self.fbank_seconds = 0.0
        self.model_seconds = 0.0
        self.windows = 0

    def _fbank(self, samples: np.ndarray) -> np.ndarray:
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SR
        opts.frame_opts.dither = 0.0
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.frame_shift_ms = 10.0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(opts)
        scaled = samples if self.normalize_samples else samples * 32768.0
        fbank.accept_waveform(SR, scaled.tolist())
        fbank.input_finished()
        feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        return (feats - feats.mean(axis=0, keepdims=True)).astype(np.float32)

    def embed(self, samples: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        feats = self._fbank(samples)
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: feats[None]})[0][0]
        t2 = time.perf_counter()
        self.fbank_seconds += t1 - t0
        self.model_seconds += t2 - t1
        self.windows += 1
        return out / (np.linalg.norm(out) + 1e-9)


# ------------------------------------------------------------------- diarization
@dataclass
class Window:
    start: float
    end: float
    region: int


def make_windows(
    regions: list[tuple[float, float]], window_s: float, hop_s: float, min_s: float
) -> list[Window]:
    windows: list[Window] = []
    for index, (start, end) in enumerate(regions):
        length = end - start
        if length < min_s:
            continue
        if length <= window_s:
            windows.append(Window(start, end, index))
            continue
        t = start
        while t + window_s < end:
            windows.append(Window(t, t + window_s, index))
            t += hop_s
        windows.append(Window(end - window_s, end, index))
    return windows


def cluster_two(embeddings: np.ndarray) -> tuple[np.ndarray, dict]:
    """Average-link cosine clustering into two groups, then centroid refinement."""
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int), {"centroid_cosine": None}
    tree = linkage(embeddings, method="average", metric="cosine")
    labels = fcluster(tree, t=2, criterion="maxclust") - 1
    for _ in range(10):  # cosine k-means refinement against the two centroids
        centroids = []
        for k in (0, 1):
            members = embeddings[labels == k]
            if len(members) == 0:
                return np.zeros(len(embeddings), dtype=int), {"centroid_cosine": None}
            c = members.mean(axis=0)
            centroids.append(c / (np.linalg.norm(c) + 1e-9))
        scores = embeddings @ np.stack(centroids).T
        updated = scores.argmax(axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
    centroids_m = np.stack(centroids)
    scores = embeddings @ centroids_m.T
    own = scores[np.arange(len(labels)), labels]
    other = scores[np.arange(len(labels)), 1 - labels]
    stats = {
        # Low value = clearly different voices; near 1 = probably one speaker.
        "centroid_cosine": float(centroids_m[0] @ centroids_m[1]),
        "mean_margin": float((own - other).mean()),
        "low_margin_fraction": float(((own - other) < 0.05).mean()),
        "cluster_sizes": [int((labels == 0).sum()), int((labels == 1).sum())],
    }
    return labels, stats


def smooth(labels: np.ndarray, windows: list[Window]) -> np.ndarray:
    """Flip single windows that disagree with both neighbours in the same region."""
    out = labels.copy()
    for i in range(1, len(labels) - 1):
        same_region = windows[i - 1].region == windows[i].region == windows[i + 1].region
        if same_region and labels[i - 1] == labels[i + 1] != labels[i]:
            out[i] = labels[i - 1]
    return out


def to_frames(labels: np.ndarray, windows: list[Window], duration: float) -> np.ndarray:
    """Per-10ms speaker (-1 = no speech). Overlapping windows vote, centre-weighted."""
    n = int(np.ceil(duration / FRAME_S))
    votes = np.zeros((n, 2), dtype=np.float32)
    for label, window in zip(labels, windows):
        a, b = int(window.start / FRAME_S), min(n, int(window.end / FRAME_S))
        if b <= a:
            continue
        weight = np.hanning(b - a + 2)[1:-1] + 1e-3
        votes[a:b, label] += weight
    frames = np.where(votes.sum(axis=1) > 0, votes.argmax(axis=1), -1)
    return frames


def frames_to_turns(frames: np.ndarray, names=("A", "B")) -> list[dict]:
    turns = []
    start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] != frames[start]:
            if frames[start] >= 0:
                turns.append(
                    {
                        "start": round(start * FRAME_S, 2),
                        "end": round(i * FRAME_S, 2),
                        "speaker": names[frames[start]],
                    }
                )
            start = i
    return turns


def agreement(a: np.ndarray, b: np.ndarray) -> float | None:
    """Fraction of commonly-voiced frames with the same speaker (best permutation)."""
    both = (a >= 0) & (b >= 0)
    if not both.any():
        return None
    same = float((a[both] == b[both]).mean())
    return max(same, 1.0 - same)


def reference_frames(path: Path, duration: float) -> tuple[np.ndarray, list[str]]:
    turns = json.loads(path.read_text())
    speakers = sorted({t["speaker"] for t in turns})
    n = int(np.ceil(duration / FRAME_S))
    frames = np.full(n, -1)
    for t in turns:
        frames[int(t["start"] / FRAME_S) : int(t["end"] / FRAME_S)] = speakers.index(
            t["speaker"]
        )
    return frames, speakers


def diarization_error(ref: np.ndarray, hyp: np.ndarray, n_ref: int) -> dict:
    """DER over reference speech with the best speaker mapping, no collar."""
    speech = ref >= 0
    total = int(speech.sum())
    missed = int((speech & (hyp < 0)).sum())
    false_alarm = int((~speech & (hyp >= 0)).sum())
    best_confusion, best_map = None, None
    for perm in itertools.permutations(range(max(n_ref, 2)), 2):
        mapped = np.where(hyp >= 0, np.asarray(perm)[np.clip(hyp, 0, 1)], -1)
        confusion = int((speech & (hyp >= 0) & (mapped != ref)).sum())
        if best_confusion is None or confusion < best_confusion:
            best_confusion, best_map = confusion, perm
    return {
        "der": (missed + false_alarm + best_confusion) / max(total, 1),
        "missed": missed / max(total, 1),
        "false_alarm": false_alarm / max(total, 1),
        "confusion": best_confusion / max(total, 1),
        "mapping": list(best_map),
    }


# -------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--models-dir", type=Path, default=REPO / "build/speaker-spike/models")
    parser.add_argument(
        "--vad-model",
        type=Path,
        default=REPO / "app/src/main/generatedVadAssets/silero/silero_vad.onnx",
    )
    parser.add_argument("--out", type=Path, default=REPO / "build/reports/speaker-ident-spike")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--window", type=float, default=1.5)
    parser.add_argument("--hop", type=float, default=0.75)
    parser.add_argument("--min-window", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--enroll", action="append", default=[], metavar="CHANNEL=WAV",
        help="solo sample of the user for one input channel (repeatable)",
    )
    args = parser.parse_args()

    vad = SileroVad(args.vad_model)
    embedders = {
        name: Embedder(name, args.models_dir / MODELS[name][0], args.threads)
        for name in args.models
    }

    enrolled: dict[str, dict[str, np.ndarray]] = {name: {} for name in embedders}
    for item in args.enroll:
        channel, wav = item.split("=", 1)
        audio = load_audio(Path(wav))
        regions = vad.regions(audio)
        windows = make_windows(regions, args.window, args.hop, args.min_window)
        for name, embedder in embedders.items():
            vectors = np.stack(
                [embedder.embed(audio[int(w.start * SR) : int(w.end * SR)]) for w in windows]
            )
            centroid = vectors.mean(axis=0)
            enrolled[name][channel] = centroid / np.linalg.norm(centroid)
    for embedder in embedders.values():  # timing below covers the test audio only
        embedder.fbank_seconds = embedder.model_seconds = 0.0
        embedder.windows = 0

    summary = {"settings": vars(args) | {"audio": [str(p) for p in args.audio]}, "samples": []}
    summary["settings"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in summary["settings"].items()}
    audio_seconds = 0.0

    for path in args.audio:
        audio = load_audio(path)
        duration = len(audio) / SR
        audio_seconds += duration
        sample_name = path.parent.name if path.stem in ("audio", "reference") else path.stem
        out_dir = args.out / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)

        regions = vad.regions(audio)
        windows = make_windows(regions, args.window, args.hop, args.min_window)
        speech_seconds = sum(e - s for s, e in regions)
        entry = {
            "sample": sample_name,
            "duration_s": round(duration, 2),
            "speech_s": round(speech_seconds, 2),
            "regions": len(regions),
            "windows": len(windows),
            "models": {},
        }
        label_path = path.with_suffix(".speakers.json")
        ref = reference_frames(label_path, duration) if label_path.exists() else None

        frames_by_model = {}
        for name, embedder in embedders.items():
            vectors = np.stack(
                [embedder.embed(audio[int(w.start * SR) : int(w.end * SR)]) for w in windows]
            )
            labels, stats = cluster_two(vectors)
            labels = smooth(labels, windows)
            frames = to_frames(labels, windows, duration)
            frames_by_model[name] = frames
            names = ["A", "B"]
            result = dict(stats)

            if enrolled[name]:
                centroids = []
                for k in (0, 1):
                    members = vectors[labels == k]
                    c = members.mean(axis=0) if len(members) else np.zeros(vectors.shape[1])
                    centroids.append(c / (np.linalg.norm(c) + 1e-9))
                scores = {
                    channel: [float(c @ voice) for c in centroids]
                    for channel, voice in enrolled[name].items()
                }
                best = np.max(np.asarray(list(scores.values())), axis=0)
                me = int(best.argmax())
                names[me], names[1 - me] = "me", "other"
                result["enroll_scores"] = scores
                result["me_cluster_score"] = float(best[me])
                result["other_cluster_score"] = float(best[1 - me])

            turns = frames_to_turns(frames, tuple(names))
            result["turns"] = len(turns)
            result["seconds_per_speaker"] = {
                names[k]: round(float((frames == k).sum() * FRAME_S), 2) for k in (0, 1)
            }
            if ref is not None:
                result["reference"] = diarization_error(ref[0], frames, len(ref[1]))
            (out_dir / f"{name}.turns.json").write_text(json.dumps(turns, indent=1))
            for k in (0, 1):
                mask = np.repeat(frames == k, int(SR * FRAME_S))[: len(audio)]
                sf.write(str(out_dir / f"{name}.speaker{names[k].upper()}.wav"), audio[: len(mask)][mask], SR)
            entry["models"][name] = result

        entry["agreement"] = {
            f"{a} vs {b}": agreement(frames_by_model[a], frames_by_model[b])
            for a, b in itertools.combinations(frames_by_model, 2)
        }
        summary["samples"].append(entry)

    summary["timing_host"] = {
        name: {
            "windows": e.windows,
            "fbank_ms_per_window": round(1000 * e.fbank_seconds / max(e.windows, 1), 2),
            "model_ms_per_window": round(1000 * e.model_seconds / max(e.windows, 1), 2),
            "realtime_factor": round((e.fbank_seconds + e.model_seconds) / audio_seconds, 4),
            "threads": args.threads,
        }
        for name, e in embedders.items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    for entry in summary["samples"]:
        print(f"\n== {entry['sample']}: {entry['duration_s']} s, speech {entry['speech_s']} s, "
              f"{entry['windows']} windows")
        for name, r in entry["models"].items():
            line = (f"  {name:24s} centroid_cos={r['centroid_cosine']!s:.6} "
                    f"margin={r.get('mean_margin', 0):.3f} lowmargin={r.get('low_margin_fraction', 0):.2f} "
                    f"turns={r['turns']} secs={r['seconds_per_speaker']}")
            if "reference" in r:
                line += f" DER={r['reference']['der']:.3f} (conf {r['reference']['confusion']:.3f})"
            if "me_cluster_score" in r:
                line += f" me={r['me_cluster_score']:.3f} other={r['other_cluster_score']:.3f}"
            print(line)
        for pair, value in entry["agreement"].items():
            print(f"  agreement {pair}: {value if value is None else round(value, 3)}")
    print("\nhost timing:", json.dumps(summary["timing_host"], indent=1))
    print(f"\nreports: {args.out}")


if __name__ == "__main__":
    main()
