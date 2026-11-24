#!/usr/bin/env python3
"""Writes the parity fixture for the Kotlin speaker-labelling logic.

Reference implementation (numpy + scipy) of SpeakerDecision.adapt and
SpeakerTimeline.build/labelFor on synthetic voiceprints. Run with a Python that has
numpy and scipy, e.g. build/speaker-spike/venv/bin/python; the output is committed at
app/src/test/resources/speaker/labeling-parity.txt.
"""
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app/src/test/resources/speaker/labeling-parity.txt"
DIM, SAME_VOICE, MIN_ADAPT = 24, 0.5, 8
SR, SHIFT, CENTRE, FRAMES = 16000, 270, 495, 589


def unit(v):
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def otsu(s):
    best, thr = -1.0, float(np.median(s))
    for t in np.linspace(np.percentile(s, 2), np.percentile(s, 98), 100):
        a, b = s[s < t], s[s >= t]
        if len(a) < 3 or len(b) < 3:
            continue
        score = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if score > best:
            best, thr = score, float(t)
    return thr


def adapt(v, profile):
    """Returns (threshold, is_me per voiceprint)."""
    score = lambda x: (x @ profile.T).max(-1)
    if len(v) < MIN_ADAPT:
        return SAME_VOICE, score(v) >= SAME_VOICE
    s = score(v).astype(np.float64)
    thr = otsu(s)
    hi = s >= (thr + s[s >= thr].mean()) / 2
    lo = s < (thr + s[s < thr].mean()) / 2
    if not hi.any() or not lo.any():
        return SAME_VOICE, score(v) >= SAME_VOICE
    me, others = unit(v[hi].mean(0)), v[lo]
    lab = fcluster(linkage(others, "average", "cosine"), t=0.65, criterion="distance") if len(others) > 2 else np.ones(len(others))
    cents = [unit(others[lab == k].mean(0)) for k in np.unique(lab) if (lab == k).sum() >= max(5, 0.05 * len(others))]
    if not cents:
        cents = [unit(others.mean(0))]
    if score(me) < SAME_VOICE:
        return thr, np.zeros(len(v), bool)
    cents = [c for c in cents if score(c) < SAME_VOICE]
    if not cents:
        return thr, np.ones(len(v), bool)
    return thr, (v @ me) >= (v @ np.stack(cents).T).max(1)


def timeline(speakers, samples):
    nf = int(np.ceil(samples / SR * 100)) + 10
    me, ot = np.zeros(nf, np.float32), np.zeros(nf, np.float32)
    centre = (np.hanning(FRAMES + 2)[1:-1] + 1e-3).astype(np.float32)
    for start, active, is_me in speakers:
        idx = np.flatnonzero(active)
        t = ((start + idx * SHIFT + CENTRE) * 100 // SR).astype(int)
        keep = t < nf
        np.add.at(me if is_me else ot, t[keep], centre[idx[keep]])
    labels = np.zeros(nf, np.int8)
    voiced = (me + ot) > 0
    labels[voiced & (me >= ot)] = 1
    labels[voiced & (ot > me)] = 2
    return labels


def label_for(labels, start_ms, end_ms):
    a = min(start_ms // 10, len(labels)); b = min(max(end_ms // 10, a + 1), len(labels))
    part = labels[a:b]; voiced = int((part > 0).sum())
    if voiced < 5:
        return "UNKNOWN"
    share = float((part == 1).sum() / voiced)
    return "ME" if share >= 0.75 else "OTHER" if share <= 0.25 else "MIXED"


def main():
    rng = np.random.default_rng(20261119)
    voices = unit(rng.standard_normal((4, DIM)))
    around = lambda voice, n, noise: unit(voices[voice] + noise * rng.standard_normal((n, DIM)) / np.sqrt(DIM))
    profile = unit(np.stack([around(0, 40, 0.7).mean(0) for _ in range(3)]))
    cases = {
        "two_voices": np.concatenate([around(0, 40, 0.9), around(1, 25, 0.9)]),
        "three_voices": np.concatenate([around(0, 30, 0.9), around(1, 20, 0.9), around(2, 18, 0.9), around(3, 2, 0.9)]),
        "only_me": around(0, 50, 1.0),
        "nobody_enrolled": np.concatenate([around(1, 30, 0.9), around(2, 25, 0.9)]),
        "too_few_to_adapt": np.concatenate([around(0, 4, 0.8), around(1, 3, 0.8)]),
    }
    lines = [f"dim {DIM}", f"profile {len(profile)}"] + [" ".join(f"{x:.9g}" for x in row) for row in profile]
    for name, v in cases.items():
        v = v[rng.permutation(len(v))]
        thr, is_me = adapt(v, profile)
        lines.append(f"case {name} {len(v)} {thr:.6f} {''.join('1' if m else '0' for m in is_me)}")
        lines += [" ".join(f"{x:.9g}" for x in row) for row in v]

    samples = 30 * SR + 1234
    speakers = []
    for chunk, start in enumerate(range(0, samples - 160000 + 40000, 40000)):
        t = (start + np.arange(FRAMES) * SHIFT + CENTRE) / SR
        inside = start + np.arange(FRAMES) * SHIFT + CENTRE < samples
        mine = inside & (((t > 1.0) & (t < 9.5)) | ((t > 17.0) & (t < 21.0)))
        theirs = inside & (((t > 9.0) & (t < 16.8)) | ((t > 21.2) & (t < 28.0)))
        if mine.sum() >= 15:
            speakers.append((start, mine, True))
        if theirs.sum() >= 15:
            # One window gets the other voice wrong, as happens in practice.
            speakers.append((start, theirs, chunk == 2))
    labels = timeline(speakers, samples)
    lines.append(f"timeline {samples} {len(speakers)} {''.join(map(str, labels))}")
    for start, active, is_me in speakers:
        lines.append(f"{start} {int(is_me)} {''.join('1' if a else '0' for a in active)}")
    segments = [(0, 900), (1500, 6000), (8800, 9800), (10000, 16000), (16700, 17300), (21000, 21030), (22000, 27000), (29000, 31000), (40000, 41000), (8000, 11000), (15000, 19000), (20000, 22500)]
    lines.append(f"segments {len(segments)}")
    lines += [f"{a} {b} {label_for(labels, a, b)}" for a, b in segments]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT, {name: None for name in cases})
    for line in lines:
        if line.startswith("case") or "UNKNOWN" in line or "MIXED" in line or line.endswith(("ME", "OTHER")):
            print(line[:140])


if __name__ == "__main__":
    main()
