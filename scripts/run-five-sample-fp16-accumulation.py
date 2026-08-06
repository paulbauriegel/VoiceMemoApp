#!/usr/bin/env python3
"""Benchmark the corrected FP16 encoder/decoder for one accumulator runtime."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MATRIX_RUNNER = ROOT / "scripts/run-five-sample-precision-matrix.py"
ENCODER = (
    ROOT
    / "build/experiments/fp16-safe-layernorm-encoder/"
    "whisper-large-v3-turbo-fp16-safe-layernorm-encoder.tflite"
)
ENCODER_SHA256 = "14e580c6ce2815a67c8ba0a4db334c8c016b904ecbf464d6f31911191700a87e"


def load_matrix():
    spec = importlib.util.spec_from_file_location("five_sample_matrix", MATRIX_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MATRIX_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--accumulation-mode",
        choices=("native-fp16", "f32-f16"),
        required=True,
    )
    known, remaining = parser.parse_known_args()
    matrix = load_matrix()
    if not ENCODER.is_file() or matrix.load_suite().sha256(ENCODER) != ENCODER_SHA256:
        raise RuntimeError(f"Missing or changed FP16-safe encoder: {ENCODER}")

    labels = {
        "native-fp16": "LiteRT corrected FP16 encoder + FP16 decoder, native FP16 accumulation",
        "f32-f16": "LiteRT corrected FP16 encoder + FP16 decoder, FP32 accumulation",
    }
    matrix.ENCODER = ENCODER
    matrix.ENCODER_SHA256 = ENCODER_SHA256
    matrix.CONFIGURATIONS = (
        matrix.Configuration(
            f"litert-fp16-safe-fp16-{known.accumulation_mode}",
            labels[known.accumulation_mode],
            "litert",
            "fp16-native",
            "fp16-native",
        ),
    )
    sys.argv = [sys.argv[0], *remaining]
    if "--matrix-id" not in remaining:
        sys.argv.extend(
            (
                "--matrix-id",
                datetime.now(timezone.utc).strftime(
                    f"five-sample-fp16-{known.accumulation_mode}-%Y%m%dT%H%M%SZ"
                ),
            )
        )
    return matrix.main()


if __name__ == "__main__":
    raise SystemExit(main())
