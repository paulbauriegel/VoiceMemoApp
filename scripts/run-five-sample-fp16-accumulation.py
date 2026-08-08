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
FP16_STORAGE_ENCODER = (
    ROOT
    / "build/experiments/fp16-safe-layernorm-encoder-fp16-storage/"
    "whisper-large-v3-turbo-fp16-safe-layernorm-fp16-storage-encoder.tflite"
)
FP16_STORAGE_ENCODER_SHA256 = (
    "faf331b520fbf67896e0080b8f7803037d32b57987ec914f93e185c3692c3fc1"
)


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
    parser.add_argument(
        "--fp16-storage-encoder",
        action="store_true",
        help="Use the FLOAT16-stored corrected encoder (the production asset).",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Decode with Whisper timestamp tokens enabled.",
    )
    known, remaining = parser.parse_known_args()
    matrix = load_matrix()
    encoder = FP16_STORAGE_ENCODER if known.fp16_storage_encoder else ENCODER
    encoder_sha = (
        FP16_STORAGE_ENCODER_SHA256 if known.fp16_storage_encoder else ENCODER_SHA256
    )
    if not encoder.is_file() or matrix.load_suite().sha256(encoder) != encoder_sha:
        raise RuntimeError(f"Missing or changed FP16-safe encoder: {encoder}")

    labels = {
        "native-fp16": "LiteRT corrected FP16 encoder + FP16 decoder, native FP16 accumulation",
        "f32-f16": "LiteRT corrected FP16 encoder + FP16 decoder, FP32 accumulation",
    }
    matrix.ENCODER = encoder
    matrix.ENCODER_SHA256 = encoder_sha
    matrix.TIMESTAMP_TOKENS = known.timestamps
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
        suffix = known.accumulation_mode
        if known.fp16_storage_encoder:
            suffix += "-fp16sto"
        if known.timestamps:
            suffix += "-timestamps"
        sys.argv.extend(
            (
                "--matrix-id",
                datetime.now(timezone.utc).strftime(
                    f"five-sample-fp16-{suffix}-%Y%m%dT%H%M%SZ"
                ),
            )
        )
    return matrix.main()


if __name__ == "__main__":
    raise SystemExit(main())
