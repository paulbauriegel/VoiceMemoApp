#!/usr/bin/env python3
"""Benchmark corrected FP16 encoder plus active-row decoder projection."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MATRIX_RUNNER = ROOT / "scripts/run-five-sample-precision-matrix.py"
ENCODER = ROOT / "build/experiments/fp16-safe-layernorm-encoder/whisper-large-v3-turbo-fp16-safe-layernorm-encoder.tflite"
CORE = ROOT / "build/experiments/original-decoder-active-row/decoder-active-row-core.tflite"
PROJECTION = ROOT / "build/experiments/original-decoder-active-row/active-row-projection-fp32.tflite"


def load_matrix():
    spec = importlib.util.spec_from_file_location("five_sample_active_row", MATRIX_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MATRIX_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    matrix = load_matrix()
    matrix.ENCODER = ENCODER
    matrix.ENCODER_SHA256 = "14e580c6ce2815a67c8ba0a4db334c8c016b904ecbf464d6f31911191700a87e"
    matrix.DECODER = CORE
    matrix.DECODER_SHA256 = "429c3bdbab4ad60698beda187158fd98fdfd9e9b8af1ae4aec2056593a0a9301"
    matrix.PROJECTION = PROJECTION
    matrix.PROJECTION_SHA256 = "ba1eef8fed0d2edbd33fffafc2c7dfd0ac408ca2d8ed2262fec1f1a430dffd7b"
    matrix.CONFIGURATIONS = (
        matrix.Configuration(
            "litert-fp16-safe-active-row",
            "LiteRT corrected FP16 encoder + FP16 decoder core + FP32 active-row projection",
            "litert",
            "fp16-native",
            "fp16-native",
        ),
    )
    if "--matrix-id" not in sys.argv:
        sys.argv.extend(
            (
                "--matrix-id",
                datetime.now(timezone.utc).strftime(
                    "five-sample-active-row-decoder-%Y%m%dT%H%M%SZ"
                ),
            )
        )
    return matrix.main()


if __name__ == "__main__":
    raise SystemExit(main())
