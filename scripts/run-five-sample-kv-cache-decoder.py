#!/usr/bin/env python3
"""Benchmark corrected FP16 encoder plus stateful KV-cache decoder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MATRIX_RUNNER = ROOT / "scripts/run-five-sample-precision-matrix.py"
ENCODER = ROOT / "build/experiments/fp16-safe-layernorm-encoder/whisper-large-v3-turbo-fp16-safe-layernorm-encoder.tflite"
FP16_STORAGE_ENCODER = (
    ROOT
    / "build/experiments/fp16-safe-layernorm-encoder-fp16-storage/"
    "whisper-large-v3-turbo-fp16-safe-layernorm-fp16-storage-encoder.tflite"
)


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
KV_ROOT = ROOT / "build/experiments/original-decoder-kv-cache"
STEP = KV_ROOT / "decoder-kv-step.tflite"
FUSED_STEP = (
    ROOT
    / "build/experiments/original-decoder-kv-cache-fused/decoder-kv-step-fused.tflite"
)
CROSS = KV_ROOT / "decoder-cross-cache.tflite"
PROJECTION = ROOT / "build/experiments/original-decoder-active-row/active-row-projection-fp32.tflite"


def load_matrix():
    spec = importlib.util.spec_from_file_location("five_sample_kv_cache", MATRIX_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MATRIX_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--calculation-mode",
        choices=("native-fp16", "f32-f16"),
        default="f32-f16",
    )
    parser.add_argument(
        "--fused-projection",
        action="store_true",
        help="Use the fused step model that emits logits directly.",
    )
    parser.add_argument(
        "--fp16-storage-encoder",
        action="store_true",
        help="Use the FLOAT16-stored corrected encoder instead of the FLOAT32-stored one.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Decode with Whisper timestamp tokens enabled.",
    )
    known, remaining = parser.parse_known_args()
    matrix = load_matrix()
    if "--first-sample-only" in sys.argv:
        sys.argv.remove("--first-sample-only")
        matrix.SAMPLES = matrix.SAMPLES[:1]
    if known.fp16_storage_encoder:
        matrix.ENCODER = FP16_STORAGE_ENCODER
        matrix.ENCODER_SHA256 = file_sha256(FP16_STORAGE_ENCODER)
    else:
        matrix.ENCODER = ENCODER
        matrix.ENCODER_SHA256 = "14e580c6ce2815a67c8ba0a4db334c8c016b904ecbf464d6f31911191700a87e"
    if known.fused_projection:
        import hashlib

        digest = hashlib.sha256()
        with FUSED_STEP.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        matrix.DECODER = FUSED_STEP
        matrix.DECODER_SHA256 = digest.hexdigest()
        matrix.PROJECTION = None
        matrix.PROJECTION_SHA256 = None
    else:
        matrix.DECODER = STEP
        matrix.DECODER_SHA256 = "13ed9528e5df1ad62582f25fe8677bc63f2c26812017e84b80e015d5aa394b7a"
        matrix.PROJECTION = PROJECTION
        matrix.PROJECTION_SHA256 = "ba1eef8fed0d2edbd33fffafc2c7dfd0ac408ca2d8ed2262fec1f1a430dffd7b"
    matrix.CROSS_CACHE = CROSS
    matrix.CROSS_CACHE_SHA256 = "d3402e3b1aa9b2cece3b28d2919a1e3cca3ac9655c0f0a983a58d14b937b7b21"
    matrix.TIMESTAMP_TOKENS = known.timestamps
    projection_suffix = "fused projection" if known.fused_projection else "FP32 projection"
    if known.timestamps:
        projection_suffix += ", timestamp tokens"
    modes = {
        "native-fp16": (
            "litert-fp16-safe-kv-cache-native-fp16",
            f"LiteRT corrected FP16-safe encoder + native-FP16 KV-cache decoder + {projection_suffix}",
            "fp16-native",
        ),
        "f32-f16": (
            "litert-fp16-safe-kv-cache-f32-f16",
            f"LiteRT corrected FP16-safe encoder + F32_F16 KV-cache decoder + {projection_suffix}",
            "fp16-fp32-accum",
        ),
    }
    configuration_id, label, decoder_precision = modes[known.calculation_mode]
    matrix.CONFIGURATIONS = (
        matrix.Configuration(
            configuration_id,
            label,
            "litert",
            "fp16-native",
            decoder_precision,
        ),
    )
    sys.argv = [sys.argv[0], *remaining]
    if "--matrix-id" not in remaining:
        suffix = "native-fp16" if known.calculation_mode == "native-fp16" else "f32-f16"
        if known.fused_projection:
            suffix += "-fused"
        if known.fp16_storage_encoder:
            suffix += "-fp16-storage-encoder"
        if known.timestamps:
            suffix += "-timestamps"
        sys.argv.extend(
            (
                "--matrix-id",
                datetime.now(timezone.utc).strftime(
                    f"five-sample-kv-cache-{suffix}-%Y%m%dT%H%M%SZ"
                ),
            )
        )
    return matrix.main()


if __name__ == "__main__":
    raise SystemExit(main())
