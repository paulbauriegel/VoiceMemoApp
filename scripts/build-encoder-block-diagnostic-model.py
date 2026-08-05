#!/usr/bin/env python3
"""Expose selected Whisper encoder residual boundaries as TFLite outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build/experiments/original-fp32-encoder/whisper-large-v3-turbo-original-fp32-encoder.tflite"
OUTPUT = ROOT / "build/experiments/encoder-block-diagnostic/whisper-large-v3-turbo-encoder-late-block-boundaries-v2.tflite"
SOURCE_SHA256 = "01be8e811608cc9eb3780408f8e020509d216a1d5e280320232b2c0a13d5e353"
BOUNDARIES = (
    ("conv_stem", "encoder/add"),
    ("block_00", "encoder/layers.0/add_1"),
    ("block_03", "encoder/layers.3/add_1"),
    ("block_07", "encoder/layers.7/add_1"),
    ("block_15", "encoder/layers.15/add_1"),
    ("block_23", "encoder/layers.23/add_1"),
    ("block_24", "encoder/layers.24/add_1"),
    ("block_25", "encoder/layers.25/add_1"),
    ("block_26", "encoder/layers.26/add_1"),
    ("block_27", "encoder/layers.27/add_1"),
    ("block_28", "encoder/layers.28/add_1"),
    ("block_29", "encoder/layers.29/add_1"),
    ("block_30", "encoder/layers.30/add_1"),
    ("block_31", "encoder/layers.31/add_1"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_split_helpers():
    path = ROOT / "scripts/split-whisper-encode-decode-model.py"
    spec = importlib.util.spec_from_file_location("split_whisper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load split-model helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    if sha256(source_path) != SOURCE_SHA256:
        raise RuntimeError("Original FP32 encoder SHA-256 does not match")

    helper = load_split_helpers()
    schema = helper.schema
    source = helper.OpenModel.open(source_path)
    try:
        model, payloads = helper.make_part(source, 0, "encode")
        graph = model.subgraphs[0]
        graph.outputs = list(graph.outputs)
        by_name = {
            tensor.name.decode() if isinstance(tensor.name, bytes) else tensor.name: index
            for index, tensor in enumerate(graph.tensors)
        }
        signature = model.signatureDefs[0]
        signature.outputs = list(signature.outputs)
        selected = []
        for public_name, tensor_name in BOUNDARIES:
            if tensor_name not in by_name:
                raise RuntimeError(f"Missing encoder boundary tensor {tensor_name}")
            tensor_index = by_name[tensor_name]
            tensor = graph.tensors[tensor_index]
            if list(tensor.shape) != [1, 1500, 1280] or int(tensor.type) != 0:
                raise RuntimeError(f"Unexpected encoder boundary contract for {tensor_name}")
            mapping = schema.TensorMapT()
            mapping.name = public_name
            mapping.tensorIndex = tensor_index
            signature.outputs.append(mapping)
            graph.outputs.append(tensor_index)
            selected.append(
                {
                    "name": public_name,
                    "tensor_name": tensor_name,
                    "tensor_index": tensor_index,
                }
            )
        helper.write_part(output_path, model, payloads)
    finally:
        source.close()

    inspection = {
        "kind": "encoder-block-boundary-diagnostic",
        "source": str(source_path),
        "source_sha256": SOURCE_SHA256,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "bytes": output_path.stat().st_size,
        "boundaries": selected,
    }
    output_path.with_suffix(".inspection.json").write_text(
        json.dumps(inspection, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inspection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
