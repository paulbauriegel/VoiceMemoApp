#!/usr/bin/env python3
"""Split the original-weight Whisper decoder before vocabulary projection.

The decoder core is graph-identical through the final decoder LayerNorm and
returns ``[1, 128, 1280]``.  The projection model reuses the exact terminal
weight payload but accepts only one active hidden row, reducing the terminal
matmul from 128 rows to one.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "build/experiments/original-fp32-decoder"
    / "whisper-large-v3-turbo-original-fp32-decoder.tflite"
)
SOURCE_SHA256 = "d661702d7753165cfaf01bf5b060f46065df500959e42e5254fd2598a54dedb7"
DEFAULT_OUTPUT = ROOT / "build/experiments/original-decoder-active-row"

HIDDEN_TENSOR_NAME = "layer_norm/batchnorm/add_1"
LOGITS_TENSOR_NAME = "StatefulPartitionedCall:0"
PROJECTION_WEIGHT_NAME = "arith.constant16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_helpers():
    path = ROOT / "scripts/split-whisper-encode-decode-model.py"
    spec = importlib.util.spec_from_file_location("split_whisper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load split-model helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_name(tensor: object) -> str:
    value = tensor.name
    return value.decode() if isinstance(value, bytes) else str(value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def dependency_ops(graph: object, output: int, inputs: set[int]) -> list[int]:
    producer = {
        int(tensor): operator_index
        for operator_index, operator in enumerate(graph.operators)
        for tensor in operator.outputs
        if int(tensor) >= 0
    }
    needed: set[int] = set()
    pending = [output]
    visited: set[int] = set()
    while pending:
        tensor = int(pending.pop())
        if tensor in visited or tensor in inputs:
            continue
        visited.add(tensor)
        operator_index = producer.get(tensor)
        if operator_index is None:
            continue
        needed.add(operator_index)
        pending.extend(
            int(value)
            for value in graph.operators[operator_index].inputs
            if int(value) >= 0
        )
    return sorted(needed)


def subset_model(
    helper: object,
    source_model: object,
    source_payloads: list[object],
    operator_indices: list[int],
    inputs: list[tuple[str, int]],
    outputs: list[tuple[str, int]],
    graph_name: bytes,
    signature_key: str,
) -> tuple[object, list[object]]:
    schema = helper.schema
    source_graph = source_model.subgraphs[0]
    selected_tensors = {index for _, index in inputs + outputs}
    for operator_index in operator_indices:
        operator = source_graph.operators[operator_index]
        selected_tensors.update(int(value) for value in operator.inputs if int(value) >= 0)
        selected_tensors.update(int(value) for value in operator.outputs if int(value) >= 0)
        selected_tensors.update(
            int(value)
            for value in (operator.intermediates or ())
            if int(value) >= 0
        )
    tensor_indices = sorted(selected_tensors)
    tensor_mapping = {old: new for new, old in enumerate(tensor_indices)}

    output_model = schema.ModelT()
    output_model.version = int(source_model.version)
    output_model.description = source_model.description
    output_model.operatorCodes = []
    output_model.buffers = [schema.BufferT()]
    output_payloads = [helper.Payload(None, 0, 0)]
    buffer_mapping = {0: 0}

    graph = schema.SubGraphT()
    graph.name = graph_name
    graph.tensors = []
    input_tensor_indices = {index for _, index in inputs}
    for old_index in tensor_indices:
        tensor = copy.deepcopy(source_graph.tensors[old_index])
        old_buffer = int(tensor.buffer)
        if old_index in input_tensor_indices:
            tensor.buffer = 0
        else:
            if old_buffer not in buffer_mapping:
                buffer_mapping[old_buffer] = len(output_model.buffers)
                output_model.buffers.append(copy.deepcopy(source_model.buffers[old_buffer]))
                output_payloads.append(source_payloads[old_buffer])
            tensor.buffer = buffer_mapping[old_buffer]
        graph.tensors.append(tensor)

    opcode_mapping: dict[int, int] = {}
    graph.operators = []
    for operator_index in operator_indices:
        operator = copy.deepcopy(source_graph.operators[operator_index])
        old_opcode = int(operator.opcodeIndex)
        if old_opcode not in opcode_mapping:
            opcode_mapping[old_opcode] = len(output_model.operatorCodes)
            output_model.operatorCodes.append(
                copy.deepcopy(source_model.operatorCodes[old_opcode])
            )
        operator.opcodeIndex = opcode_mapping[old_opcode]
        operator.inputs = [
            tensor_mapping[int(value)] if int(value) >= 0 else -1
            for value in operator.inputs
        ]
        operator.outputs = [
            tensor_mapping[int(value)] if int(value) >= 0 else -1
            for value in operator.outputs
        ]
        operator.intermediates = [
            tensor_mapping[int(value)] if int(value) >= 0 else -1
            for value in (operator.intermediates or ())
        ]
        graph.operators.append(operator)
    graph.inputs = [tensor_mapping[index] for _, index in inputs]
    graph.outputs = [tensor_mapping[index] for _, index in outputs]
    output_model.subgraphs = [graph]

    signature = schema.SignatureDefT()
    signature.signatureKey = signature_key
    signature.subgraphIndex = 0
    signature.inputs = []
    signature.outputs = []
    for public_name, old_index in inputs:
        mapping = schema.TensorMapT()
        mapping.name = public_name
        mapping.tensorIndex = tensor_mapping[old_index]
        signature.inputs.append(mapping)
    for public_name, old_index in outputs:
        mapping = schema.TensorMapT()
        mapping.name = public_name
        mapping.tensorIndex = tensor_mapping[old_index]
        signature.outputs.append(mapping)
    output_model.signatureDefs = [signature]
    return output_model, output_payloads


def build(source_path: Path, output: Path) -> dict[str, object]:
    helper = load_helpers()
    require(source_path.is_file(), f"Source decoder is missing: {source_path}")
    actual_sha = sha256(source_path)
    require(actual_sha == SOURCE_SHA256, f"Source SHA-256 changed: {actual_sha}")
    require(not output.exists() or not any(output.iterdir()), f"Output is not empty: {output}")

    opened = helper.OpenModel.open(source_path)
    try:
        source_model, source_payloads = helper.make_part(opened, 0, "decode")
    finally:
        opened.close()
    graph = source_model.subgraphs[0]
    names = {tensor_name(tensor): index for index, tensor in enumerate(graph.tensors)}
    hidden = names[HIDDEN_TENSOR_NAME]
    logits = names[LOGITS_TENSOR_NAME]
    weight = names[PROJECTION_WEIGHT_NAME]
    require(list(graph.tensors[hidden].shape) == [1, 128, 1280], "Hidden shape changed")
    require(list(graph.tensors[weight].shape) == [1, 1280, 51866], "Weight shape changed")
    require(list(graph.tensors[logits].shape) == [1, 128, 51866], "Logits shape changed")

    projection_ops = [
        index
        for index, operator in enumerate(graph.operators)
        if logits in [int(value) for value in operator.outputs]
    ]
    require(len(projection_ops) == 1, "Expected exactly one logits producer")
    projection_op = projection_ops[0]
    require(
        [int(value) for value in graph.operators[projection_op].inputs] == [hidden, weight],
        "Terminal projection wiring changed",
    )

    input_specs = [("args_2", int(graph.inputs[0])), ("args_0", int(graph.inputs[1])), ("args_1", int(graph.inputs[2]))]
    core_ops = dependency_ops(graph, hidden, {index for _, index in input_specs})
    require(projection_op not in core_ops, "Projection leaked into decoder core")
    core, core_payloads = subset_model(
        helper,
        source_model,
        source_payloads,
        core_ops,
        input_specs,
        [("output_0", hidden)],
        b"original_decoder_active_row_core",
        "decode",
    )

    projection, projection_payloads = subset_model(
        helper,
        source_model,
        source_payloads,
        [projection_op],
        [("hidden", hidden)],
        [("logits", logits)],
        b"original_decoder_active_row_projection",
        "project",
    )
    projection_graph = projection.subgraphs[0]
    projection_graph.tensors[projection_graph.inputs[0]].shape = [1, 1280]
    projection_graph.tensors[projection_graph.inputs[0]].shapeSignature = None
    projection_weight = int(projection_graph.operators[0].inputs[1])
    projection_graph.tensors[projection_weight].shape = [1280, 51866]
    projection_graph.tensors[projection_weight].shapeSignature = None
    projection_graph.tensors[projection_graph.outputs[0]].shape = [1, 51866]
    projection_graph.tensors[projection_graph.outputs[0]].shapeSignature = None

    core_path = output / "decoder-active-row-core.tflite"
    projection_path = output / "active-row-projection-fp32.tflite"
    helper.write_part(core_path, core, core_payloads)
    helper.write_part(projection_path, projection, projection_payloads)
    report = {
        "source": str(source_path),
        "source_sha256": actual_sha,
        "split_operator_index": projection_op,
        "core": {
            "path": str(core_path),
            "bytes": core_path.stat().st_size,
            "sha256": sha256(core_path),
            "operators": len(core_ops),
            "output_shape": [1, 128, 1280],
        },
        "projection": {
            "path": str(projection_path),
            "bytes": projection_path.stat().st_size,
            "sha256": sha256(projection_path),
            "input_shape": [1, 1280],
            "output_shape": [1, 51866],
            "weight_shape": [1280, 51866],
            "stored_weights": "FLOAT32",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.input.resolve(), args.output.resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
