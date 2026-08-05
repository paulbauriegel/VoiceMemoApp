#!/usr/bin/env python3
"""Build narrow LiteRT encoder probes without modifying production models.

The stage mode retains only the dependency closure from the mel input to one
selected tensor in an encoder block. The final-layernorm mode extracts the
encoder's last LayerNorm as a standalone graph whose input is the block-31
residual state. Models are generated one at a time to avoid duplicating the
multi-gigabyte encoder weights for every checkpoint.
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
SOURCE = ROOT / "build/experiments/original-fp32-encoder/whisper-large-v3-turbo-original-fp32-encoder.tflite"
SOURCE_SHA256 = "01be8e811608cc9eb3780408f8e020509d216a1d5e280320232b2c0a13d5e353"
DEFAULT_STAGE_OUTPUT = ROOT / "build/experiments/encoder-stage-diagnostic/encoder-stage-diagnostic.tflite"
DEFAULT_LAYER_NORM_OUTPUT = ROOT / "build/experiments/encoder-stage-diagnostic/encoder-final-layernorm.tflite"
DEFAULT_FFN_LAYER_NORM_REPRO_OUTPUT = (
    ROOT / "repro/litert-fp16-layernorm-overflow/layernorm-variance-repro.tflite"
)


STAGE_SUFFIXES = {
    "input": None,
    "attn_ln_mean": "self_attn_layer_norm/moments/mean",
    "attn_ln_variance": "self_attn_layer_norm/moments/variance",
    "attn_ln_rsqrt": "self_attn_layer_norm/batchnorm/Rsqrt",
    "attn_ln_output": "self_attn_layer_norm/batchnorm/add_1",
    "k_projection": "self_attn/k_proj/Tensordot/MatMul",
    "q_projection": "self_attn/q_proj/Tensordot;encoder/layers.{block}/self_attn/q_proj/BiasAdd;encoder/layers.{block}/self_attn/mul",
    "qk_logits": "self_attn/MatMul",
    "softmax": "self_attn/Softmax",
    "v_projection": "self_attn/v_proj/Tensordot;encoder/layers.{block}/self_attn/v_proj/BiasAdd",
    "attention_value": "self_attn/MatMul_1",
    "attention_projection": "self_attn/out_proj/Tensordot/MatMul;encoder/layers.{block}/self_attn/out_proj/BiasAdd;encoder/layers.{block}/self_attn/out_proj/Tensordot;",
    "attention_residual": "add;encoder/layers.{block}/self_attn/out_proj/BiasAdd;encoder/layers.{block}/self_attn/out_proj/Tensordot;",
    "ffn_ln_mean": "final_layer_norm/moments/mean",
    "ffn_ln_variance": "final_layer_norm/moments/variance",
    "ffn_ln_rsqrt": "final_layer_norm/batchnorm/Rsqrt",
    "ffn_ln_output": "final_layer_norm/batchnorm/add_1",
    "fc1": "fc1/Tensordot;encoder/layers.{block}/fc1/BiasAdd",
    "gelu": "Gelu/mul_2;;encoder/layers.{block}/Gelu/mul;encoder/layers.{block}/Gelu/Neg;encoder/layers.{block}/Gelu/mul_1;encoder/layers.{block}/Gelu/Erfc",
    "fc2": "fc2/Tensordot;encoder/layers.{block}/fc2/BiasAdd",
    "block_output": "add_1",
}


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


def tensor_name(tensor: object) -> str:
    value = tensor.name
    return value.decode() if isinstance(value, bytes) else str(value)


def dependency_ops(graph: object, outputs: list[int], stop_inputs: set[int]) -> list[int]:
    producer = {
        int(tensor): operator_index
        for operator_index, operator in enumerate(graph.operators)
        for tensor in operator.outputs
        if int(tensor) >= 0
    }
    needed: set[int] = set()
    pending = list(outputs)
    visited: set[int] = set()
    while pending:
        tensor = int(pending.pop())
        if tensor in visited or tensor in stop_inputs:
            continue
        visited.add(tensor)
        operator_index = producer.get(tensor)
        if operator_index is None:
            continue
        if operator_index in needed:
            continue
        needed.add(operator_index)
        pending.extend(int(value) for value in graph.operators[operator_index].inputs if int(value) >= 0)
    return sorted(needed)


def subset_model(
    helper: object,
    source_model: object,
    source_payloads: list[object],
    operator_indices: list[int],
    inputs: list[tuple[str, int]],
    outputs: list[tuple[str, int]],
) -> tuple[object, list[object]]:
    schema = helper.schema
    source_graph = source_model.subgraphs[0]
    selected_tensors = {index for _, index in inputs + outputs}
    for operator_index in operator_indices:
        operator = source_graph.operators[operator_index]
        selected_tensors.update(int(value) for value in operator.inputs if int(value) >= 0)
        selected_tensors.update(int(value) for value in operator.outputs if int(value) >= 0)
        selected_tensors.update(
            int(value) for value in (operator.intermediates or []) if int(value) >= 0
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
    graph.name = b"encoder_precision_diagnostic"
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
            output_model.operatorCodes.append(copy.deepcopy(source_model.operatorCodes[old_opcode]))
        operator.opcodeIndex = opcode_mapping[old_opcode]
        operator.inputs = [tensor_mapping[int(value)] if int(value) >= 0 else -1 for value in operator.inputs]
        operator.outputs = [tensor_mapping[int(value)] if int(value) >= 0 else -1 for value in operator.outputs]
        operator.intermediates = [
            tensor_mapping[int(value)] if int(value) >= 0 else -1
            for value in (operator.intermediates or [])
        ]
        graph.operators.append(operator)
    graph.inputs = [tensor_mapping[index] for _, index in inputs]
    graph.outputs = [tensor_mapping[index] for _, index in outputs]
    output_model.subgraphs = [graph]

    signature = schema.SignatureDefT()
    signature.signatureKey = "diagnostic"
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


def stage_tensor_name(block: int, stage: str) -> str:
    if stage == "input":
        return "encoder/add" if block == 0 else f"encoder/layers.{block - 1}/add_1"
    suffix = STAGE_SUFFIXES[stage]
    assert suffix is not None
    return f"encoder/layers.{block}/{suffix.format(block=block)}"


def build_stage(helper: object, source: object, block: int, stage: str, output: Path) -> dict:
    model, payloads = helper.make_part(source, 0, "encode")
    graph = model.subgraphs[0]
    names = {tensor_name(tensor): index for index, tensor in enumerate(graph.tensors)}
    target_name = stage_tensor_name(block, stage)
    if target_name not in names:
        raise RuntimeError(f"Missing encoder stage tensor: {target_name}")
    target = names[target_name]
    input_index = int(graph.inputs[0])
    operators = dependency_ops(graph, [target], {input_index})
    diagnostic, diagnostic_payloads = subset_model(
        helper,
        model,
        payloads,
        operators,
        [("mel", input_index)],
        [("stage", target)],
    )
    helper.write_part(output, diagnostic, diagnostic_payloads)
    tensor = graph.tensors[target]
    return {
        "kind": "encoder-stage",
        "block": block,
        "stage": stage,
        "tensor_name": target_name,
        "tensor_shape": [int(value) for value in tensor.shape],
    }


def build_stage_group(
    helper: object,
    source: object,
    block: int,
    stages: list[str],
    output: Path,
) -> dict:
    model, payloads = helper.make_part(source, 0, "encode")
    graph = model.subgraphs[0]
    names = {tensor_name(tensor): index for index, tensor in enumerate(graph.tensors)}
    output_specs = []
    for stage in stages:
        target_name = stage_tensor_name(block, stage)
        if target_name not in names:
            raise RuntimeError(f"Missing encoder stage tensor: {target_name}")
        output_specs.append((stage, names[target_name]))
    input_index = int(graph.inputs[0])
    operators = dependency_ops(graph, [index for _, index in output_specs], {input_index})
    diagnostic, diagnostic_payloads = subset_model(
        helper,
        model,
        payloads,
        operators,
        [("mel", input_index)],
        output_specs,
    )
    helper.write_part(output, diagnostic, diagnostic_payloads)
    return {
        "kind": "encoder-stage-group",
        "block": block,
        "stages": {
            stage: {
                "tensor_name": tensor_name(graph.tensors[index]),
                "tensor_shape": [int(value) for value in graph.tensors[index].shape],
            }
            for stage, index in output_specs
        },
    }


def build_final_layer_norm(helper: object, source: object, output: Path) -> dict:
    model, payloads = helper.make_part(source, 0, "encode")
    graph = model.subgraphs[0]
    names = {tensor_name(tensor): index for index, tensor in enumerate(graph.tensors)}
    input_index = names["encoder/layers.31/add_1"]
    output_specs = [
        ("mean", names["encoder/layer_norm/moments/mean"]),
        ("variance", names["encoder/layer_norm/moments/variance"]),
        ("rsqrt", names["encoder/layer_norm/batchnorm/Rsqrt"]),
        ("output", int(graph.outputs[0])),
    ]
    operators = dependency_ops(graph, [index for _, index in output_specs], {input_index})
    diagnostic, diagnostic_payloads = subset_model(
        helper,
        model,
        payloads,
        operators,
        [("block_31", input_index)],
        output_specs,
    )
    helper.write_part(output, diagnostic, diagnostic_payloads)
    return {
        "kind": "encoder-final-layernorm",
        "input_tensor": "encoder/layers.31/add_1",
        "outputs": {
            public_name: tensor_name(graph.tensors[index])
            for public_name, index in output_specs
        },
    }


def build_ffn_layer_norm_repro(
    helper: object,
    source: object,
    block: int,
    frames: int,
    output: Path,
    stats_only: bool = False,
) -> dict:
    model, payloads = helper.make_part(source, 0, "encode")
    graph = model.subgraphs[0]
    names = {tensor_name(tensor): index for index, tensor in enumerate(graph.tensors)}
    input_index = names[stage_tensor_name(block, "attention_residual")]
    output_specs = [
        ("mean", names[stage_tensor_name(block, "ffn_ln_mean")]),
        ("variance", names[stage_tensor_name(block, "ffn_ln_variance")]),
        ("rsqrt", names[stage_tensor_name(block, "ffn_ln_rsqrt")]),
    ]
    if not stats_only:
        output_specs.append(("output", names[stage_tensor_name(block, "ffn_ln_output")]))
    operators = dependency_ops(graph, [index for _, index in output_specs], {input_index})
    diagnostic, diagnostic_payloads = subset_model(
        helper,
        model,
        payloads,
        operators,
        [("input", input_index)],
        output_specs,
    )
    for tensor in diagnostic.subgraphs[0].tensors:
        shape = [int(value) for value in tensor.shape]
        if len(shape) >= 2 and shape[0] == 1 and shape[1] == 1500:
            shape[1] = frames
            tensor.shape = shape
        if tensor.shapeSignature is not None:
            signature = [int(value) for value in tensor.shapeSignature]
            if len(signature) >= 2 and signature[0] == 1 and signature[1] == 1500:
                signature[1] = frames
                tensor.shapeSignature = signature
    helper.write_part(output, diagnostic, diagnostic_payloads)
    return {
        "kind": "ffn-layernorm-overflow-repro",
        "block": block,
        "frames": frames,
        "stats_only": stats_only,
        "input_tensor": tensor_name(graph.tensors[input_index]),
        "outputs": {
            public_name: tensor_name(graph.tensors[index])
            for public_name, index in output_specs
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--block", type=int, choices=range(24, 32), required=True)
    stage_parser.add_argument("--stage", choices=tuple(STAGE_SUFFIXES), required=True)
    stage_group_parser = subparsers.add_parser("stage-group")
    stage_group_parser.add_argument("--block", type=int, choices=range(24, 32), required=True)
    stage_group_parser.add_argument(
        "--stages", nargs="+", choices=tuple(STAGE_SUFFIXES), required=True
    )
    subparsers.add_parser("final-layernorm")
    repro_parser = subparsers.add_parser("ffn-layernorm-repro")
    repro_parser.add_argument("--block", type=int, choices=range(24, 32), default=31)
    repro_parser.add_argument("--frames", type=int, choices=range(1, 1501), default=6)
    repro_parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    source_path = args.source.resolve()
    if not source_path.is_file() or sha256(source_path) != SOURCE_SHA256:
        raise RuntimeError("Original FP32 encoder is missing or has the wrong SHA-256")
    default_output = {
        "stage": DEFAULT_STAGE_OUTPUT,
        "stage-group": DEFAULT_STAGE_OUTPUT,
        "final-layernorm": DEFAULT_LAYER_NORM_OUTPUT,
        "ffn-layernorm-repro": DEFAULT_FFN_LAYER_NORM_REPRO_OUTPUT,
    }[args.mode]
    output = (args.output or default_output).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    helper = load_split_helpers()
    source = helper.OpenModel.open(source_path)
    try:
        if args.mode == "stage":
            inspection = build_stage(helper, source, args.block, args.stage, output)
        elif args.mode == "stage-group":
            inspection = build_stage_group(helper, source, args.block, args.stages, output)
        elif args.mode == "ffn-layernorm-repro":
            inspection = build_ffn_layer_norm_repro(
                helper, source, args.block, args.frames, output, args.stats_only
            )
        else:
            inspection = build_final_layer_norm(helper, source, output)
    finally:
        source.close()
    inspection.update(
        {
            "source": str(source_path),
            "source_sha256": SOURCE_SHA256,
            "output": str(output),
            "output_sha256": sha256(output),
            "bytes": output.stat().st_size,
            "signature": "diagnostic",
        }
    )
    output.with_suffix(".inspection.json").write_text(
        json.dumps(inspection, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inspection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
