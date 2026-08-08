#!/usr/bin/env python3
"""Rewrite Whisper encoder LayerNorm variance math for safe FP16 execution.

For each encoder LayerNorm, this transforms

    variance = MEAN(SQUARED_DIFFERENCE(x, MEAN(x)))

into the algebraically equivalent

    delta = x - MEAN(x)
    scaled_delta = delta * 0.25
    scaled_variance = MEAN(scaled_delta * scaled_delta)
    variance = scaled_variance * 16.0

The power-of-two scale is exact for normal binary floating-point values. It
keeps both individual squares and a potential FP16 reduction sum below the
FP16 finite limit for the observed Whisper encoder activation range. Epsilon
is still added after variance is restored to its original scale.
"""

from __future__ import annotations

import argparse
from array import array
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/split-whisper-encode-decode-model.py"
DEFAULT_INPUT = (
    ROOT
    / "build/experiments/original-fp32-encoder/"
    "whisper-large-v3-turbo-original-fp32-encoder.tflite"
)
DEFAULT_OUTPUT = (
    ROOT
    / "build/experiments/fp16-safe-layernorm-encoder/"
    "whisper-large-v3-turbo-fp16-safe-layernorm-encoder.tflite"
)
PINNED_INPUT_SHA256 = "01be8e811608cc9eb3780408f8e020509d216a1d5e280320232b2c0a13d5e353"
SCALE = 0.25
RESTORE = 16.0
EXPECTED_LAYER_NORMS = 65


class RewriteError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RewriteError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_helpers():
    spec = importlib.util.spec_from_file_location("split_whisper_model", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RewriteError(f"Could not load {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_name(tensor: object) -> str:
    value = tensor.name
    return value.decode() if isinstance(value, bytes) else str(value)


def builtin_code(model: object, operator: object) -> int:
    code = model.operatorCodes[int(operator.opcodeIndex)]
    return max(int(code.builtinCode), int(code.deprecatedBuiltinCode))


def clone_operator(template: object, inputs: list[int], outputs: list[int]) -> object:
    operator = copy.deepcopy(template)
    operator.inputs = array("i", inputs)
    operator.outputs = array("i", outputs)
    operator.intermediates = array("i")
    return operator


def binary_template(model: object, schema: object, opcode: int) -> object:
    for operator in model.subgraphs[0].operators:
        if builtin_code(model, operator) == opcode:
            return operator

    code = schema.OperatorCodeT()
    code.deprecatedBuiltinCode = opcode
    code.builtinCode = opcode
    code.version = 1
    model.operatorCodes.append(code)

    operator = schema.OperatorT()
    operator.opcodeIndex = len(model.operatorCodes) - 1
    operator.inputs = array("i")
    operator.outputs = array("i")
    operator.intermediates = array("i")
    operator.debugMetadataIndex = -1
    if opcode == schema.BuiltinOperator.SUB:
        operator.builtinOptionsType = schema.BuiltinOptions.SubOptions
        operator.builtinOptions = schema.SubOptionsT()
    elif opcode == schema.BuiltinOperator.MUL:
        operator.builtinOptionsType = schema.BuiltinOptions.MulOptions
        operator.builtinOptions = schema.MulOptionsT()
    else:
        raise RewriteError(f"No binary operator template builder for opcode {opcode}.")
    return operator


def clone_activation_tensor(template: object, name: str) -> object:
    tensor = copy.deepcopy(template)
    tensor.name = name.encode()
    tensor.buffer = 0
    tensor.isVariable = False
    return tensor


def append_float_constant(
    helper: object,
    model: object,
    payloads: list[object],
    scalar_template: object,
    name: str,
    value: float,
) -> int:
    buffer = helper.schema.BufferT()
    buffer.data = bytearray(struct.pack("<f", value))
    buffer.offset = 0
    buffer.size = 0
    model.buffers.append(buffer)
    payloads.append(helper.Payload(None, 0, 0))

    tensor = copy.deepcopy(scalar_template)
    tensor.name = name.encode()
    tensor.buffer = len(model.buffers) - 1
    tensor.isVariable = False
    model.subgraphs[0].tensors.append(tensor)
    return len(model.subgraphs[0].tensors) - 1


def rewrite_model(
    helper: object,
    model: object,
    payloads: list[object],
    expected_count: int,
) -> list[dict[str, object]]:
    schema = helper.schema
    graph = model.subgraphs[0]
    names = [tensor_name(tensor) for tensor in graph.tensors]
    producer = {
        int(tensor): index
        for index, operator in enumerate(graph.operators)
        for tensor in operator.outputs
        if int(tensor) >= 0
    }
    consumers: dict[int, list[int]] = {}
    for index, operator in enumerate(graph.operators):
        for tensor in operator.inputs:
            if int(tensor) >= 0:
                consumers.setdefault(int(tensor), []).append(index)

    templates = {
        schema.BuiltinOperator.SUB: binary_template(
            model, schema, schema.BuiltinOperator.SUB
        ),
        schema.BuiltinOperator.MUL: binary_template(
            model, schema, schema.BuiltinOperator.MUL
        ),
    }

    scalar_template = next(
        (
            tensor
            for tensor in graph.tensors
            if len(tensor.shape) == 0 and int(tensor.type) == schema.TensorType.FLOAT32
        ),
        None,
    )
    if scalar_template is None:
        # FP16-storage graphs may hold every float constant as FLOAT16, so no
        # FLOAT32 scalar exists to clone. The scale/restore constants must
        # still be FLOAT32 because they multiply FLOAT32 activation tensors.
        scalar_template = schema.TensorT()
        scalar_template.shape = []
        scalar_template.type = int(schema.TensorType.FLOAT32)
        scalar_template.buffer = 0
        scalar_template.name = b"fp16_safe_layernorm_scalar_template"
        scalar_template.isVariable = False
        scalar_template.hasRank = True
    scale_tensor = append_float_constant(
        helper, model, payloads, scalar_template, "fp16_safe_layernorm_scale", SCALE
    )
    restore_tensor = append_float_constant(
        helper, model, payloads, scalar_template, "fp16_safe_layernorm_restore", RESTORE
    )

    candidates: dict[int, dict[str, int | str]] = {}
    variance_means: dict[int, dict[str, int | str]] = {}
    for squared_index, squared in enumerate(graph.operators):
        if builtin_code(model, squared) != schema.BuiltinOperator.SQUARED_DIFFERENCE:
            continue
        require(len(squared.inputs) == 2 and len(squared.outputs) == 1,
                f"SQUARED_DIFFERENCE {squared_index} has an unexpected contract.")
        source_tensor = int(squared.inputs[0])
        mean_tensor = int(squared.inputs[1])
        mean_index = producer.get(mean_tensor)
        require(mean_index is not None, f"LayerNorm mean producer is missing at {squared_index}.")
        mean_operator = graph.operators[mean_index]
        require(builtin_code(model, mean_operator) == schema.BuiltinOperator.MEAN,
                f"SQUARED_DIFFERENCE {squared_index} does not consume MEAN.")
        require(int(mean_operator.inputs[0]) == source_tensor,
                f"LayerNorm mean source mismatch at {squared_index}.")

        squared_tensor = int(squared.outputs[0])
        squared_consumers = consumers.get(squared_tensor, [])
        require(len(squared_consumers) == 1,
                f"Squared difference {squared_index} has {len(squared_consumers)} consumers.")
        variance_mean_index = squared_consumers[0]
        variance_mean = graph.operators[variance_mean_index]
        require(builtin_code(model, variance_mean) == schema.BuiltinOperator.MEAN,
                f"Squared difference {squared_index} is not consumed by MEAN.")
        require(int(mean_operator.inputs[1]) == int(variance_mean.inputs[1]),
                f"LayerNorm reduction axes differ at {squared_index}.")
        require(len(variance_mean.outputs) == 1,
                f"Variance MEAN {variance_mean_index} has an unexpected contract.")

        delta_tensor = len(graph.tensors)
        graph.tensors.append(
            clone_activation_tensor(
                graph.tensors[source_tensor],
                names[squared_tensor] + "/fp16_safe_delta",
            )
        )
        scaled_delta_tensor = len(graph.tensors)
        graph.tensors.append(
            clone_activation_tensor(
                graph.tensors[source_tensor],
                names[squared_tensor] + "/fp16_safe_scaled_delta",
            )
        )
        scaled_variance_tensor = len(graph.tensors)
        original_variance_tensor = int(variance_mean.outputs[0])
        graph.tensors.append(
            clone_activation_tensor(
                graph.tensors[original_variance_tensor],
                names[original_variance_tensor] + "/fp16_safe_scaled",
            )
        )
        spec: dict[str, int | str] = {
            "squared_index": squared_index,
            "variance_mean_index": variance_mean_index,
            "source_tensor": source_tensor,
            "mean_tensor": mean_tensor,
            "squared_tensor": squared_tensor,
            "original_variance_tensor": original_variance_tensor,
            "delta_tensor": delta_tensor,
            "scaled_delta_tensor": scaled_delta_tensor,
            "scaled_variance_tensor": scaled_variance_tensor,
            "name": names[original_variance_tensor],
        }
        candidates[squared_index] = spec
        require(variance_mean_index not in variance_means,
                f"Variance MEAN {variance_mean_index} is shared.")
        variance_means[variance_mean_index] = spec

    require(
        len(candidates) == expected_count,
        f"Found {len(candidates)} LayerNorm variance branches; expected {expected_count}.",
    )

    rewritten = []
    for index, operator in enumerate(graph.operators):
        if index in candidates:
            spec = candidates[index]
            rewritten.extend(
                (
                    clone_operator(
                        templates[schema.BuiltinOperator.SUB],
                        [int(spec["source_tensor"]), int(spec["mean_tensor"])],
                        [int(spec["delta_tensor"])],
                    ),
                    clone_operator(
                        templates[schema.BuiltinOperator.MUL],
                        [int(spec["delta_tensor"]), scale_tensor],
                        [int(spec["scaled_delta_tensor"])],
                    ),
                    clone_operator(
                        templates[schema.BuiltinOperator.MUL],
                        [int(spec["scaled_delta_tensor"]), int(spec["scaled_delta_tensor"])],
                        [int(spec["squared_tensor"])],
                    ),
                )
            )
        elif index in variance_means:
            spec = variance_means[index]
            scaled_mean = copy.deepcopy(operator)
            scaled_mean.outputs = array("i", [int(spec["scaled_variance_tensor"])])
            rewritten.append(scaled_mean)
            rewritten.append(
                clone_operator(
                    templates[schema.BuiltinOperator.MUL],
                    [int(spec["scaled_variance_tensor"]), restore_tensor],
                    [int(spec["original_variance_tensor"])],
                )
            )
        else:
            rewritten.append(operator)
    graph.operators = rewritten
    return [dict(spec) for spec in candidates.values()]


def inspect_output(helper: object, output: Path, expected_count: int) -> dict[str, object]:
    opened = helper.OpenModel.open(output)
    try:
        model = opened.model
        require(model.SubgraphsLength() == 1, "Output must have one subgraph.")
        graph = model.Subgraphs(0)
        opcodes = [
            max(
                int(model.OperatorCodes(i).BuiltinCode()),
                int(model.OperatorCodes(i).DeprecatedBuiltinCode()),
            )
            for i in range(model.OperatorCodesLength())
        ]
        counts: dict[int, int] = {}
        for index in range(graph.OperatorsLength()):
            code = opcodes[int(graph.Operators(index).OpcodeIndex())]
            counts[code] = counts.get(code, 0) + 1
        require(
            counts.get(helper.schema.BuiltinOperator.SQUARED_DIFFERENCE, 0) == 0,
            "Rewritten graph still contains SQUARED_DIFFERENCE.",
        )
        require(
            counts.get(helper.schema.BuiltinOperator.MEAN, 0) >= expected_count * 2,
            "Rewritten graph lost LayerNorm MEAN operators.",
        )
        return {
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "operators": graph.OperatorsLength(),
            "tensors": graph.TensorsLength(),
            "squared_difference_operators": counts.get(
                helper.schema.BuiltinOperator.SQUARED_DIFFERENCE, 0
            ),
            "sub_operators": counts.get(helper.schema.BuiltinOperator.SUB, 0),
            "mul_operators": counts.get(helper.schema.BuiltinOperator.MUL, 0),
        }
    finally:
        opened.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-layer-norms", type=int, default=EXPECTED_LAYER_NORMS)
    parser.add_argument(
        "--allow-unpinned-input",
        action="store_true",
        help="allow a non-production input hash, for the standalone repro model",
    )
    args = parser.parse_args()
    source_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(source_path.is_file(), f"Input model is missing: {source_path}")
    require("build" in output.parts, "Output must be inside a build directory.")
    require(not output.exists(), f"Refusing to replace {output}")
    source_hash = sha256(source_path)
    if not args.allow_unpinned_input:
        require(
            source_hash == PINNED_INPUT_SHA256,
            f"Input SHA-256 changed: {source_hash}",
        )

    helper = load_helpers()
    source = helper.OpenModel.open(source_path)
    try:
        require(source.model.SubgraphsLength() == 1, "Input must contain one subgraph.")
        signature = source.model.SignatureDefs(0).SignatureKey().decode()
        model, payloads = helper.make_part(source, 0, signature)
        changes = rewrite_model(helper, model, payloads, args.expected_layer_norms)
        helper.write_part(output, model, payloads)
    finally:
        source.close()

    report = {
        "kind": "whisper-encoder-fp16-safe-layernorm",
        "input": str(source_path),
        "input_sha256": source_hash,
        "output": str(output),
        "scale": SCALE,
        "restore": RESTORE,
        "layer_norms_rewritten": len(changes),
        "rewritten_variance_tensors": [change["name"] for change in changes],
        "output_inspection": inspect_output(
            helper, output, args.expected_layer_norms
        ),
    }
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RewriteError as error:
        print(f"FP16-safe LayerNorm rewrite failed: {error}", file=sys.stderr)
        raise SystemExit(2)
