#!/usr/bin/env python3
"""Combine the pinned LiteRT INT8 encoder with the original FP16 decoder.

The two source models expose matching ``decode`` and ``encode`` signatures in
subgraphs 0 and 1. This tool retains the GPU-fixed decoder graph and constants
from the direct original-checkpoint FP16 export, retains the production INT8
encoder graph and constants, and writes a new model using external buffers.

Large payloads are streamed from the source files. They are never collected in
Python memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import BinaryIO

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


FP16_DECODER_SHA256 = (
    "fd50a74a3d0f47fcd8e11c3bdd167fe051b3f0f160c77c162b802881914710f3"
)
INT8_ENCODER_SHA256 = (
    "8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418"
)
DECODE_SUBGRAPH = 0
ENCODE_SUBGRAPH = 1
ALIGNMENT = 16
COPY_BLOCK_BYTES = 8 * 1024 * 1024


class BuildError(RuntimeError):
    """A fail-closed model-combination error."""


@dataclass
class Payload:
    source: Path | None
    offset: int
    size: int
    source_size: int
    transform: str = "copy"


@dataclass
class OpenModel:
    path: Path
    stream: BinaryIO
    mapping: mmap.mmap
    model: object

    @classmethod
    def open(cls, path: Path) -> "OpenModel":
        stream = path.open("rb")
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        return cls(
            path=path,
            stream=stream,
            mapping=mapping,
            model=schema.Model.GetRootAsModel(mapping, 0),
        )

    def close(self) -> None:
        try:
            self.mapping.close()
        except BufferError:
            # Generated FlatBuffer object APIs retain zero-copy NumPy views.
            # The process releases these mappings after serialization.
            pass
        self.stream.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(COPY_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def has_build_segment(path: Path) -> bool:
    return "build" in path.resolve().parts


def buffer_payload(source: OpenModel, index: int) -> Payload:
    buffer = source.model.Buffers(index)
    inline_size = int(buffer.DataLength())
    external_offset = int(buffer.Offset())
    external_size = int(buffer.Size())
    require(
        not (inline_size and external_size),
        f"{source.path.name} buffer {index} has inline and external data.",
    )
    if external_size:
        offset = external_offset
        size = external_size
    elif inline_size:
        field = buffer._tab.Offset(4)  # FlatBuffer Buffer.data slot.
        require(field != 0, f"{source.path.name} buffer {index} data is missing.")
        offset = int(buffer._tab.Vector(field))
        size = inline_size
    else:
        return Payload(None, 0, 0, 0)
    require(
        offset >= 0 and size >= 0 and offset + size <= len(source.mapping),
        f"{source.path.name} buffer {index} points outside its file.",
    )
    return Payload(source.path, offset, size, size)


def signature(source: OpenModel, key: bytes) -> object:
    for index in range(source.model.SignatureDefsLength()):
        candidate = source.model.SignatureDefs(index)
        if candidate.SignatureKey() == key:
            return schema.SignatureDefT.InitFromObj(candidate)
    raise BuildError(f"{source.path.name} is missing signature {key!r}.")


def pack_model(model: object) -> bytes:
    builder = flatbuffers.Builder(4 * 1024 * 1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytes(builder.Output())


def builtin_code(source: OpenModel, opcode_index: int) -> int:
    opcode = source.model.OperatorCodes(opcode_index)
    return max(int(opcode.BuiltinCode()), int(opcode.DeprecatedBuiltinCode()))


def fold_fp16_decoder_constants(
    decode: object,
    source: OpenModel,
) -> dict[int, int]:
    input_uses: dict[int, int] = {}
    for operator in decode.operators:
        for tensor_index in operator.inputs:
            if tensor_index >= 0:
                input_uses[int(tensor_index)] = (
                    input_uses.get(int(tensor_index), 0) + 1
                )

    expanded_outputs: dict[int, int] = {}
    cleared_inputs: set[int] = set()
    retained_operators = []
    for operator in decode.operators:
        if (
            builtin_code(source, int(operator.opcodeIndex))
            != schema.BuiltinOperator.DEQUANTIZE
        ):
            retained_operators.append(operator)
            continue
        require(
            len(operator.inputs) == 1 and len(operator.outputs) == 1,
            "Decoder DEQUANTIZE arity changed.",
        )
        input_index = int(operator.inputs[0])
        output_index = int(operator.outputs[0])
        input_tensor = decode.tensors[input_index]
        output_tensor = decode.tensors[output_index]
        require(
            input_tensor.type == schema.TensorType.FLOAT16,
            "Decoder DEQUANTIZE input is not FLOAT16.",
        )
        require(
            output_tensor.type == schema.TensorType.FLOAT32,
            "Decoder DEQUANTIZE output is not FLOAT32.",
        )
        require(
            input_uses.get(input_index) == 1,
            "FLOAT16 constant has consumers beyond its DEQUANTIZE.",
        )
        output_payload = buffer_payload(source, int(output_tensor.buffer))
        require(
            output_payload.size == 0,
            "Decoder DEQUANTIZE output unexpectedly contains constant data.",
        )
        source_buffer = int(input_tensor.buffer)
        payload = buffer_payload(source, source_buffer)
        require(
            payload.size > 0 and payload.size % 2 == 0,
            "Decoder FLOAT16 constant payload is invalid.",
        )
        require(
            output_index not in expanded_outputs,
            "Decoder DEQUANTIZE output is duplicated.",
        )
        expanded_outputs[output_index] = source_buffer
        cleared_inputs.add(input_index)

    require(
        len(expanded_outputs) == 103,
        f"Expected 103 decoder constant folds, found {len(expanded_outputs)}.",
    )
    decode.operators = retained_operators
    for tensor_index in cleared_inputs:
        decode.tensors[tensor_index].buffer = 0
    return expanded_outputs


def combine(
    fp16: OpenModel,
    int8: OpenModel,
    *,
    expand_decoder_to_fp32: bool,
) -> tuple[object, list[Payload], int]:
    for source, label in ((fp16, "FP16"), (int8, "INT8")):
        require(source.model.Version() == 3, f"{label} model version changed.")
        require(
            source.model.SubgraphsLength() == 2,
            f"{label} model must contain decode and encode subgraphs.",
        )

    decode = schema.SubGraphT.InitFromObj(
        fp16.model.Subgraphs(DECODE_SUBGRAPH)
    )
    encode = schema.SubGraphT.InitFromObj(
        int8.model.Subgraphs(ENCODE_SUBGRAPH)
    )
    require(
        len(decode.tensors) == 556 and len(decode.operators) == 441,
        "FP16 decoder graph contract changed.",
    )
    require(
        len(encode.tensors) == 1997 and len(encode.operators) == 1494,
        "INT8 encoder graph contract changed.",
    )
    expanded_outputs = (
        fold_fp16_decoder_constants(decode, fp16)
        if expand_decoder_to_fp32
        else {}
    )

    combined = schema.ModelT()
    combined.version = 3
    combined.description = (
        b"VoiceMemo Whisper Large v3 Turbo: INT8 encoder plus original "
        + (
            b"FP16 decoder expanded to FP32 constants offline"
            if expand_decoder_to_fp32
            else b"FP16 decoder with FP32 interfaces"
        )
    )
    combined.subgraphs = [decode, encode]
    combined.operatorCodes = []
    combined.buffers = [schema.BufferT()]
    payloads = [Payload(None, 0, 0, 0)]

    def remap_opcodes(graph: object, source: OpenModel) -> None:
        mapping: dict[int, int] = {}
        for operator in graph.operators:
            old = int(operator.opcodeIndex)
            if old not in mapping:
                mapping[old] = len(combined.operatorCodes)
                combined.operatorCodes.append(
                    schema.OperatorCodeT.InitFromObj(
                        source.model.OperatorCodes(old)
                    )
                )
            operator.opcodeIndex = mapping[old]

    def remap_buffers(
        graph: object,
        source: OpenModel,
        expanded: dict[int, int] | None = None,
    ) -> None:
        mapping: dict[int, int] = {0: 0}
        for tensor_index, tensor in enumerate(graph.tensors):
            if expanded is not None and tensor_index in expanded:
                original = buffer_payload(source, expanded[tensor_index])
                payload = Payload(
                    source=original.source,
                    offset=original.offset,
                    size=original.size * 2,
                    source_size=original.size,
                    transform="fp16_to_fp32",
                )
                replacement = schema.BufferT()
                replacement.data = None
                replacement.offset = 1
                replacement.size = payload.size
                tensor.buffer = len(combined.buffers)
                combined.buffers.append(replacement)
                payloads.append(payload)
                continue
            old = int(tensor.buffer)
            if old not in mapping:
                payload = buffer_payload(source, old)
                replacement = schema.BufferT()
                replacement.data = None
                replacement.offset = 1 if payload.size else 0
                replacement.size = payload.size
                mapping[old] = len(combined.buffers)
                combined.buffers.append(replacement)
                payloads.append(payload)
            tensor.buffer = mapping[old]

    remap_opcodes(decode, fp16)
    remap_opcodes(encode, int8)
    remap_buffers(decode, fp16, expanded_outputs)
    remap_buffers(encode, int8)

    decode_signature = signature(fp16, b"decode")
    decode_signature.subgraphIndex = DECODE_SUBGRAPH
    encode_signature = signature(int8, b"encode")
    encode_signature.subgraphIndex = ENCODE_SUBGRAPH
    combined.signatureDefs = [decode_signature, encode_signature]
    combined.metadataBuffer = None
    combined.metadata = None
    return combined, payloads, len(expanded_outputs)


def assign_offsets(model: object, payloads: list[Payload]) -> bytes:
    header = pack_model(model)
    previous_state: tuple[int, tuple[int, ...]] | None = None
    for _ in range(4):
        position = align(len(header))
        offsets: list[int] = []
        for buffer, payload in zip(model.buffers, payloads, strict=True):
            if payload.size:
                buffer.offset = position
                buffer.size = payload.size
                offsets.append(position)
                position = align(position + payload.size)
            else:
                buffer.offset = 0
                buffer.size = 0
                offsets.append(0)
        header = pack_model(model)
        state = (len(header), tuple(offsets))
        if state == previous_state:
            return header
        previous_state = state
    raise BuildError("External buffer offsets did not converge.")


def write_model(
    output: Path,
    header: bytes,
    model: object,
    payloads: list[Payload],
) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    require(not output.exists(), f"Refusing to replace existing output: {output}")
    temporary.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources: dict[Path, BinaryIO] = {}
    try:
        with temporary.open("wb") as destination:
            destination.write(header)
            for buffer, payload in zip(model.buffers, payloads, strict=True):
                if not payload.size:
                    continue
                require(payload.source is not None, "Payload source is missing.")
                destination.seek(int(buffer.offset))
                source = sources.get(payload.source)
                if source is None:
                    source = payload.source.open("rb")
                    sources[payload.source] = source
                source.seek(payload.offset)
                remaining = payload.source_size
                written = 0
                while remaining:
                    block = source.read(min(remaining, COPY_BLOCK_BYTES))
                    require(bool(block), f"Short read from {payload.source}.")
                    if payload.transform == "copy":
                        output_block = block
                    elif payload.transform == "fp16_to_fp32":
                        require(
                            len(block) % 2 == 0,
                            "FLOAT16 conversion block is not element-aligned.",
                        )
                        output_block = (
                            np.frombuffer(block, dtype="<f2")
                            .astype("<f4")
                            .tobytes()
                        )
                    else:
                        raise BuildError(
                            f"Unsupported payload transform: {payload.transform}"
                        )
                    destination.write(output_block)
                    written += len(output_block)
                    remaining -= len(block)
                require(
                    written == payload.size,
                    f"Expanded payload size changed for {payload.source}.",
                )
            final_size = max(
                len(header),
                max(
                    (
                        int(buffer.offset) + int(buffer.size)
                        for buffer in model.buffers
                    ),
                    default=0,
                ),
            )
            destination.truncate(final_size)
            destination.flush()
        temporary.replace(output)
    finally:
        for source in sources.values():
            source.close()
        temporary.unlink(missing_ok=True)


def inspect(
    output: Path,
    *,
    expanded_decoder_constants: int,
) -> dict[str, object]:
    opened = OpenModel.open(output)
    try:
        model = opened.model
        require(model.Version() == 3, "Output model version changed.")
        require(model.SubgraphsLength() == 2, "Output subgraph count changed.")
        decode = model.Subgraphs(DECODE_SUBGRAPH)
        encode = model.Subgraphs(ENCODE_SUBGRAPH)
        expected_decode_operators = (
            441 - expanded_decoder_constants
        )
        require(
            decode.TensorsLength() == 556
            and decode.OperatorsLength() == expected_decode_operators,
            "Output decoder contract changed.",
        )
        require(
            encode.TensorsLength() == 1997
            and encode.OperatorsLength() == 1494,
            "Output INT8 encoder contract changed.",
        )
        signatures = {}
        for index in range(model.SignatureDefsLength()):
            item = model.SignatureDefs(index)
            signatures[item.SignatureKey().decode()] = int(item.SubgraphIndex())
        require(
            signatures == {"decode": 0, "encode": 1},
            f"Output signatures changed: {signatures}",
        )
        external_bytes = 0
        dequantize_operators = 0
        fp32_constant_bytes = 0
        fp16_constant_bytes = 0
        for operator_index in range(decode.OperatorsLength()):
            operator = decode.Operators(operator_index)
            if (
                builtin_code(opened, int(operator.OpcodeIndex()))
                == schema.BuiltinOperator.DEQUANTIZE
            ):
                dequantize_operators += 1
        for index in range(model.BuffersLength()):
            buffer = model.Buffers(index)
            offset = int(buffer.Offset())
            size = int(buffer.Size())
            require(
                not buffer.DataLength(),
                f"Output buffer {index} unexpectedly contains inline data.",
            )
            require(
                not size or offset + size <= len(opened.mapping),
                f"Output buffer {index} points outside the file.",
            )
            external_bytes += size
        for tensor_index in range(decode.TensorsLength()):
            tensor = decode.Tensors(tensor_index)
            size = int(model.Buffers(tensor.Buffer()).Size())
            if tensor.Type() == schema.TensorType.FLOAT32:
                fp32_constant_bytes += size
            elif tensor.Type() == schema.TensorType.FLOAT16:
                fp16_constant_bytes += size
        require(
            dequantize_operators == 0
            if expanded_decoder_constants
            else dequantize_operators == 103,
            "Output decoder DEQUANTIZE count changed.",
        )
        return {
            "kind": (
                "whisper-int8-encoder-original-fp16-decoder-expanded-fp32"
                if expanded_decoder_constants
                else "whisper-int8-encoder-original-fp16-decoder"
            ),
            "fp16_decoder_source_sha256": FP16_DECODER_SHA256,
            "int8_encoder_source_sha256": INT8_ENCODER_SHA256,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "subgraphs": [
                {
                    "name": decode.Name().decode(),
                    "tensors": decode.TensorsLength(),
                    "operators": decode.OperatorsLength(),
                },
                {
                    "name": encode.Name().decode(),
                    "tensors": encode.TensorsLength(),
                    "operators": encode.OperatorsLength(),
                },
            ],
            "signatures": signatures,
            "operator_codes": model.OperatorCodesLength(),
            "buffers": model.BuffersLength(),
            "external_buffer_bytes": external_bytes,
            "decoder_dequantize_operators": dequantize_operators,
            "decoder_fp32_constant_bytes": fp32_constant_bytes,
            "decoder_fp16_constant_bytes": fp16_constant_bytes,
            "offline_expanded_decoder_constants": expanded_decoder_constants,
        }
    finally:
        opened.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-decoder-model", type=Path, required=True)
    parser.add_argument("--int8-encoder-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expand-decoder-to-fp32",
        action="store_true",
        help=(
            "expand the 103 original FP16 decoder constants to FLOAT32 "
            "offline and remove their DEQUANTIZE operators"
        ),
    )
    args = parser.parse_args()
    fp16_path = args.fp16_decoder_model.expanduser().resolve()
    int8_path = args.int8_encoder_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(has_build_segment(output), "Output must stay under a build directory.")
    require(fp16_path.is_file(), f"FP16 model is missing: {fp16_path}")
    require(int8_path.is_file(), f"INT8 model is missing: {int8_path}")
    require(
        shutil.disk_usage(output.parent).free >= 4 * 1024**3,
        "Output filesystem must have at least 4 GiB free.",
    )
    require(
        sha256(fp16_path) == FP16_DECODER_SHA256,
        "FP16 decoder source SHA-256 changed.",
    )
    require(
        sha256(int8_path) == INT8_ENCODER_SHA256,
        "INT8 encoder source SHA-256 changed.",
    )

    fp16 = OpenModel.open(fp16_path)
    int8 = OpenModel.open(int8_path)
    try:
        model, payloads, expanded_decoder_constants = combine(
            fp16,
            int8,
            expand_decoder_to_fp32=args.expand_decoder_to_fp32,
        )
        header = assign_offsets(model, payloads)
        write_model(output, header, model, payloads)
    finally:
        fp16.close()
        int8.close()
    report = inspect(
        output,
        expanded_decoder_constants=expanded_decoder_constants,
    )
    report_path = output.with_suffix(".inspection.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        print(f"Mixed Whisper model build failed: {error}", file=sys.stderr)
        raise SystemExit(2)
