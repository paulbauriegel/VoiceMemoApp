#!/usr/bin/env python3
"""Split the pinned fast Whisper model into graph-identical encode/decode files."""

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
try:
    from tensorflow.lite.python import schema_py_generated as schema
except ModuleNotFoundError:
    from ai_edge_litert import schema_py_generated as schema


SOURCE_SHA256 = "a2613d64e6cb73ff40db7c548215fd086f096aa088c6cc113c833ccc87501e89"
ALIGNMENT = 16
COPY_BLOCK_BYTES = 8 * 1024 * 1024


class SplitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Payload:
    source: Path | None
    offset: int
    size: int


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
        return cls(path, stream, mapping, schema.Model.GetRootAsModel(mapping, 0))

    def close(self) -> None:
        try:
            self.mapping.close()
        except BufferError:
            pass
        self.stream.close()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(COPY_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def payload_for(source: OpenModel, buffer_index: int) -> Payload:
    buffer = source.model.Buffers(buffer_index)
    inline_size = int(buffer.DataLength())
    external_offset = int(buffer.Offset())
    external_size = int(buffer.Size())
    require(
        not (inline_size and external_size),
        f"Buffer {buffer_index} has inline and external payloads.",
    )
    if external_size:
        offset, size = external_offset, external_size
    elif inline_size:
        field = buffer._tab.Offset(4)
        require(field != 0, f"Buffer {buffer_index} data is missing.")
        offset, size = int(buffer._tab.Vector(field)), inline_size
    else:
        return Payload(None, 0, 0)
    require(offset + size <= len(source.mapping), f"Buffer {buffer_index} is out of range.")
    return Payload(source.path, offset, size)


def find_signature(source: OpenModel, key: bytes) -> object:
    for index in range(source.model.SignatureDefsLength()):
        candidate = source.model.SignatureDefs(index)
        if candidate.SignatureKey() == key:
            return schema.SignatureDefT.InitFromObj(candidate)
    raise SplitError(f"Missing signature {key.decode()}.")


def pack(model: object) -> bytes:
    builder = flatbuffers.Builder(4 * 1024 * 1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytes(builder.Output())


def make_part(source: OpenModel, subgraph_index: int, signature_key: str) -> tuple[object, list[Payload]]:
    graph = schema.SubGraphT.InitFromObj(source.model.Subgraphs(subgraph_index))
    output = schema.ModelT()
    output.version = int(source.model.Version())
    output.description = source.model.Description()
    output.subgraphs = [graph]
    output.operatorCodes = []
    output.buffers = [schema.BufferT()]
    payloads = [Payload(None, 0, 0)]

    opcode_mapping: dict[int, int] = {}
    for operator in graph.operators:
        old = int(operator.opcodeIndex)
        if old not in opcode_mapping:
            opcode_mapping[old] = len(output.operatorCodes)
            output.operatorCodes.append(
                schema.OperatorCodeT.InitFromObj(source.model.OperatorCodes(old))
            )
        operator.opcodeIndex = opcode_mapping[old]

    buffer_mapping: dict[int, int] = {0: 0}
    for tensor in graph.tensors:
        old = int(tensor.buffer)
        if old not in buffer_mapping:
            payload = payload_for(source, old)
            replacement = schema.BufferT()
            replacement.data = None
            replacement.offset = 1 if payload.size else 0
            replacement.size = payload.size
            buffer_mapping[old] = len(output.buffers)
            output.buffers.append(replacement)
            payloads.append(payload)
        tensor.buffer = buffer_mapping[old]

    signature = find_signature(source, signature_key.encode())
    signature.subgraphIndex = 0
    output.signatureDefs = [signature]
    output.metadataBuffer = None
    output.metadata = None
    return output, payloads


def assign_offsets(model: object, payloads: list[Payload]) -> bytes:
    require(len(model.buffers) == len(payloads), "Buffer payload count changed.")
    header = pack(model)
    previous = None
    for _ in range(5):
        position = align(len(header))
        offsets = []
        for buffer, payload in zip(model.buffers, payloads):
            if payload.size:
                buffer.offset = position
                buffer.size = payload.size
                offsets.append(position)
                position = align(position + payload.size)
            else:
                buffer.offset = 0
                buffer.size = 0
                offsets.append(0)
        header = pack(model)
        state = (len(header), tuple(offsets))
        if state == previous:
            return header
        previous = state
    raise SplitError("External buffer offsets did not converge.")


def write_part(path: Path, model: object, payloads: list[Payload]) -> None:
    require(len(model.buffers) == len(payloads), "Buffer payload count changed.")
    require(not path.exists(), f"Refusing to replace {path}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = assign_offsets(model, payloads)
    sources: dict[Path, BinaryIO] = {}
    try:
        with temporary.open("wb") as destination:
            destination.write(header)
            for buffer, payload in zip(model.buffers, payloads):
                if not payload.size:
                    continue
                require(payload.source is not None, "Payload source is missing.")
                destination.seek(int(buffer.offset))
                stream = sources.get(payload.source)
                if stream is None:
                    stream = payload.source.open("rb")
                    sources[payload.source] = stream
                stream.seek(payload.offset)
                remaining = payload.size
                while remaining:
                    block = stream.read(min(remaining, COPY_BLOCK_BYTES))
                    require(bool(block), f"Short read from {payload.source}.")
                    destination.write(block)
                    remaining -= len(block)
            final_size = max(
                len(header),
                max((int(b.offset) + int(b.size) for b in model.buffers), default=0),
            )
            destination.truncate(final_size)
            destination.flush()
        temporary.replace(path)
    finally:
        for stream in sources.values():
            stream.close()
        temporary.unlink(missing_ok=True)


def graph_contract(model: object, index: int) -> dict[str, int | str]:
    graph = model.Subgraphs(index)
    return {
        "name": graph.Name().decode(),
        "tensors": graph.TensorsLength(),
        "operators": graph.OperatorsLength(),
        "inputs": graph.InputsLength(),
        "outputs": graph.OutputsLength(),
    }


def verify_part(path: Path, source: OpenModel, source_index: int, signature_key: str) -> dict[str, object]:
    part = OpenModel.open(path)
    try:
        require(part.model.Version() == source.model.Version(), "Model version changed.")
        require(part.model.SubgraphsLength() == 1, "Split model must have one subgraph.")
        require(part.model.SignatureDefsLength() == 1, "Split model must have one signature.")
        signature = part.model.SignatureDefs(0)
        require(signature.SignatureKey() == signature_key.encode(), "Signature key changed.")
        require(signature.SubgraphIndex() == 0, "Signature subgraph index changed.")
        expected = graph_contract(source.model, source_index)
        actual = graph_contract(part.model, 0)
        require(actual == expected, f"Graph contract changed: {actual} != {expected}")
        source_graph = source.model.Subgraphs(source_index)
        part_graph = part.model.Subgraphs(0)
        for operator_index in range(source_graph.OperatorsLength()):
            original = source_graph.Operators(operator_index)
            candidate = part_graph.Operators(operator_index)
            original_code = source.model.OperatorCodes(original.OpcodeIndex())
            candidate_code = part.model.OperatorCodes(candidate.OpcodeIndex())
            require(
                max(original_code.BuiltinCode(), original_code.DeprecatedBuiltinCode())
                == max(candidate_code.BuiltinCode(), candidate_code.DeprecatedBuiltinCode()),
                f"Operator {operator_index} builtin code changed.",
            )
            require(
                list(original.InputsAsNumpy()) == list(candidate.InputsAsNumpy())
                and list(original.OutputsAsNumpy()) == list(candidate.OutputsAsNumpy()),
                f"Operator {operator_index} tensor wiring changed.",
            )
        external_bytes = sum(
            int(part.model.Buffers(i).Size()) for i in range(part.model.BuffersLength())
        )
        return {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "signature": signature_key,
            "graph": actual,
            "operator_codes": part.model.OperatorCodesLength(),
            "buffers": part.model.BuffersLength(),
            "external_buffer_bytes": external_bytes,
        }
    finally:
        part.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require("build" in output.parts, "Output must be under a build directory.")
    require(source_path.is_file(), f"Source model is missing: {source_path}")
    require(sha256(source_path) == SOURCE_SHA256, "Source model SHA-256 changed.")
    require(not output.exists() or not any(output.iterdir()), "Output directory is not empty.")
    require(shutil.disk_usage(output.parent).free >= 3 * 1024**3, "At least 3 GiB free space is required.")

    source = OpenModel.open(source_path)
    try:
        require(source.model.SubgraphsLength() == 2, "Source must have decode and encode subgraphs.")
        results = []
        for index, key, name in ((1, "encode", "encoder.tflite"), (0, "decode", "decoder.tflite")):
            model, payloads = make_part(source, index, key)
            destination = output / name
            write_part(destination, model, payloads)
            results.append(verify_part(destination, source, index, key))
        report = {"source_sha256": SOURCE_SHA256, "parts": results}
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SplitError as error:
        print(f"Whisper model split failed: {error}", file=sys.stderr)
        raise SystemExit(2)
