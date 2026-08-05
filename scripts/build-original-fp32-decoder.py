#!/usr/bin/env python3
"""Export a true-FP32 decoder from the pinned original OpenAI checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import shutil
import struct
import sys
from typing import Any, Sequence

import numpy as np


os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_SCRIPT = PROJECT_ROOT / "scripts/run-original-fp16-gpu-experiment.py"
MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


class ExportError(RuntimeError):
    pass


def load_shared() -> Any:
    spec = importlib.util.spec_from_file_location("original_fp16_experiment", SHARED_SCRIPT)
    if spec is None or spec.loader is None:
        raise ExportError(f"Cannot load shared exporter helpers from {SHARED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split_helpers() -> Any:
    path = PROJECT_ROOT / "scripts/split-whisper-encode-decode-model.py"
    spec = importlib.util.spec_from_file_location("split_whisper_model", path)
    if spec is None or spec.loader is None:
        raise ExportError(f"Cannot load external-buffer helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_decoder(path: Path) -> dict[str, Any]:
    from tensorflow.lite.python import schema_py_generated as schema

    shared = load_shared().inspect_tflite(path)
    if shared["signatures"] != ["decode"] or len(shared["subgraphs"]) != 1:
        raise ExportError(f"Expected one decode signature, found {shared['signatures']}")
    signature = shared["signature_contracts"]["decode"]
    expected_inputs = {
        "args_0": (int(schema.TensorType.FLOAT32), [1, 1500, 1280]),
        "args_1": (int(schema.TensorType.INT32), [1, 128]),
        "args_2": (int(schema.TensorType.FLOAT32), [1, 1, 128, 128]),
    }
    actual_inputs = {
        item["name"]: (item["type"], item["shape"])
        for item in signature["inputs"]
    }
    actual_outputs = {
        item["name"]: (item["type"], item["shape"])
        for item in signature["outputs"]
    }
    if actual_inputs != expected_inputs:
        raise ExportError(f"Decode input contract changed: {actual_inputs}")
    expected_outputs = {"output_0": (int(schema.TensorType.FLOAT32), [1, 128, 51866])}
    if actual_outputs != expected_outputs:
        raise ExportError(f"Decode output contract changed: {actual_outputs}")

    with path.open("rb") as stream:
        data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            model = schema.Model.GetRootAsModel(data, 0)
            graph = model.Subgraphs(0)
            names = {
                value: name
                for name, value in vars(schema.TensorType).items()
                if isinstance(value, int)
            }
            constant_types: dict[str, int] = {}
            for index in range(graph.TensorsLength()):
                tensor = graph.Tensors(index)
                buffer_index = int(tensor.Buffer())
                buffer = model.Buffers(buffer_index)
                has_data = buffer_index > 0 and (
                    buffer.DataLength() > 0
                    or int(getattr(buffer, "Size", lambda: 0)()) > 0
                )
                if has_data:
                    name = names.get(int(tensor.Type()), str(int(tensor.Type())))
                    constant_types[name] = constant_types.get(name, 0) + 1
        finally:
            data.close()
    if constant_types.get("INT8", 0) or constant_types.get("FLOAT16", 0):
        raise ExportError(f"Decoder contains reduced-precision constants: {constant_types}")
    if constant_types.get("FLOAT32", 0) <= 0:
        raise ExportError(f"Decoder contains no FLOAT32 constants: {constant_types}")
    return {**shared, "constant_type_counts": constant_types}


def rewrite_decoder_attention(raw_model: Path, explicit_model: Path) -> int:
    from tensorflow.lite.python import schema_py_generated as schema

    helpers = load_split_helpers()
    source = helpers.OpenModel.open(raw_model)
    try:
        graph = schema.SubGraphT.InitFromObj(source.model.Subgraphs(0))
        model = schema.ModelT()
        model.version = int(source.model.Version())
        model.description = source.model.Description()
        model.subgraphs = [graph]
        model.operatorCodes = [
            schema.OperatorCodeT.InitFromObj(source.model.OperatorCodes(index))
            for index in range(source.model.OperatorCodesLength())
        ]
        payloads = []
        model.buffers = []
        for index in range(source.model.BuffersLength()):
            payload = helpers.payload_for(source, index)
            replacement = schema.BufferT()
            replacement.data = None
            replacement.offset = 1 if payload.size else 0
            replacement.size = payload.size
            model.buffers.append(replacement)
            payloads.append(payload)
        model.signatureDefs = [
            schema.SignatureDefT.InitFromObj(source.model.SignatureDefs(index))
            for index in range(source.model.SignatureDefsLength())
        ]

        permutation_bytes = struct.pack("<3i", 0, 2, 1)
        permutation_offset = source.mapping.find(permutation_bytes)
        if permutation_offset < 0:
            raise ExportError("Could not source the 12-byte transpose permutation payload")
        permutation_buffer = len(model.buffers)
        replacement = schema.BufferT()
        replacement.data = None
        replacement.offset = 1
        replacement.size = len(permutation_bytes)
        model.buffers.append(replacement)
        payloads.append(
            helpers.Payload(source.path, permutation_offset, len(permutation_bytes))
        )
        permutation_tensor = len(graph.tensors)
        tensor = schema.TensorT()
        tensor.shape = np.asarray([3], dtype=np.int32)
        tensor.type = int(schema.TensorType.INT32)
        tensor.buffer = permutation_buffer
        tensor.name = "voicememo_attention_key_permutation"
        tensor.isVariable = False
        tensor.hasRank = True
        graph.tensors.append(tensor)

        transpose_opcode = next(
            (
                index
                for index, code in enumerate(model.operatorCodes)
                if max(int(code.builtinCode), int(code.deprecatedBuiltinCode))
                == int(schema.BuiltinOperator.TRANSPOSE)
            ),
            None,
        )
        if transpose_opcode is None:
            raise ExportError("Raw decoder has no TRANSPOSE opcode")

        rewritten = []
        rewrite_count = 0
        for operator in graph.operators:
            code = model.operatorCodes[int(operator.opcodeIndex)]
            builtin = max(int(code.builtinCode), int(code.deprecatedBuiltinCode))
            options = operator.builtinOptions
            target = (
                builtin == int(schema.BuiltinOperator.BATCH_MATMUL)
                and isinstance(options, schema.BatchMatMulOptionsT)
                and bool(options.adjY)
            )
            if not target:
                rewritten.append(operator)
                continue
            if bool(options.adjX) or len(operator.inputs) != 2 or len(operator.outputs) != 1:
                raise ExportError("Unexpected decoder BATCH_MATMUL contract")
            key_index = int(operator.inputs[1])
            key = graph.tensors[key_index]
            key_shape = list(key.shape)
            if int(key.type) != int(schema.TensorType.FLOAT32) or len(key_shape) != 3:
                raise ExportError("Decoder attention key is not rank-3 FLOAT32")
            key_shape[-2], key_shape[-1] = key_shape[-1], key_shape[-2]
            output_index = len(graph.tensors)
            output = schema.TensorT()
            output.shape = np.asarray(key_shape, dtype=np.int32)
            output.type = int(key.type)
            output.buffer = 0
            output.name = f"{key.name}/voicememo_explicit_transpose"
            output.isVariable = False
            output.hasRank = True
            graph.tensors.append(output)

            transpose = schema.OperatorT()
            transpose.opcodeIndex = transpose_opcode
            transpose.inputs = np.asarray([key_index, permutation_tensor], dtype=np.int32)
            transpose.outputs = np.asarray([output_index], dtype=np.int32)
            transpose.builtinOptionsType = int(schema.BuiltinOptions.NONE)
            rewritten.append(transpose)

            inputs = np.asarray(operator.inputs, dtype=np.int32).copy()
            inputs[1] = output_index
            operator.inputs = inputs
            options.adjY = False
            rewritten.append(operator)
            rewrite_count += 1
        if rewrite_count != 8:
            raise ExportError(f"Expected eight decoder attention rewrites, found {rewrite_count}")
        graph.operators = rewritten
        helpers.write_part(explicit_model, model, payloads)
        return rewrite_count
    finally:
        source.close()


def export(args: argparse.Namespace) -> None:
    import tensorflow as tf
    from transformers import TFWhisperForConditionalGeneration

    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(args.output_dir).free / 1024**3
    if free_gib < args.min_free_gib:
        raise ExportError(
            f"Only {free_gib:.1f} GiB is free; decoder-only export requires "
            f"at least {args.min_free_gib} GiB."
        )
    saved_model = args.output_dir / "saved-model-fp32-decoder"
    raw_model = args.output_dir / "whisper-large-v3-turbo-original-fp32-decoder-raw.tflite"
    final_model = args.output_dir / "whisper-large-v3-turbo-original-fp32-decoder.tflite"
    for path in (saved_model, final_model):
        if path.exists():
            raise ExportError(f"Refusing to overwrite existing export path: {path}")
    if raw_model.exists() and not args.reuse_raw:
        raise ExportError(
            f"Raw decoder already exists: {raw_model}. Pass --reuse-raw only "
            "for output from this exporter's interrupted verified conversion."
        )

    weight_dtypes = ["float32"]
    if not args.reuse_raw:
        print(f"Loading {MODEL_ID}@{MODEL_REVISION} from the local cache as FLOAT32", flush=True)
        model = TFWhisperForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            from_pt=False,
            use_safetensors=True,
            local_files_only=True,
        )
        decoder = model.model.decoder
        weight_dtypes = sorted({str(weight.dtype.name) for weight in decoder.weights})
        if weight_dtypes != ["float32"]:
            raise ExportError(f"Original decoder weights are not exclusively FLOAT32: {weight_dtypes}")

    class DecoderModule(tf.Module):
        def __init__(self, whisper: Any) -> None:
            super().__init__()
            self.whisper = whisper

        @tf.function(
            input_signature=[
                tf.TensorSpec([1, 1500, 1280], tf.float32, name="args_0"),
                tf.TensorSpec([1, 128], tf.int32, name="args_1"),
                tf.TensorSpec([1, 1, 128, 128], tf.float32, name="args_2"),
            ]
        )
        def decode(self, args_0: Any, args_1: Any, args_2: Any) -> dict[str, Any]:
            target = self.whisper.model.decoder
            hidden_states = target.embed_tokens(args_1) * target.embed_scale
            hidden_states = hidden_states + target.embed_positions(args_1)
            for layer in target.decoder_layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=args_2,
                    encoder_hidden_states=args_0,
                    encoder_attention_mask=None,
                    layer_head_mask=None,
                    cross_attn_layer_head_mask=None,
                    past_key_value=None,
                    training=False,
                )[0]
            hidden_states = target.layer_norm(hidden_states)
            logits = tf.matmul(
                hidden_states,
                self.whisper.get_output_embeddings().weights,
                transpose_b=True,
            )
            return {"output_0": tf.ensure_shape(logits, [1, 128, 51866])}

    if not args.reuse_raw:
        module = DecoderModule(model)
        original_matmul = tf.matmul

        def explicit_key_matmul(
            a: Any,
            b: Any,
            transpose_a: bool = False,
            transpose_b: bool = False,
            adjoint_a: bool = False,
            adjoint_b: bool = False,
            a_is_sparse: bool = False,
            b_is_sparse: bool = False,
            output_type: Any = None,
            grad_a: bool = False,
            grad_b: bool = False,
            name: str | None = None,
        ) -> Any:
            if transpose_b:
                b = tf.convert_to_tensor(b)
                rank = b.shape.rank
                if rank is None or rank < 2:
                    raise ExportError("transpose_b matmul operand has unknown rank")
                permutation = list(range(rank))
                permutation[-2], permutation[-1] = permutation[-1], permutation[-2]
                b = tf.transpose(b, permutation)
                transpose_b = False
            return original_matmul(
                a,
                b,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                adjoint_a=adjoint_a,
                adjoint_b=adjoint_b,
                a_is_sparse=a_is_sparse,
                b_is_sparse=b_is_sparse,
                output_type=output_type,
                grad_a=grad_a,
                grad_b=grad_b,
                name=name,
            )

        tf.matmul = explicit_key_matmul
        try:
            concrete = module.decode.get_concrete_function()
        finally:
            tf.matmul = original_matmul
        print("Saving decoder-only FLOAT32 SavedModel", flush=True)
        tf.saved_model.save(module, str(saved_model), signatures={"decode": concrete})
        del model

        print("Converting without quantization or FLOAT16 storage", flush=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model), signature_keys=["decode"])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.experimental_enable_resource_variables = True
        converted = converter.convert()
        raw_model.write_bytes(converted)
        del converted
        shutil.rmtree(saved_model)
    else:
        print(f"Reusing interrupted exporter output: {raw_model}", flush=True)

    shared = load_shared()
    raw = inspect_decoder(raw_model)
    if raw["rewrite_counts"] != [8] or raw["mask_rank_barriers"] != 0:
        raise ExportError(
            f"Unexpected raw decoder graph: rewrites={raw['rewrite_counts']}, "
            f"mask barriers={raw['mask_rank_barriers']}"
        )
    if not list(shared.java_tool_classpath().glob("*.jar")):
        shared.build_rewriter()
    raw_sha = sha256(raw_model)
    explicit_model = args.output_dir / "whisper-large-v3-turbo-original-fp32-decoder-explicit-key.tflite"
    if explicit_model.exists():
        raise ExportError(f"Refusing to overwrite existing export path: {explicit_model}")
    rewrite_decoder_attention(raw_model, explicit_model)
    explicit_inspection = inspect_decoder(explicit_model)
    if explicit_inspection["remaining_adj_y_true"] != 0:
        raise ExportError("Explicit-key decoder still contains adj_y=true")
    explicit_sha = sha256(explicit_model)
    shared.java_rewrite([
        "generate-mask-rank-reshapes-verified",
        str(explicit_model),
        explicit_sha,
        str(final_model),
    ])
    final = inspect_decoder(final_model)
    if final["remaining_adj_y_true"] != 0 or final["mask_rank_barriers"] != 4:
        raise ExportError("Final decoder is missing required GPU graph fixes")
    raw_model.unlink()
    explicit_model.unlink()

    checkpoint = (
        Path.home()
        / ".cache/huggingface/hub/models--openai--whisper-large-v3-turbo"
        / f"snapshots/{MODEL_REVISION}/model.safetensors"
    ).resolve()
    report = {
        "kind": "original-openai-fp32-weight-decoder-litert-gpu-fixed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_decoder_weight_dtypes": weight_dtypes,
        "conversion_optimizations": [],
        "sha256": sha256(final_model),
        "bytes": final_model.stat().st_size,
        **final,
    }
    write_json(final_model.with_suffix(".inspection.json"), report)
    print(json.dumps({"model": str(final_model), "sha256": report["sha256"], "bytes": report["bytes"]}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "build/experiments/original-fp32-decoder",
    )
    result.add_argument("--min-free-gib", type=int, default=7)
    result.add_argument("--reuse-raw", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    export(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as error:
        print(f"FP32 decoder export failed: {error}", file=sys.stderr)
        raise SystemExit(2)
