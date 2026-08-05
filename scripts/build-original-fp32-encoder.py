#!/usr/bin/env python3
"""Export the pinned original Whisper encoder with true FLOAT32 weights.

This intentionally exports only the encoder.  It avoids materializing the much
larger decoder and makes the provenance distinction explicit: constants come
directly from the pinned OpenAI checkpoint, with no TFLite quantization or
FLOAT16 storage pass.  The existing explicit attention-key transpose rewrite
is then applied so the graph has the same Mali workaround as VoiceMemo's other
GPU models.
"""

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
SOURCE_SCRIPT = PROJECT_ROOT / "scripts/run-original-fp16-gpu-experiment.py"
MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


class ExportError(RuntimeError):
    pass


def load_shared() -> Any:
    spec = importlib.util.spec_from_file_location("original_fp16_experiment", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ExportError(f"Cannot load shared exporter helpers from {SOURCE_SCRIPT}")
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


def inspect_encoder(path: Path) -> dict[str, Any]:
    from tensorflow.lite.python import schema_py_generated as schema

    with path.open("rb") as stream:
        data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            model = schema.Model.GetRootAsModel(data, 0)
            if model.SubgraphsLength() != 1:
                raise ExportError(f"Expected one encoder subgraph, found {model.SubgraphsLength()}")
            graph = model.Subgraphs(0)
            type_counts: dict[str, int] = {}
            constant_type_counts: dict[str, int] = {}
            tensor_type_names = {
                value: name
                for name, value in vars(schema.TensorType).items()
                if isinstance(value, int)
            }
            for index in range(graph.TensorsLength()):
                tensor = graph.Tensors(index)
                type_name = tensor_type_names.get(int(tensor.Type()), str(int(tensor.Type())))
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                buffer_index = int(tensor.Buffer())
                buffer = model.Buffers(buffer_index)
                # TFLite stores buffers externally once a model exceeds the
                # 2 GiB FlatBuffer limit.  In that representation DataLength
                # is zero and Size/Offset identify the constant payload.
                has_data = buffer_index > 0 and (
                    buffer.DataLength() > 0
                    or int(getattr(buffer, "Size", lambda: 0)()) > 0
                )
                if has_data:
                    constant_type_counts[type_name] = constant_type_counts.get(type_name, 0) + 1
            shared = load_shared().inspect_tflite(path)
            signature = shared.get("signature_contracts", {}).get("encode")
            expected = {
                "inputs": [{"name": "args_0", "type": int(schema.TensorType.FLOAT32), "shape": [1, 128, 3000]}],
                "outputs": [{"name": "output_0", "type": int(schema.TensorType.FLOAT32), "shape": [1, 1500, 1280]}],
            }
            if signature is None:
                raise ExportError("Missing encode signature")
            for direction, contract in expected.items():
                actual = [
                    {"name": item["name"], "type": item["type"], "shape": item["shape"]}
                    for item in signature[direction]
                ]
                if actual != contract:
                    raise ExportError(f"Encode {direction} contract changed: {actual}")
            if constant_type_counts.get("INT8", 0) != 0:
                raise ExportError("Encoder unexpectedly contains INT8 constants")
            if constant_type_counts.get("FLOAT16", 0) != 0:
                raise ExportError("Encoder unexpectedly contains FLOAT16 constants")
            if constant_type_counts.get("FLOAT32", 0) <= 0:
                raise ExportError("Encoder contains no FLOAT32 constants")
            return {
                **shared,
                "tensor_type_counts": type_counts,
                "constant_type_counts": constant_type_counts,
            }
        finally:
            data.close()


def rewrite_external_encoder(raw_model: Path, final_model: Path) -> int:
    """Rewrite adj_y score matmuls while streaming >2 GiB weight buffers."""
    from tensorflow.lite.python import schema_py_generated as schema

    helpers = load_split_helpers()
    source = helpers.OpenModel.open(raw_model)
    try:
        payloads = [
            helpers.payload_for(source, index)
            for index in range(source.model.BuffersLength())
        ]
        model = schema.ModelT.InitFromObj(source.model)
        graph = model.subgraphs[0]
        transpose_opcode = next(
            (
                index
                for index in range(source.model.OperatorCodesLength())
                if max(
                    int(source.model.OperatorCodes(index).BuiltinCode()),
                    int(source.model.OperatorCodes(index).DeprecatedBuiltinCode()),
                )
                == int(schema.BuiltinOperator.TRANSPOSE)
            ),
            None,
        )
        if transpose_opcode is None:
            raise ExportError("Raw encoder has no TRANSPOSE opcode")

        permutation_tensor = None
        for index, tensor in enumerate(graph.tensors):
            if int(tensor.type) != int(schema.TensorType.INT32) or list(tensor.shape) != [3]:
                continue
            payload = payloads[int(tensor.buffer)]
            if payload.size != 12 or payload.source is None:
                continue
            with payload.source.open("rb") as stream:
                stream.seek(payload.offset)
                values = struct.unpack("<3i", stream.read(12))
            if values == (0, 2, 1):
                permutation_tensor = index
                break
        if permutation_tensor is None:
            raise ExportError("Raw encoder has no [0,2,1] transpose permutation tensor")

        rewritten: list[Any] = []
        rewrite_count = 0
        for operator in graph.operators:
            code = source.model.OperatorCodes(int(operator.opcodeIndex))
            builtin = max(int(code.BuiltinCode()), int(code.DeprecatedBuiltinCode()))
            options = operator.builtinOptions
            is_target = (
                builtin == int(schema.BuiltinOperator.BATCH_MATMUL)
                and isinstance(options, schema.BatchMatMulOptionsT)
                and bool(options.adjY)
            )
            if not is_target:
                rewritten.append(operator)
                continue
            if bool(options.adjX) or len(operator.inputs) != 2 or len(operator.outputs) != 1:
                raise ExportError("Unexpected encoder BATCH_MATMUL contract")
            key_index = int(operator.inputs[1])
            key = graph.tensors[key_index]
            key_shape = list(key.shape)
            if int(key.type) != int(schema.TensorType.FLOAT32) or len(key_shape) != 3:
                raise ExportError("Attention key is not rank-3 FLOAT32")
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

        if rewrite_count != 32:
            raise ExportError(f"Expected 32 encoder attention rewrites, found {rewrite_count}")
        graph.operators = rewritten
        helpers.write_part(final_model, model, payloads)
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
            f"Only {free_gib:.1f} GiB is free; encoder-only export requires at least "
            f"{args.min_free_gib} GiB."
        )
    saved_model = args.output_dir / "saved-model-fp32-encoder"
    raw_model = args.output_dir / "whisper-large-v3-turbo-original-fp32-encoder-raw.tflite"
    final_model = args.output_dir / "whisper-large-v3-turbo-original-fp32-encoder.tflite"
    guarded_paths = (saved_model, final_model)
    for path in guarded_paths:
        if path.exists():
            raise ExportError(f"Refusing to overwrite existing export path: {path}")

    if raw_model.exists() and not args.reuse_raw:
        raise ExportError(
            f"Raw model already exists: {raw_model}. Pass --reuse-raw only when "
            "it came from this exporter's interrupted verified conversion."
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
        encoder = model.model.encoder
        weight_dtypes = sorted({str(weight.dtype.name) for weight in encoder.weights})
        if weight_dtypes != ["float32"]:
            raise ExportError(f"Original encoder weight dtypes are not exclusively FLOAT32: {weight_dtypes}")

    class EncoderModule(tf.Module):
        def __init__(self, whisper_encoder: Any) -> None:
            super().__init__()
            self.encoder = whisper_encoder

        @tf.function(input_signature=[tf.TensorSpec([1, 128, 3000], tf.float32, name="args_0")])
        def encode(self, args_0: Any) -> dict[str, Any]:
            output = self.encoder(
                input_features=args_0,
                training=False,
                return_dict=True,
            ).last_hidden_state
            return {"output_0": tf.ensure_shape(output, [1, 1500, 1280])}

    if not args.reuse_raw:
        module = EncoderModule(encoder)
        concrete = module.encode.get_concrete_function()
        print("Saving encoder-only FLOAT32 SavedModel", flush=True)
        tf.saved_model.save(module, str(saved_model), signatures={"encode": concrete})
        del model

        print("Converting without quantization or FLOAT16 storage", flush=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model), signature_keys=["encode"])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.experimental_enable_resource_variables = True
        converted = converter.convert()
        raw_model.write_bytes(converted)
        del converted
        shutil.rmtree(saved_model)
    else:
        print(f"Reusing interrupted exporter output: {raw_model}", flush=True)

    raw_inspection = inspect_encoder(raw_model)
    counts = raw_inspection["rewrite_counts"]
    if sum(counts) <= 0:
        raise ExportError("Raw encoder has no adj_y=true attention operations to rewrite")
    raw_sha = sha256(raw_model)
    rewrite_count = rewrite_external_encoder(raw_model, final_model)
    final_inspection = inspect_encoder(final_model)
    if final_inspection["remaining_adj_y_true"] != 0:
        raise ExportError("Rewritten encoder still contains adj_y=true")
    if final_inspection["subgraphs"][0]["operators"] != raw_inspection["subgraphs"][0]["operators"] + rewrite_count:
        raise ExportError("Rewritten encoder operator count changed unexpectedly")
    raw_model.unlink()

    checkpoint = (
        Path.home()
        / ".cache/huggingface/hub/models--openai--whisper-large-v3-turbo"
        / f"snapshots/{MODEL_REVISION}/model.safetensors"
    ).resolve()
    report = {
        "kind": "original-openai-fp32-weight-encoder-litert-gpu-fixed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_encoder_weight_dtypes": weight_dtypes,
        "conversion_optimizations": [],
        "sha256": sha256(final_model),
        "bytes": final_model.stat().st_size,
        **final_inspection,
    }
    write_json(final_model.with_suffix(".inspection.json"), report)
    print(json.dumps({"model": str(final_model), "sha256": report["sha256"], "bytes": report["bytes"]}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "build/experiments/original-fp32-encoder",
    )
    result.add_argument("--min-free-gib", type=int, default=8)
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
        print(f"FP32 encoder export failed: {error}", file=sys.stderr)
        raise SystemExit(2)
