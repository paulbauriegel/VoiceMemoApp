#!/usr/bin/env python3
"""Original Whisper baseline -> FP16 LiteRT -> Android GPU experiment.

The model checkpoint is pinned. The generated LiteRT model stores constants as
FLOAT16 while retaining FLOAT32 inputs, activations, and outputs. The script
then applies the two VoiceMemo GPU graph workarounds:

1. explicit key TRANSPOSE + BATCH_MATMUL(adj_y=false)
2. explicit [20,128,128] -> [1,20,128,128] causal-mask score reshapes

Nothing is copied into app/src/main/generatedAssets. The candidate model and
audio are streamed into the already-installed debug app's private diagnostic
directories, and removed after the instrumentation run. `adb install -r` is
used; the app is never uninstalled, so existing recordings remain intact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import mmap
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from typing import Any, Iterable, Sequence


os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
SAMPLE_RATE = 16_000
PAYLOAD_SECONDS = 28
OVERLAP_SECONDS = 2
WINDOW_SAMPLES = PAYLOAD_SECONDS * SAMPLE_RATE
WINDOW_STEP_SAMPLES = (PAYLOAD_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE
MAX_TOKENS = 128
PACKAGE = "com.example.voicememo"
TEST_PACKAGE = f"{PACKAGE}.test"
TEST_RUNNER = "androidx.test.runner.AndroidJUnitRunner"
BENCHMARK_CLASS = (
    "com.example.voicememo.transcription."
    "GpuWholeMemoWerBenchmarkTest#compareWholeMemoBackends"
)
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SAFE_LANGUAGE = re.compile(r"[a-z]{2,3}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_EXPORT_MODULES = (
    "numpy",
    "tensorflow",
    "torch",
    "transformers",
    "soundfile",
)


class ExperimentError(RuntimeError):
    """A user-actionable, fail-closed experiment error."""


def log(message: str) -> None:
    print(message, flush=True)


def run(
    command: Sequence[str],
    *,
    cwd: Path = PROJECT_ROOT,
    capture: bool = False,
    check: bool = True,
    stdin: Any = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log("+ " + shlex.join(str(part) for part in command))
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        stdin=stdin,
        env=env,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_modules(names: Iterable[str]) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise ExperimentError(
            "Missing Python packages: "
            + ", ".join(missing)
            + ". Create the documented Python 3.11 environment with "
            "`python -m pip install -r scripts/requirements-original-fp16.txt`."
        )


def require_free_space(path: Path, gibibytes: int) -> None:
    free = shutil.disk_usage(path).free
    required = gibibytes * 1024**3
    if free < required:
        raise ExperimentError(
            f"Only {free / 1024**3:.1f} GiB is free at {path}; "
            f"the conversion requires at least {gibibytes} GiB. "
            "The original checkpoint, TensorFlow export, raw LiteRT model, "
            "and two rewritten candidates temporarily coexist."
        )


def read_pcm16_wav(path: Path) -> tuple[list[float], int]:
    if not path.is_file():
        raise ExperimentError(f"Audio file is missing: {path}")
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    if channels not in (1, 2) or sample_width != 2:
        raise ExperimentError(
            "The experiment currently accepts mono or stereo PCM16 WAV. "
            "Export/import the memo as WAV before running it."
        )
    values = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    if channels == 2:
        mono = [
            (values[index] + values[index + 1]) / (2.0 * 32768.0)
            for index in range(0, len(values), 2)
        ]
    else:
        mono = [value / 32768.0 for value in values]
    return mono, sample_rate


def resample_linear(samples: Sequence[float], source_rate: int) -> list[float]:
    if source_rate == SAMPLE_RATE:
        return list(samples)
    if source_rate <= 0:
        raise ExperimentError("WAV sample rate is invalid.")
    target_count = max(1, round(len(samples) * SAMPLE_RATE / source_rate))
    if len(samples) <= 1:
        return [float(samples[0]) if samples else 0.0] * target_count
    scale = source_rate / SAMPLE_RATE
    result: list[float] = []
    for index in range(target_count):
        source = min(index * scale, len(samples) - 1)
        left = int(math.floor(source))
        right = min(left + 1, len(samples) - 1)
        fraction = source - left
        result.append(
            float(samples[left]) * (1.0 - fraction)
            + float(samples[right]) * fraction
        )
    return result


def audio_windows(samples: Sequence[float]) -> list[tuple[int, list[float]]]:
    if not samples:
        return [(0, [])]
    windows: list[tuple[int, list[float]]] = []
    start = 0
    while start < len(samples):
        windows.append((start, list(samples[start : start + WINDOW_SAMPLES])))
        if start + WINDOW_SAMPLES >= len(samples):
            break
        start += WINDOW_STEP_SAMPLES
    return windows


def merge_token_overlap(left: Sequence[int], right: Sequence[int]) -> list[int]:
    maximum = min(len(left), len(right))
    overlap = 0
    for count in range(maximum, 0, -1):
        if list(left[-count:]) == list(right[:count]):
            overlap = count
            break
    return list(left) + list(right[overlap:])


def normalize_words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[\w]+(?:['’\-][\w]+)*", folded, flags=re.UNICODE)


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (0 if left_value == right_value else 1),
                )
            )
        previous = current
    return previous[-1]


def comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_words = normalize_words(str(reference.get("text", "")))
    candidate_words = normalize_words(str(candidate.get("text", "")))
    word_edits = edit_distance(reference_words, candidate_words)
    reference_tokens = [int(value) for value in reference.get("tokens", [])]
    candidate_tokens = [int(value) for value in candidate.get("tokens", [])]
    token_edits = edit_distance(reference_tokens, candidate_tokens)
    return {
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "candidate_words": len(candidate_words),
        "word_error_rate": (
            word_edits / len(reference_words) if reference_words else None
        ),
        "token_edits": token_edits,
        "reference_tokens": len(reference_tokens),
        "candidate_tokens": len(candidate_tokens),
        "token_error_rate": (
            token_edits / len(reference_tokens) if reference_tokens else None
        ),
        "exact_text_match": reference.get("text") == candidate.get("text"),
    }


def baseline(args: argparse.Namespace) -> dict[str, Any]:
    require_modules(("numpy", "torch", "transformers"))
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    samples, source_rate = read_pcm16_wav(args.audio)
    samples = resample_linear(samples, source_rate)
    log(
        f"Loading original {MODEL_ID}@{MODEL_REVISION} on CPU in FLOAT32 "
        "for the baseline…"
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()
    chunks: list[dict[str, Any]] = []
    merged_tokens: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for chunk_index, (start_sample, window) in enumerate(audio_windows(samples)):
            features = processor(
                np.asarray(window, dtype=np.float32),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            ).input_features.to(dtype=torch.float32)
            generate_kwargs: dict[str, Any] = {
                "task": "transcribe",
                "return_timestamps": False,
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": MAX_TOKENS,
            }
            if args.language != "auto":
                generate_kwargs["language"] = args.language
            chunk_started = time.perf_counter()
            generated = model.generate(features, **generate_kwargs)
            elapsed = time.perf_counter() - chunk_started
            special_ids = set(processor.tokenizer.all_special_ids)
            tokens = [
                int(value)
                for value in generated[0].tolist()
                if int(value) not in special_ids
            ]
            text = processor.decode(tokens, skip_special_tokens=True).strip()
            merged_tokens = merge_token_overlap(merged_tokens, tokens)
            chunks.append(
                {
                    "index": chunk_index,
                    "start_sample": start_sample,
                    "end_sample": min(start_sample + len(window), len(samples)),
                    "wall_seconds": elapsed,
                    "tokens": tokens,
                    "text": text,
                }
            )
            log(
                f"Baseline chunk {chunk_index + 1}: "
                f"{elapsed:.2f}s, {len(tokens)} tokens"
            )
    text = processor.decode(merged_tokens, skip_special_tokens=True).strip()
    result = {
        "kind": "original-openai-pytorch-float32-baseline",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "language": args.language,
        "audio": str(args.audio.resolve()),
        "audio_samples_16khz": len(samples),
        "window_seconds": PAYLOAD_SECONDS,
        "overlap_seconds": OVERLAP_SECONDS,
        "wall_seconds": time.perf_counter() - started,
        "tokens": merged_tokens,
        "text": text,
        "chunks": chunks,
    }
    atomic_json(args.baseline_json, result)
    log(f"Baseline written to {args.baseline_json}")
    return result


def _tf_projection(model: Any, hidden_states: Any) -> Any:
    import tensorflow as tf

    return tf.matmul(
        hidden_states,
        model.get_output_embeddings().weights,
        transpose_b=True,
    )


def export_raw_fp16(args: argparse.Namespace) -> dict[str, Any]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    require_free_space(args.work_dir, args.min_free_gib)
    require_modules(REQUIRED_EXPORT_MODULES)
    import tensorflow as tf
    from transformers import TFWhisperForConditionalGeneration

    saved_model_dir = args.work_dir / "saved-model"
    if saved_model_dir.exists():
        raise ExperimentError(
            f"Refusing to replace existing SavedModel: {saved_model_dir}. "
            "Choose an empty --work-dir."
        )
    log(
        f"Loading the pinned original checkpoint into TensorFlow: "
        f"{MODEL_ID}@{MODEL_REVISION}"
    )
    model = TFWhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        # Transformers' TF loader discovers a PyTorch-format safetensors
        # checkpoint only through the safetensors branch. `from_pt=True`
        # incorrectly forces a search for the absent pytorch_model.bin.
        from_pt=False,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )

    class WhisperLiteRtModule(tf.Module):
        def __init__(self, whisper: Any) -> None:
            super().__init__()
            self.whisper = whisper

        @tf.function(
            input_signature=[
                tf.TensorSpec([1, 128, 3000], tf.float32, name="args_0")
            ]
        )
        def encode(self, args_0: Any) -> dict[str, Any]:
            output = self.whisper.model.encoder(
                input_features=args_0,
                training=False,
                return_dict=True,
            ).last_hidden_state
            return {"output_0": tf.ensure_shape(output, [1, 1500, 1280])}

        @tf.function(
            input_signature=[
                tf.TensorSpec([1, 1500, 1280], tf.float32, name="args_0"),
                tf.TensorSpec([1, 128], tf.int32, name="args_1"),
                tf.TensorSpec(
                    [1, 1, 128, 128], tf.float32, name="args_2"
                ),
            ]
        )
        def decode(
            self, args_0: Any, args_1: Any, args_2: Any
        ) -> dict[str, Any]:
            decoder = self.whisper.model.decoder
            hidden_states = decoder.embed_tokens(args_1) * decoder.embed_scale
            positions = decoder.embed_positions(args_1)
            hidden_states = hidden_states + positions
            for layer in decoder.decoder_layers:
                outputs = layer(
                    hidden_states,
                    attention_mask=args_2,
                    encoder_hidden_states=args_0,
                    encoder_attention_mask=None,
                    layer_head_mask=None,
                    cross_attn_layer_head_mask=None,
                    past_key_value=None,
                    training=False,
                )
                hidden_states = outputs[0]
            hidden_states = decoder.layer_norm(hidden_states)
            logits = _tf_projection(self.whisper, hidden_states)
            return {
                "output_0": tf.ensure_shape(logits, [1, 128, 51866])
            }

    module = WhisperLiteRtModule(model)
    encode = module.encode.get_concrete_function()
    decode = module.decode.get_concrete_function()
    log("Exporting fixed-shape encode/decode SavedModel signatures…")
    tf.saved_model.save(
        module,
        str(saved_model_dir),
        signatures={"encode": encode, "decode": decode},
    )
    log(
        "Converting with FLOAT16 constant storage and FLOAT32 "
        "inputs/activations/outputs…"
    )
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_dir),
        signature_keys=["encode", "decode"],
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    converter.experimental_enable_resource_variables = True
    converted = converter.convert()
    args.raw_model.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.raw_model.with_suffix(".tflite.tmp")
    temporary.write_bytes(converted)
    temporary.replace(args.raw_model)
    del converted
    inspection = inspect_tflite(args.raw_model)
    inspection.update(
        {
            "kind": "raw-fp16-weight-fp32-compute-litert",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sha256": sha256(args.raw_model),
            "bytes": args.raw_model.stat().st_size,
        }
    )
    require_fp16_contract(inspection, require_graph_fixes=False)
    atomic_json(args.raw_model.with_suffix(".inspection.json"), inspection)
    shutil.rmtree(saved_model_dir)
    log(
        f"Raw FP16 LiteRT model: {args.raw_model} "
        f"({inspection['bytes']} bytes, {inspection['sha256']})"
    )
    log("Removed the verified intermediate SavedModel to reclaim working space.")
    return inspection


def _builtin_code(operator_code: Any) -> int:
    deprecated = int(operator_code.DeprecatedBuiltinCode())
    builtin = int(operator_code.BuiltinCode())
    return max(deprecated, builtin)


def _tensor_shape(tensor: Any) -> list[int]:
    return [int(tensor.Shape(index)) for index in range(tensor.ShapeLength())]


def _signature_key(signature: Any) -> str:
    for method in ("SignatureKey", "MethodName"):
        accessor = getattr(signature, method, None)
        if accessor is not None:
            value = accessor()
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
    return ""


def _tensor_map_contract(graph: Any, tensor_map: Any) -> dict[str, Any]:
    name = tensor_map.Name()
    tensor_index = int(tensor_map.TensorIndex())
    tensor = graph.Tensors(tensor_index)
    return {
        "name": name.decode("utf-8") if isinstance(name, bytes) else str(name),
        "tensor_index": tensor_index,
        "type": int(tensor.Type()),
        "shape": _tensor_shape(tensor),
    }


def inspect_tflite(path: Path) -> dict[str, Any]:
    require_modules(("tensorflow",))
    from tensorflow.lite.python import schema_py_generated as schema

    with path.open("rb") as stream:
        data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            model = schema.Model.GetRootAsModel(data, 0)
            builtin_names = {
                value: name
                for name, value in vars(schema.BuiltinOperator).items()
                if isinstance(value, int)
            }
            opcodes = [
                _builtin_code(model.OperatorCodes(index))
                for index in range(model.OperatorCodesLength())
            ]
            subgraphs: list[dict[str, Any]] = []
            rewrite_counts: list[int] = []
            mask_barriers = 0
            float16_constant_tensors = 0
            for graph_index in range(model.SubgraphsLength()):
                graph = model.Subgraphs(graph_index)
                adj_y_true = 0
                histogram: dict[str, int] = {}
                for tensor_index in range(graph.TensorsLength()):
                    tensor = graph.Tensors(tensor_index)
                    if (
                        int(tensor.Type()) == int(schema.TensorType.FLOAT16)
                        and int(tensor.Buffer()) > 0
                        and model.Buffers(int(tensor.Buffer())).DataLength() > 0
                    ):
                        float16_constant_tensors += 1
                    name = tensor.Name()
                    if (
                        name
                        and b"voicememo_explicit_rank4" in name
                        and _tensor_shape(tensor) == [1, 20, 128, 128]
                    ):
                        mask_barriers += 1
                for operator_index in range(graph.OperatorsLength()):
                    operator = graph.Operators(operator_index)
                    code = opcodes[int(operator.OpcodeIndex())]
                    name = builtin_names.get(code, str(code))
                    histogram[name] = histogram.get(name, 0) + 1
                    if code == int(schema.BuiltinOperator.BATCH_MATMUL):
                        table = operator.BuiltinOptions()
                        options = schema.BatchMatMulOptions()
                        options.Init(table.Bytes, table.Pos)
                        if options.AdjY():
                            adj_y_true += 1
                rewrite_counts.append(adj_y_true)
                subgraphs.append(
                    {
                        "index": graph_index,
                        "inputs": [
                            int(graph.Inputs(index))
                            for index in range(graph.InputsLength())
                        ],
                        "outputs": [
                            int(graph.Outputs(index))
                            for index in range(graph.OutputsLength())
                        ],
                        "tensors": graph.TensorsLength(),
                        "operators": graph.OperatorsLength(),
                        "adj_y_true": adj_y_true,
                        "operator_histogram": histogram,
                    }
                )
            signature_contracts: dict[str, Any] = {}
            for index in range(model.SignatureDefsLength()):
                signature = model.SignatureDefs(index)
                key = _signature_key(signature)
                subgraph_accessor = getattr(signature, "SubgraphIndex", None)
                graph_index = (
                    int(subgraph_accessor()) if subgraph_accessor is not None else 0
                )
                graph = model.Subgraphs(graph_index)
                signature_contracts[key] = {
                    "subgraph_index": graph_index,
                    "inputs": [
                        _tensor_map_contract(graph, signature.Inputs(map_index))
                        for map_index in range(signature.InputsLength())
                    ],
                    "outputs": [
                        _tensor_map_contract(graph, signature.Outputs(map_index))
                        for map_index in range(signature.OutputsLength())
                    ],
                }
            return {
                "signatures": list(signature_contracts),
                "signature_contracts": signature_contracts,
                "subgraphs": subgraphs,
                "rewrite_counts": rewrite_counts,
                "remaining_adj_y_true": sum(rewrite_counts),
                "mask_rank_barriers": mask_barriers,
                "float16_constant_tensors": float16_constant_tensors,
            }
        finally:
            data.close()


def require_fp16_contract(
    inspection: dict[str, Any], *, require_graph_fixes: bool
) -> None:
    if sorted(inspection["signatures"]) != ["decode", "encode"]:
        raise ExperimentError(
            f"LiteRT signatures changed: {inspection['signatures']}"
        )
    if inspection["float16_constant_tensors"] <= 0:
        raise ExperimentError("Converted model contains no FLOAT16 constants.")
    expected = {
        "encode": {
            "inputs": {"args_0": (0, [1, 128, 3000])},
            "outputs": {"output_0": (0, [1, 1500, 1280])},
        },
        "decode": {
            "inputs": {
                "args_0": (0, [1, 1500, 1280]),
                "args_1": (2, [1, 128]),
                "args_2": (0, [1, 1, 128, 128]),
            },
            "outputs": {"output_0": (0, [1, 128, 51866])},
        },
    }
    for signature_name, directions in expected.items():
        actual_signature = inspection["signature_contracts"][signature_name]
        for direction, expected_tensors in directions.items():
            actual_tensors = {
                item["name"]: (item["type"], item["shape"])
                for item in actual_signature[direction]
            }
            if actual_tensors != expected_tensors:
                raise ExperimentError(
                    f"{signature_name}/{direction} contract changed: "
                    f"{actual_tensors}"
                )
    if require_graph_fixes:
        if inspection["remaining_adj_y_true"] != 0:
            raise ExperimentError("Final model still contains adj_y=true.")
        if inspection["mask_rank_barriers"] != 4:
            raise ExperimentError(
                "Final model does not contain exactly four explicit "
                "causal-mask rank barriers."
            )


def java_tool_classpath() -> Path:
    return (
        PROJECT_ROOT
        / "tools/whisper-decoder-split/build/install/"
        "whisper-decoder-split/lib"
    )


def build_rewriter() -> None:
    java_home = Path(
        os.environ.get(
            "JAVA_HOME",
            "/Applications/Android Studio.app/Contents/jbr/Contents/Home",
        )
    )
    environment = os.environ.copy()
    environment["JAVA_HOME"] = str(java_home)
    environment["PATH"] = str(java_home / "bin") + os.pathsep + environment["PATH"]
    env_command = [
        str(PROJECT_ROOT / "gradlew"),
        "-p",
        str(PROJECT_ROOT / "tools/whisper-decoder-split"),
        "--no-daemon",
        "installDist",
    ]
    run(env_command, env=environment)


def java_rewrite(arguments: Sequence[str]) -> None:
    libraries = sorted(java_tool_classpath().glob("*.jar"))
    if not libraries:
        raise ExperimentError("Graph rewriter installation is missing.")
    classpath = os.pathsep.join(str(path) for path in libraries)
    java_home = Path(
        os.environ.get(
            "JAVA_HOME",
            "/Applications/Android Studio.app/Contents/jbr/Contents/Home",
        )
    )
    run(
        [
            str(java_home / "bin/java"),
            "-Xms128m",
            "-Xmx1g",
            "-cp",
            classpath,
            "com.example.voicememo.tools.WhisperAttentionKeyTransposeTool",
            *arguments,
        ]
    )


def rewrite_fp16(args: argparse.Namespace) -> dict[str, Any]:
    if not args.raw_model.is_file():
        raise ExperimentError(f"Raw model is missing: {args.raw_model}")
    raw_inspection = inspect_tflite(args.raw_model)
    raw_sha = sha256(args.raw_model)
    counts = raw_inspection["rewrite_counts"]
    if sum(counts) <= 0:
        raise ExperimentError(
            "Raw conversion contains no adj_y=true attention scores; "
            "the exporter graph changed and must be reviewed before proceeding."
        )
    build_rewriter()
    java_rewrite(
        [
            "generate-verified",
            str(args.raw_model),
            raw_sha,
            ",".join(str(value) for value in counts),
            str(args.explicit_key_model),
        ]
    )
    explicit_sha = sha256(args.explicit_key_model)
    java_rewrite(
        [
            "generate-mask-rank-reshapes-verified",
            str(args.explicit_key_model),
            explicit_sha,
            str(args.final_model),
        ]
    )
    final_inspection = inspect_tflite(args.final_model)
    final_inspection.update(
        {
            "kind": "fp16-weight-fp32-compute-litert-gpu-fixed",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_sha256": raw_sha,
            "explicit_key_sha256": explicit_sha,
            "sha256": sha256(args.final_model),
            "bytes": args.final_model.stat().st_size,
        }
    )
    require_fp16_contract(final_inspection, require_graph_fixes=True)
    atomic_json(args.final_model.with_suffix(".inspection.json"), final_inspection)
    args.raw_model.unlink()
    args.explicit_key_model.unlink()
    log(
        f"GPU-fixed FP16 model: {args.final_model} "
        f"({final_inspection['bytes']} bytes, {final_inspection['sha256']})"
    )
    log("Removed the two verified intermediate LiteRT models.")
    return final_inspection


def adb_base(serial: str | None) -> list[str]:
    return ["adb", *(["-s", serial] if serial else [])]


def one_device(serial: str | None) -> str:
    output = run([*adb_base(serial), "devices"], capture=True).stdout
    devices = [
        line.split()[0]
        for line in output.splitlines()
        if line.strip().endswith("\tdevice")
    ]
    if serial:
        if serial not in devices:
            raise ExperimentError(f"ADB device is unavailable: {serial}")
        return serial
    if len(devices) != 1:
        raise ExperimentError(
            f"Expected exactly one connected ADB device, found {devices}."
        )
    return devices[0]


def require_safe_name(path: Path) -> str:
    name = path.name
    if not SAFE_NAME.fullmatch(name):
        raise ExperimentError(f"Unsafe staged file name: {name}")
    return name


def adb_private_write(serial: str, source: Path, relative: str) -> None:
    if not re.fullmatch(r"files/gpu-(?:models|audio)/[A-Za-z0-9._-]+", relative):
        raise ExperimentError(f"Unsafe private staging path: {relative}")
    parent = str(Path(relative).parent)
    run(
        [
            *adb_base(serial),
            "shell",
            "run-as",
            PACKAGE,
            "mkdir",
            "-p",
            parent,
        ]
    )
    command = [
        *adb_base(serial),
        "shell",
        "run-as",
        PACKAGE,
        "tee",
        relative,
    ]
    log("+ " + shlex.join(command) + f" < {source}")
    with source.open("rb") as stream:
        completed = subprocess.run(
            command,
            stdin=stream,
            stdout=subprocess.DEVNULL,
            check=False,
        )
    if completed.returncode != 0:
        raise ExperimentError(f"Could not stage {source.name} in app-private storage.")
    device_hash = run(
        [
            *adb_base(serial),
            "shell",
            "run-as",
            PACKAGE,
            "sha256sum",
            relative,
        ],
        capture=True,
    ).stdout.split()[0]
    source_hash = sha256(source)
    if device_hash != source_hash:
        raise ExperimentError(
            f"Staged file SHA-256 changed for {source.name}."
        )


def adb_private_remove(serial: str, relative: str) -> None:
    if not re.fullmatch(r"files/gpu-(?:models|audio)/[A-Za-z0-9._-]+", relative):
        raise ExperimentError(f"Unsafe private cleanup path: {relative}")
    run(
        [
            *adb_base(serial),
            "shell",
            "run-as",
            PACKAGE,
            "rm",
            "-f",
            relative,
        ],
        check=False,
    )


def adb_capture_file(serial: str, relative: str, destination: Path) -> bool:
    exists = subprocess.run(
        [
            *adb_base(serial),
            "shell",
            "run-as",
            PACKAGE,
            "test",
            "-f",
            relative,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode != 0:
        return False
    with destination.open("wb") as output:
        completed = subprocess.run(
            [
                *adb_base(serial),
                "exec-out",
                "run-as",
                PACKAGE,
                "cat",
                relative,
            ],
            stdout=output,
            stderr=subprocess.PIPE,
        )
    if completed.returncode != 0:
        raise ExperimentError(f"Could not snapshot {relative} for the idle check.")
    return True


def assert_device_idle(serial: str) -> None:
    run(
        [*adb_base(serial), "exec-out", "run-as", PACKAGE, "id"],
        capture=True,
    )
    services = run(
        [
            *adb_base(serial),
            "shell",
            "dumpsys",
            "activity",
            "services",
            PACKAGE,
        ],
        capture=True,
    ).stdout
    if "RecordingService" in services or "SystemForegroundService" in services:
        raise ExperimentError(
            "VoiceMemo is recording or transcribing. Stop it before the "
            "in-place benchmark update."
        )
    with tempfile.TemporaryDirectory(prefix="voicememo-fp16-idle-") as directory:
        snapshot = Path(directory)
        database = snapshot / "voice-memo.db"
        if not adb_capture_file(
            serial, "databases/voice-memo.db", database
        ):
            raise ExperimentError("The installed VoiceMemo database is missing.")
        for suffix in ("-wal", "-shm"):
            adb_capture_file(
                serial,
                f"databases/voice-memo.db{suffix}",
                snapshot / f"voice-memo.db{suffix}",
            )
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            active = connection.execute(
                """
                SELECT COUNT(*) FROM transcription_jobs
                WHERE status IN (
                    'QUEUED',
                    'RUNNING',
                    'PAUSED_FOR_RECORDING',
                    'CANCEL_REQUESTED'
                )
                """
            ).fetchone()[0]
        finally:
            connection.close()
    if active:
        raise ExperimentError(
            f"{active} transcription job(s) are active. "
            "Wait or cancel them before installing the benchmark in place."
        )


def materialize_device_memo(
    memo_selector: str, serial_option: str | None, destination_dir: Path
) -> tuple[Path, str]:
    if memo_selector != "latest" and not re.fullmatch(
        r"[A-Za-z0-9-]+", memo_selector
    ):
        raise ExperimentError("--memo-id must be latest or a safe memo ID.")
    serial = one_device(serial_option)
    assert_device_idle(serial)
    with tempfile.TemporaryDirectory(prefix="voicememo-fp16-memo-") as directory:
        snapshot = Path(directory)
        database = snapshot / "voice-memo.db"
        if not adb_capture_file(
            serial, "databases/voice-memo.db", database
        ):
            raise ExperimentError("The installed VoiceMemo database is missing.")
        for suffix in ("-wal", "-shm"):
            adb_capture_file(
                serial,
                f"databases/voice-memo.db{suffix}",
                snapshot / f"voice-memo.db{suffix}",
            )
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            if memo_selector == "latest":
                row = connection.execute(
                    """
                    SELECT id, audioPath FROM memos
                    WHERE pendingDeletion=0
                    ORDER BY createdAtMillis DESC LIMIT 1
                    """
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT id, audioPath FROM memos
                    WHERE id=? AND pendingDeletion=0
                    """,
                    (memo_selector,),
                ).fetchone()
        finally:
            connection.close()
    if row is None:
        raise ExperimentError(f"Voice memo was not found: {memo_selector}")
    memo_id, device_audio_path = str(row[0]), str(row[1])
    if (
        f"/{PACKAGE}/files/memos/" not in device_audio_path
        or not device_audio_path.endswith(".wav")
    ):
        raise ExperimentError("Memo audio path is outside VoiceMemo private storage.")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"source-{memo_id}.wav"
    if destination.exists():
        raise ExperimentError(
            f"Refusing to replace existing memo snapshot: {destination}"
        )
    log(f"Copying memo {memo_id} to the local experiment directory…")
    with destination.open("wb") as output:
        completed = subprocess.run(
            [
                *adb_base(serial),
                "exec-out",
                "run-as",
                PACKAGE,
                "cat",
                device_audio_path,
            ],
            stdout=output,
            stderr=subprocess.PIPE,
        )
    if completed.returncode != 0 or destination.stat().st_size < 44:
        destination.unlink(missing_ok=True)
        raise ExperimentError("Could not copy the selected memo WAV.")
    return destination, serial


def build_android_harness(gpu_precision: str) -> tuple[Path, Path]:
    java_home = Path(
        os.environ.get(
            "JAVA_HOME",
            "/Applications/Android Studio.app/Contents/jbr/Contents/Home",
        )
    )
    environment = os.environ.copy()
    environment["JAVA_HOME"] = str(java_home)
    environment["PATH"] = str(java_home / "bin") + os.pathsep + environment["PATH"]
    command = [
        str(PROJECT_ROOT / "gradlew"),
        "--no-daemon",
        "-Pkotlin.compiler.execution.strategy=in-process",
        *(
            ["-PuseFp16Fp32AccumGpuProbe=true"]
            if gpu_precision == "fp16-fp32-accum"
            else []
        ),
        ":app:assembleDebug",
        ":app:assembleDebugAndroidTest",
    ]
    log("+ " + shlex.join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    app_apk = PROJECT_ROOT / "app/build/outputs/apk/debug/app-debug.apk"
    test_apk = (
        PROJECT_ROOT
        / "app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
    )
    if not app_apk.is_file() or not test_apk.is_file():
        raise ExperimentError("Android benchmark APK output is missing.")
    return app_apk, test_apk


def require_installed_apk(serial: str, package: str, expected_apk: Path) -> None:
    path_output = run(
        [*adb_base(serial), "shell", "pm", "path", package],
        capture=True,
    ).stdout.strip()
    if not path_output.startswith("package:"):
        raise ExperimentError(f"Package is not installed: {package}")
    device_path = path_output.removeprefix("package:")
    device_hash = run(
        [*adb_base(serial), "shell", "sha256sum", device_path],
        capture=True,
    ).stdout.split()[0]
    host_hash = sha256(expected_apk)
    if device_hash != host_hash:
        raise ExperimentError(
            f"Installed {package} APK differs from {expected_apk}; "
            "rerun without --reuse-installed-harness."
        )


def device_gpu(args: argparse.Namespace) -> dict[str, Any]:
    if not args.final_model.is_file():
        raise ExperimentError(f"Final model is missing: {args.final_model}")
    if not args.baseline_json.is_file():
        raise ExperimentError(f"Baseline JSON is missing: {args.baseline_json}")
    if args.language != "auto" and not SAFE_LANGUAGE.fullmatch(args.language):
        raise ExperimentError("--language must be auto or a 2-3 letter code.")
    serial = one_device(args.serial)
    if args.skip_build:
        app_apk = PROJECT_ROOT / "app/build/outputs/apk/debug/app-debug.apk"
        test_apk = (
            PROJECT_ROOT
            / "app/build/outputs/apk/androidTest/debug/"
            "app-debug-androidTest.apk"
        )
        if not app_apk.is_file() or not test_apk.is_file():
            raise ExperimentError("--skip-build requested but benchmark APKs are missing.")
    else:
        app_apk, test_apk = build_android_harness(args.gpu_precision)
    assert_device_idle(serial)
    if args.reuse_installed_harness:
        require_installed_apk(serial, PACKAGE, app_apk)
        require_installed_apk(serial, TEST_PACKAGE, test_apk)
        log("Reusing the byte-identical installed benchmark APKs.")
    else:
        log("Installing benchmark APKs in place; VoiceMemo is not uninstalled.")
        run([*adb_base(serial), "install", "-r", str(app_apk)])
        run([*adb_base(serial), "install", "-r", str(test_apk)])

    model_name = require_safe_name(args.final_model)
    audio_name = require_safe_name(args.audio)
    model_relative = f"files/gpu-models/{model_name}"
    audio_relative = f"files/gpu-audio/{audio_name}"
    run_id = args.run_id or f"fp16-{int(time.time())}"
    if not SAFE_NAME.fullmatch(run_id):
        raise ExperimentError("--run-id contains unsafe characters.")
    try:
        adb_private_write(serial, args.final_model, model_relative)
        adb_private_write(serial, args.audio, audio_relative)
        instrument = [
            *adb_base(serial),
            "shell",
            "am",
            "instrument",
            "-w",
            "-r",
            "-e",
            "class",
            BENCHMARK_CLASS,
            "-e",
            "runId",
            run_id,
            "-e",
            "candidateOnly",
            "true",
            "-e",
            "candidateBackend",
            args.backend,
            "-e",
            "cpuWorkers",
            str(args.cpu_workers),
            "-e",
            "cpuThreads",
            str(args.cpu_threads),
            "-e",
            "externalModelName",
            model_name,
            "-e",
            "externalModelSha256",
            sha256(args.final_model),
            "-e",
            "externalAudioName",
            audio_name,
            "-e",
            "gpuPrecision",
            args.gpu_precision,
        ]
        if args.language != "auto":
            instrument += ["-e", "forcedLanguage", args.language]
        instrument.append(f"{TEST_PACKAGE}/{TEST_RUNNER}")
        run(instrument)
        report_relative = f"cache/gpu-wer/whole-{run_id}.json"
        completed = run(
            [
                *adb_base(serial),
                "exec-out",
                "run-as",
                PACKAGE,
                "cat",
                report_relative,
            ],
            capture=True,
        )
        device_report = json.loads(completed.stdout)
    finally:
        adb_private_remove(serial, model_relative)
        adb_private_remove(serial, audio_relative)

    if device_report.get("status") != "completed":
        raise ExperimentError(
            f"Android GPU benchmark failed: {device_report.get('error')}"
        )
    baseline_report = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    candidate = device_report["candidate"]
    result = {
        "kind": (
            f"original-versus-android-cpu-{args.cpu_workers}x{args.cpu_threads}"
            if args.backend == "cpu"
            else (
                "original-versus-fp16-activation-fp32-accum-android-gpu"
                if args.gpu_precision == "fp16-fp32-accum"
                else "original-versus-fp16-weight-android-gpu-fp32"
            )
        ),
        "device_serial": serial,
        "backend": args.backend,
        "cpu_workers": args.cpu_workers,
        "cpu_threads_per_worker": args.cpu_threads,
        "gpu_precision": args.gpu_precision,
        "model_sha256": sha256(args.final_model),
        "baseline": baseline_report,
        "candidate": candidate,
        "comparison": comparison(baseline_report, candidate),
        "device_report": device_report,
    }
    atomic_json(args.result_json, result)
    log(f"Comparison written to {args.result_json}")
    return result


def common_paths(
    parser: argparse.ArgumentParser, *, require_source: bool
) -> None:
    default_work = PROJECT_ROOT / "build/experiments/original-fp16"
    source = parser.add_mutually_exclusive_group(required=require_source)
    source.add_argument("--audio", type=Path)
    source.add_argument(
        "--memo-id",
        help="copy this app-private memo, or use 'latest'",
    )
    parser.add_argument("--serial")
    parser.add_argument(
        "--language",
        default="auto",
        help="auto or a lowercase ISO language code; forcing is more reproducible",
    )
    parser.add_argument("--work-dir", type=Path, default=default_work)
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="default: WORK_DIR/baseline.json",
    )
    parser.add_argument(
        "--raw-model",
        type=Path,
        help="default: WORK_DIR/whisper-large-v3-turbo-fp16-raw.tflite",
    )
    parser.add_argument(
        "--explicit-key-model",
        type=Path,
        help=(
            "default: WORK_DIR/"
            "whisper-large-v3-turbo-fp16-explicit-key.tflite"
        ),
    )
    parser.add_argument(
        "--final-model",
        type=Path,
        help="default: WORK_DIR/whisper-large-v3-turbo-fp16-gpu-fixed.tflite",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="default: WORK_DIR/android-gpu-comparison.json",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--min-free-gib", type=int, default=18)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run original Whisper as a baseline, convert FP16 weights with "
            "FP32 compute, apply VoiceMemo GPU graph fixes, and benchmark Android."
        )
    )
    subparsers = result.add_subparsers(dest="command", required=True)
    for name in ("baseline", "convert", "rewrite", "device", "all"):
        child = subparsers.add_parser(name)
        common_paths(
            child,
            require_source=name in ("baseline", "device", "all"),
        )
        if name in ("device", "all"):
            child.add_argument("--run-id")
            child.add_argument("--skip-build", action="store_true")
            child.add_argument(
                "--backend",
                choices=("gpu", "cpu"),
                default="gpu",
            )
            child.add_argument("--cpu-workers", type=int, choices=(1, 2), default=2)
            child.add_argument("--cpu-threads", type=int, default=3)
            child.add_argument(
                "--gpu-precision",
                choices=("fp32", "fp16-fp32-accum"),
                default="fp32",
                help=(
                    "FP32 GPU compute, or the diagnostic LiteRT F32_F16 mode "
                    "with FP16 data and FP32 convolution/FC accumulation"
                ),
            )
            child.add_argument(
                "--reuse-installed-harness",
                action="store_true",
                help="verify and reuse byte-identical installed benchmark APKs",
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.work_dir = args.work_dir.expanduser().resolve()
    default_files = {
        "baseline_json": "baseline.json",
        "raw_model": "whisper-large-v3-turbo-fp16-raw.tflite",
        "explicit_key_model": (
            "whisper-large-v3-turbo-fp16-explicit-key.tflite"
        ),
        "final_model": "whisper-large-v3-turbo-fp16-gpu-fixed.tflite",
        "result_json": (
            "android-gpu-fp16-activation-fp32-accum-comparison.json"
            if getattr(args, "gpu_precision", "fp32") == "fp16-fp32-accum"
            else "android-gpu-comparison.json"
        ),
    }
    for attribute, file_name in default_files.items():
        value = getattr(args, attribute)
        resolved = (
            value.expanduser().resolve()
            if value is not None
            else args.work_dir / file_name
        )
        setattr(args, attribute, resolved)
    if args.language != "auto" and not SAFE_LANGUAGE.fullmatch(args.language):
        raise ExperimentError("--language must be auto or a 2-3 letter code.")
    if args.memo_id is not None:
        args.audio, resolved_serial = materialize_device_memo(
            args.memo_id, args.serial, args.work_dir
        )
        args.serial = resolved_serial
    elif args.audio is not None:
        args.audio = args.audio.expanduser().resolve()
    if args.command in ("baseline", "all"):
        baseline(args)
    if args.command in ("convert", "all"):
        export_raw_fp16(args)
    if args.command in ("rewrite", "all"):
        rewrite_fp16(args)
    if args.command in ("device", "all"):
        device_gpu(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as error:
        print(f"FP16 GPU experiment failed: {error}", file=sys.stderr)
        raise SystemExit(2)
