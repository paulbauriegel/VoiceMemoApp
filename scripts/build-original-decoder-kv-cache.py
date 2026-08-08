#!/usr/bin/env python3
"""Export fixed-shape Whisper decoder KV-cache models from the pinned checkpoint.

The generated research artifacts are deliberately separate from production:

* ``decoder-cross-cache.tflite`` projects encoder state to four layers of
  cross-attention K/V caches once per audio window.
* ``decoder-kv-step.tflite`` consumes one token, a scalar position, fixed-size
  self-attention caches, and the cross caches. It returns one final-LayerNorm
  hidden row plus updated self-attention caches.

The fixed ``[1, 20, 128, 64]`` self caches avoid dynamic tensor shapes. Slots
after the current position are masked, so attention semantics remain causal.
The existing exact FP32 active-row projection converts the returned hidden row
to vocabulary logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
DEFAULT_OUTPUT = ROOT / "build/experiments/original-decoder-kv-cache"
LAYERS = 4
HEADS = 20
HEAD_DIM = 64
WIDTH = 1280
MAX_TOKENS = 128
ENCODER_FRAMES = 1500
VOCABULARY = 51866
FUSED_OUTPUT = ROOT / "build/experiments/original-decoder-kv-cache-fused"


class ExportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def cache_names(prefix: str) -> list[str]:
    return [f"layer_{layer}_{prefix}_{kind}" for layer in range(LAYERS) for kind in ("key", "value")]


def convert_signature(tf: Any, saved_model: Path, signature: str, output: Path) -> dict[str, Any]:
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model),
        signature_keys=[signature],
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_enable_resource_variables = True
    payload = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output)}


def inspect_model(tf: Any, path: Path) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    signatures = interpreter.get_signature_list()
    return {
        "signatures": signatures,
        "inputs": [
            {
                "name": value["name"],
                "shape": [int(item) for item in value["shape"]],
                "dtype": str(value["dtype"]),
            }
            for value in interpreter.get_input_details()
        ],
        "outputs": [
            {
                "name": value["name"],
                "shape": [int(item) for item in value["shape"]],
                "dtype": str(value["dtype"]),
            }
            for value in interpreter.get_output_details()
        ],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import tensorflow as tf
    from transformers import TFWhisperForConditionalGeneration

    output = args.output.resolve()
    require(not output.exists() or not any(output.iterdir()), f"Output is not empty: {output}")
    require(shutil.disk_usage(output.parent).free >= 4 * 1024**3, "At least 4 GiB free space is required")
    print(f"Loading {MODEL_ID}@{MODEL_REVISION}", flush=True)
    model = TFWhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        from_pt=False,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    decoder = model.model.decoder
    require(len(decoder.decoder_layers) == LAYERS, "Decoder layer count changed")
    require(model.config.decoder_attention_heads == HEADS, "Decoder head count changed")
    require(model.config.d_model == WIDTH, "Decoder width changed")

    def shape_heads(value: Any, length: int) -> Any:
        return tf.transpose(tf.reshape(value, [1, length, HEADS, HEAD_DIM]), [0, 2, 1, 3])

    class CrossCacheModule(tf.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = decoder

        @tf.function(
            input_signature=[
                tf.TensorSpec([1, ENCODER_FRAMES, WIDTH], tf.float32, name="encoder_state")
            ]
        )
        def initialize(self, encoder_state: Any) -> dict[str, Any]:
            result = {}
            for index, layer in enumerate(self.decoder.decoder_layers):
                result[f"layer_{index}_cross_key"] = shape_heads(
                    layer.encoder_attn.k_proj(encoder_state), ENCODER_FRAMES
                )
                result[f"layer_{index}_cross_value"] = shape_heads(
                    layer.encoder_attn.v_proj(encoder_state), ENCODER_FRAMES
                )
            return result

    cache_spec = tf.TensorSpec([1, HEADS, MAX_TOKENS, HEAD_DIM], tf.float32)
    cross_spec = tf.TensorSpec([1, HEADS, ENCODER_FRAMES, HEAD_DIM], tf.float32)

    class CachedStepModule(tf.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = decoder

        def attend(self, query: Any, key: Any, value: Any, mask: Any | None) -> Any:
            scores = tf.matmul(query, tf.transpose(key, [0, 1, 3, 2]))
            if mask is not None:
                scores = scores + mask
            probabilities = tf.nn.softmax(scores, axis=-1)
            attended = tf.matmul(probabilities, value)
            return tf.reshape(tf.transpose(attended, [0, 2, 1, 3]), [1, 1, WIDTH])

        @tf.function(
            input_signature=[
                tf.TensorSpec([1, 1], tf.int32, name="token"),
                tf.TensorSpec([1], tf.int32, name="position"),
                *[
                    tf.TensorSpec(cache_spec.shape, cache_spec.dtype, name=name)
                    for name in cache_names("self")
                ],
                *[
                    tf.TensorSpec(cross_spec.shape, cross_spec.dtype, name=name)
                    for name in cache_names("cross")
                ],
            ]
        )
        def step(
            self,
            token: Any,
            position: Any,
            layer_0_self_key: Any,
            layer_0_self_value: Any,
            layer_1_self_key: Any,
            layer_1_self_value: Any,
            layer_2_self_key: Any,
            layer_2_self_value: Any,
            layer_3_self_key: Any,
            layer_3_self_value: Any,
            layer_0_cross_key: Any,
            layer_0_cross_value: Any,
            layer_1_cross_key: Any,
            layer_1_cross_value: Any,
            layer_2_cross_key: Any,
            layer_2_cross_value: Any,
            layer_3_cross_key: Any,
            layer_3_cross_value: Any,
        ) -> dict[str, Any]:
            self_caches = [
                [layer_0_self_key, layer_0_self_value],
                [layer_1_self_key, layer_1_self_value],
                [layer_2_self_key, layer_2_self_value],
                [layer_3_self_key, layer_3_self_value],
            ]
            cross_caches = [
                [layer_0_cross_key, layer_0_cross_value],
                [layer_1_cross_key, layer_1_cross_value],
                [layer_2_cross_key, layer_2_cross_value],
                [layer_3_cross_key, layer_3_cross_value],
            ]
            current_position = position[0]
            hidden = self.decoder.embed_tokens(token) * self.decoder.embed_scale
            hidden = hidden + tf.gather(self.decoder.embed_positions.weight, position)[None, :, :]
            slot = tf.reshape(
                tf.one_hot(current_position, MAX_TOKENS, dtype=tf.float32),
                [1, 1, MAX_TOKENS, 1],
            )
            valid = tf.range(MAX_TOKENS, dtype=tf.int32) <= current_position
            self_mask = tf.reshape(
                tf.where(valid, tf.constant(0.0), tf.constant(-1.0e9)),
                [1, 1, 1, MAX_TOKENS],
            )
            outputs: dict[str, Any] = {}
            for index, layer in enumerate(self.decoder.decoder_layers):
                residual = hidden
                normalized = layer.self_attn_layer_norm(hidden)
                query = shape_heads(
                    layer.self_attn.q_proj(normalized) * layer.self_attn.scaling,
                    1,
                )
                new_key = shape_heads(layer.self_attn.k_proj(normalized), 1)
                new_value = shape_heads(layer.self_attn.v_proj(normalized), 1)
                old_key, old_value = self_caches[index]
                updated_key = old_key * (1.0 - slot) + new_key * slot
                updated_value = old_value * (1.0 - slot) + new_value * slot
                hidden = residual + layer.self_attn.out_proj(
                    self.attend(query, updated_key, updated_value, self_mask)
                )

                residual = hidden
                normalized = layer.encoder_attn_layer_norm(hidden)
                cross_query = shape_heads(
                    layer.encoder_attn.q_proj(normalized) * layer.encoder_attn.scaling,
                    1,
                )
                cross_key, cross_value = cross_caches[index]
                hidden = residual + layer.encoder_attn.out_proj(
                    self.attend(cross_query, cross_key, cross_value, None)
                )

                residual = hidden
                normalized = layer.final_layer_norm(hidden)
                hidden = residual + layer.fc2(layer.activation_fn(layer.fc1(normalized)))
                outputs[f"layer_{index}_self_key"] = updated_key
                outputs[f"layer_{index}_self_value"] = updated_value
            final_hidden = tf.reshape(self.decoder.layer_norm(hidden), [1, WIDTH])
            outputs["hidden"] = final_hidden
            if args.fused_projection:
                # Whisper ties the vocabulary projection to the token-embedding
                # matrix: logits = hidden @ embed_tokens.weight^T.
                outputs["logits"] = tf.matmul(
                    final_hidden,
                    self.decoder.embed_tokens.embeddings,
                    transpose_b=True,
                )
            return outputs

    cross_module = CrossCacheModule()
    step_module = CachedStepModule()
    cross_function = cross_module.initialize.get_concrete_function()
    step_function = step_module.step.get_concrete_function()

    # Host FLOAT32 parity gate against the official decoder implementation.
    encoder_state = tf.zeros([1, ENCODER_FRAMES, WIDTH], tf.float32)
    cross = cross_function(encoder_state)
    self_cache = {
        name: tf.zeros([1, HEADS, MAX_TOKENS, HEAD_DIM], tf.float32)
        for name in cache_names("self")
    }
    prompt = [model.config.decoder_start_token_id, 50259, 50360, 50364]
    maximum_error = 0.0
    maximum_logits_error = 0.0
    prefix: list[int] = []
    for position, token in enumerate(prompt):
        prefix.append(token)
        inputs = {
            "token": tf.constant([[token]], tf.int32),
            "position": tf.constant([position], tf.int32),
            **self_cache,
            **cross,
        }
        candidate = step_function(**inputs)
        self_cache = {name: candidate[name] for name in cache_names("self")}
        reference = decoder(
            input_ids=tf.constant([prefix], tf.int32),
            encoder_hidden_states=encoder_state,
            use_cache=False,
            return_dict=True,
            training=False,
        ).last_hidden_state[:, -1, :]
        error = float(np.max(np.abs(candidate["hidden"].numpy() - reference.numpy())))
        maximum_error = max(maximum_error, error)
        if args.fused_projection:
            reference_logits = model(
                {
                    "decoder_input_ids": tf.constant([prefix], tf.int32),
                    "encoder_outputs": (encoder_state,),
                },
                training=False,
            ).logits[:, -1, :]
            logits_error = float(
                np.max(np.abs(candidate["logits"].numpy() - reference_logits.numpy()))
            )
            maximum_logits_error = max(maximum_logits_error, logits_error)
    require(maximum_error <= 2.0e-5, f"Host cached-step parity failed: {maximum_error}")
    if args.fused_projection:
        require(
            maximum_logits_error <= 2.0e-4,
            f"Host fused-logits parity failed: {maximum_logits_error}",
        )

    output.mkdir(parents=True, exist_ok=True)
    step_saved = output / "saved-kv-step"
    tf.saved_model.save(step_module, str(step_saved), signatures={"step": step_function})
    step_name = (
        "decoder-kv-step-fused.tflite" if args.fused_projection else "decoder-kv-step.tflite"
    )
    step_path = output / step_name
    step_report = convert_signature(tf, step_saved, "step", step_path)
    step_report["contract"] = inspect_model(tf, step_path)
    cross_report = None
    if not args.fused_projection:
        cross_saved = output / "saved-cross-cache"
        tf.saved_model.save(
            cross_module, str(cross_saved), signatures={"initialize": cross_function}
        )
        cross_path = output / "decoder-cross-cache.tflite"
        cross_report = convert_signature(tf, cross_saved, "initialize", cross_path)
        cross_report["contract"] = inspect_model(tf, cross_path)
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "stored_weights": "FLOAT16",
        "io_dtype": "FLOAT32",
        "fused_projection": bool(args.fused_projection),
        "host_float32_max_abs_error": maximum_error,
        "cached_step": step_report,
    }
    if args.fused_projection:
        report["host_float32_max_abs_logits_error"] = maximum_logits_error
    if cross_report is not None:
        report["cross_cache"] = cross_report
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--fused-projection",
        action="store_true",
        help=(
            "Export a step model that also emits vocabulary logits via the "
            "tied embedding matrix (reuses the existing cross-cache model)."
        ),
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = FUSED_OUTPUT if args.fused_projection else DEFAULT_OUTPUT
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
