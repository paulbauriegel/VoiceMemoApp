#!/usr/bin/env python3
"""Generate the fixed first-block Whisper K-projection input.

The fixture uses the pinned production graph's real token embedding, position
embedding, and first self-attention layer-normalization constants. It keeps all
128 rows so LiteRT sees the same FC geometry as production.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import pathlib
import struct
import sys


MODEL_SHA256 = "8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418"
MODEL_SIZE = 1_094_571_944
ROWS = 128
WIDTH = 1280
VOCABULARY_SIZE = 51_866
EOS_TOKEN = 50_257
DEFAULT_PROMPT = [50_258, 50_261, 50_360, 50_364, 2743, 785, 14558, 10926, 6312, 5228, 11]

# Pinned external-buffer locations from the source FlatBuffer.
TOKEN_EMBEDDING_OFFSET = 822_787_024
TOKEN_EMBEDDING_FLOATS = VOCABULARY_SIZE * WIDTH
POSITION_EMBEDDING_OFFSET = 111_391_184
POSITION_EMBEDDING_FLOATS = ROWS * WIDTH
LAYER_NORM_BETA_OFFSET = 112_138_704
LAYER_NORM_GAMMA_OFFSET = 112_143_824
LAYER_NORM_VECTOR_FLOATS = WIDTH
LAYER_NORM_EPSILON = 1.0e-5


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_floats(source, offset: int, count: int) -> array.array:
    source.seek(offset)
    payload = source.read(count * 4)
    if len(payload) != count * 4:
        raise ValueError(f"short FLOAT32 tensor read at offset {offset}")
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def generate(model: pathlib.Path, output_dir: pathlib.Path, prompt: list[int]) -> None:
    if not model.is_file() or model.stat().st_size != MODEL_SIZE:
        raise ValueError("source model is missing or has the wrong byte count")
    actual_model_sha = sha256(model)
    if actual_model_sha != MODEL_SHA256:
        raise ValueError(f"source model SHA-256 is {actual_model_sha}")
    if not prompt or len(prompt) > ROWS or any(token < 0 or token >= VOCABULARY_SIZE for token in prompt):
        raise ValueError("prompt token IDs are outside the pinned Whisper contract")

    tokens = prompt + [EOS_TOKEN] * (ROWS - len(prompt))
    with model.open("rb") as source:
        positions = read_floats(source, POSITION_EMBEDDING_OFFSET, POSITION_EMBEDDING_FLOATS)
        beta = read_floats(source, LAYER_NORM_BETA_OFFSET, LAYER_NORM_VECTOR_FLOATS)
        gamma = read_floats(source, LAYER_NORM_GAMMA_OFFSET, LAYER_NORM_VECTOR_FLOATS)
        rows: list[array.array] = []
        for row, token in enumerate(tokens):
            embedding_offset = TOKEN_EMBEDDING_OFFSET + token * WIDTH * 4
            embedding = read_floats(source, embedding_offset, WIDTH)
            position_start = row * WIDTH
            hidden = array.array(
                "f",
                (
                    f32(embedding[channel] + positions[position_start + channel])
                    for channel in range(WIDTH)
                ),
            )
            mean = f32(math.fsum(hidden) / WIDTH)
            variance = f32(
                math.fsum(
                    f32(f32(value - mean) * f32(value - mean))
                    for value in hidden
                )
                / WIDTH
            )
            inverse_stddev = f32(1.0 / math.sqrt(f32(variance + LAYER_NORM_EPSILON)))
            normalized = array.array(
                "f",
                (
                    f32(
                        f32(f32(value - mean) * inverse_stddev) * gamma[channel]
                        + beta[channel]
                    )
                    for channel, value in enumerate(hidden)
                ),
            )
            rows.append(normalized)

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = output_dir / "whisper-first-block-k-projection-step-7-input.f32"
    temporary = fixture.with_suffix(".tmp")
    with temporary.open("wb") as destination:
        for row in rows:
            if sys.byteorder != "little":
                row.byteswap()
            row.tofile(destination)
    temporary.replace(fixture)
    if fixture.stat().st_size != ROWS * WIDTH * 4:
        raise ValueError("generated fixture has the wrong byte count")

    manifest = {
        "source_model_sha256": MODEL_SHA256,
        "fixture_sha256": sha256(fixture),
        "fixture_bytes": fixture.stat().st_size,
        "shape": [1, ROWS, WIDTH],
        "prompt_tokens": prompt,
        "padding_token": EOS_TOKEN,
        "layer_norm_epsilon": LAYER_NORM_EPSILON,
        "source_tensors": {
            "token_embedding": 435,
            "position_embedding": 76,
            "beta": 95,
            "gamma": 96,
            "k_projection_weights": 73,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={fixture}")
    print(f"bytes={manifest['fixture_bytes']}")
    print(f"sha256={manifest['fixture_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--prompt",
        default=",".join(str(token) for token in DEFAULT_PROMPT),
        help="Comma-separated forced prompt plus seven reference tokens.",
    )
    args = parser.parse_args()
    prompt = [int(value) for value in args.prompt.split(",") if value]
    generate(args.model.resolve(), args.output.resolve(), prompt)


if __name__ == "__main__":
    main()
