#!/usr/bin/env python3
"""Extract the six block-31 frames that reproduce LiteRT FP16 variance overflow."""

from __future__ import annotations

from array import array
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT
    / "build/reports/encoder-precision-localization"
    / "encoder-block31-attention-residual-20260803"
)
DEFAULT_OUTPUT = ROOT / "repro/litert-fp16-layernorm-overflow"
FAILING_FRAMES = (66, 1379, 1385, 1391, 1455, 1456)
WIDTH = 1280


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report_directory = args.report.resolve()
    output_directory = args.output.resolve()
    report = json.loads((report_directory / "report.json").read_text(encoding="utf-8"))
    candidate = next(
        row for row in report["candidates"] if row["name"] == "litert-opencl-fp16"
    )
    source_name = candidate.get("state_file")
    if source_name is None:
        output_row = next(row for row in candidate["outputs"] if row["name"] == "stage")
        source_name = output_row["state_file"]
    source = report_directory / source_name
    values = array("f")
    values.frombytes(source.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != 1500 * WIDTH:
        raise RuntimeError(f"Expected {1500 * WIDTH} floats, found {len(values)}")
    selected = array("f")
    for frame in FAILING_FRAMES:
        selected.extend(values[frame * WIDTH : (frame + 1) * WIDTH])
    if sys.byteorder != "little":
        selected.byteswap()
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / "input-six-frames.f32le"
    metadata = output_directory / "input-six-frames.json"
    for path in (output, metadata):
        if path.exists():
            raise RuntimeError(f"Refusing to replace {path}")
    output.write_bytes(selected.tobytes())
    metadata.write_text(
        json.dumps(
            {
                "source_report": str(report_directory / "report.json"),
                "source_state_file": source.name,
                "source_state_sha256": sha256(source),
                "selected_original_frame_indices": FAILING_FRAMES,
                "shape": [1, len(FAILING_FRAMES), WIDTH],
                "dtype": "FLOAT32 little-endian",
                "output_file": output.name,
                "output_sha256": sha256(output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(metadata.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
