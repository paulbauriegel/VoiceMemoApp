#!/usr/bin/env python3
"""Run the self-contained LiteRT OpenCL FP16 LayerNorm overflow reproduction."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCALIZATION_RUNNER = ROOT / "scripts/run-encoder-precision-localization.py"
REPRO_DIRECTORY = ROOT / "repro/litert-fp16-layernorm-overflow"
MODEL = REPRO_DIRECTORY / "layernorm-variance-repro.tflite"
INPUT = REPRO_DIRECTORY / "input-six-frames.f32le"
MODEL_NAME = MODEL.name
INPUT_NAME = INPUT.name
MODEL_SHA256 = "7a3780c7c531643229dfb58dca518bbeb0e31741c36720f65dd47a2e00e76579"
INPUT_SHA256 = "183205ce1637531ed878113c5a614a55c44dda0abfe798e3a749940b5a85efaa"
RUN_ID = "litert-fp16-layernorm-overflow-repro"


def load_localization():
    spec = importlib.util.spec_from_file_location("encoder_localization", LOCALIZATION_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load encoder-localization helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def property_value(suite: object, adb: str, serial: str, name: str) -> str:
    return suite.run(
        suite.adb(adb, serial, "shell", "getprop", name), capture=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="2B211FDH300F4S")
    parser.add_argument(
        "--adb", default=str(Path.home() / "Library/Android/sdk/platform-tools/adb")
    )
    parser.add_argument(
        "--output", type=Path, default=REPRO_DIRECTORY / "pixel-7-pro-result.json"
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace {output}")
    localization = load_localization()
    suite = localization.load_suite()
    serial = suite.one_device(args.adb, args.serial)
    suite.assert_device_idle(args.adb, serial)
    model = args.model.resolve()
    input_file = args.input.resolve()
    model_name = model.name
    input_name = input_file.name
    model_sha256 = suite.sha256(model)
    input_sha256 = suite.sha256(input_file)
    for source, relative, digest in (
        (model, f"files/gpu-models/{model_name}", model_sha256),
        (input_file, f"files/encoder-states/{input_name}", input_sha256),
    ):
        if not source.is_file():
            raise RuntimeError(f"Missing or invalid repro artifact: {source}")
        suite.stage_private_file(args.adb, serial, source, relative, digest)
    thermal_before = suite.thermal_snapshot(args.adb, serial)
    localization.instrument(
        suite,
        args.adb,
        serial,
        "reproduceFp16LayerNormVarianceOverflow",
        {
            "runId": args.run_id,
            "layerNormModelName": model_name,
            "layerNormModelSha256": model_sha256,
            "reproInputName": input_name,
            "reproInputSha256": input_sha256,
        },
    )
    report = json.loads(
        localization.private_bytes(
            args.adb,
            serial,
            f"cache/encoder-parity/{args.run_id}-fp16-layernorm-repro.json",
        )
    )
    result = {
        "environment": {
            "device": property_value(suite, args.adb, serial, "ro.product.model"),
            "device_codename": property_value(suite, args.adb, serial, "ro.product.device"),
            "android_release": property_value(suite, args.adb, serial, "ro.build.version.release"),
            "android_sdk": property_value(suite, args.adb, serial, "ro.build.version.sdk"),
            "build_fingerprint": property_value(
                suite, args.adb, serial, "ro.build.fingerprint"
            ),
            "soc": property_value(suite, args.adb, serial, "ro.soc.model"),
            "litert_version": "2.1.6",
            "gpu_backend": "OPENCL",
            "gpu_precision": "FP16",
        },
        "thermal_before": thermal_before,
        "thermal_after": suite.thermal_snapshot(args.adb, serial),
        "report": report,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print(f"LayerNorm overflow reproduction failed: {error}", file=sys.stderr)
        raise SystemExit(2)
