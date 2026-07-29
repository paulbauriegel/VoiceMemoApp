#!/usr/bin/env python3
"""Classify ML Drift program-cache variants without private debug symbols."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


PARAMETER_PREFIXES = (
    "weights",
    "src_",
    "dst_",
    "params_",
    "shared_",
    "kernel_param_",
    "main_function",
)


def printable_strings(data: bytes) -> set[str]:
    return {
        match.group().decode("ascii")
        for match in re.finditer(rb"[\x20-\x7e]{4,}", data)
    }


def classify(parameters: set[str]) -> str:
    if "weights_convert_uint8_to_float32" in parameters:
        return "direct-int8-fc-dequantized-to-float"
    if "src_tensor_buffer" in parameters:
        return "source-quantized-fc-conv-buffer-bindings"
    if "src_tensor_image2d" in parameters:
        return "source-quantized-fc-conv-image2d-bindings"
    return "unknown"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: analyze-whisper-int8-fc-kernel-caches.py REPORT_DIRECTORY"
        )
    report_directory = Path(sys.argv[1])
    report = json.loads((report_directory / "report.json").read_text())
    analyses: list[dict[str, object]] = []
    for variant in report["variants"]:
        name = variant["name"]
        cache = report_directory / f"{name}_mldrift_program_cache.bin"
        data = cache.read_bytes()
        strings = printable_strings(data)
        parameters = sorted(
            value
            for value in strings
            if value.startswith(PARAMETER_PREFIXES)
        )
        parameter_payload = ("\n".join(parameters) + "\n").encode()
        analyses.append(
            {
                "variant": name,
                "binding_family": classify(set(parameters)),
                "cache_bytes": len(data),
                "cache_sha256": hashlib.sha256(data).hexdigest(),
                "parameter_signature_sha256": hashlib.sha256(
                    parameter_payload
                ).hexdigest(),
                "parameters": parameters,
                "implementation_class": None,
                "implementation_class_note": (
                    "LiteRT 2.1.6 strips ML Drift implementation-class names "
                    "from the serialized program cache."
                ),
            }
        )

    groups: dict[tuple[str, str], list[str]] = {}
    for analysis in analyses:
        key = (
            str(analysis["binding_family"]),
            str(analysis["parameter_signature_sha256"]),
        )
        groups.setdefault(key, []).append(str(analysis["variant"]))
    result = {
        "source_report": "report.json",
        "variants": analyses,
        "binding_groups": [
            {
                "binding_family": family,
                "parameter_signature_sha256": signature,
                "variants": variants,
            }
            for (family, signature), variants in groups.items()
        ],
        "exact_class_status": (
            "Requires the source-built ML Drift RTTI logger; the Maven "
            "accelerator and program cache expose bindings but not the "
            "ConvWaveMatrixMali/ConvWaveMatrix/ConvWaveMemory class name."
        ),
    }
    destination = report_directory / "kernel-selection.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    for group in result["binding_groups"]:
        print(
            f"{group['binding_family']}: "
            + ", ".join(group["variants"])
        )
    print(f"Saved kernel binding analysis: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
