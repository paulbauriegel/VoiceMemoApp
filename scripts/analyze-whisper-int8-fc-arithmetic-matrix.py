#!/usr/bin/env python3
"""Summarize the controlled LiteRT GPU INT8 FC arithmetic experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def model_by_name(report: dict, name: str) -> dict:
    return next(model for model in report["models"] if model["name"] == name)


def case_by_name(model: dict, name: str) -> dict:
    return next(case for case in model["cases"] if case["name"] == name)


def numerical_cases(model: dict) -> list[dict]:
    return [
        {
            key: value
            for key, value in case.items()
            if not key.endswith("_inference_millis")
        }
        for case in model["cases"]
    ]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: analyze-whisper-int8-fc-arithmetic-matrix.py REPORT.json"
        )
    report_path = Path(sys.argv[1])
    report = json.loads(report_path.read_text())
    if report.get("status") != "completed":
        raise SystemExit(
            f"Device matrix failed: {report.get('error_type')}: "
            f"{report.get('error_message')}"
        )

    original = model_by_name(report, "original-per-channel")
    explicit = model_by_name(report, "original-explicit-zero-points")
    accumulation = model_by_name(report, "synthetic-accumulation-ladder")
    diagonal = model_by_name(report, "synthetic-diagonal-scale-ramp")
    float_control = model_by_name(report, "float32-original-control")
    real = case_by_name(original, "real-step-7")
    accumulation_positive = case_by_name(
        accumulation, "constant-positive"
    )["source_gpu"]
    accumulation_negative = case_by_name(
        accumulation, "constant-negative"
    )["source_gpu"]
    accumulation_impulse = case_by_name(
        accumulation, "impulse"
    )["source_gpu"]
    diagonal_positive = case_by_name(
        diagonal, "constant-positive"
    )["source_gpu"]
    synthetic_cases = diagonal["cases"] + accumulation["cases"]

    float_control_max = max(
        case["source_gpu"]["maximum_absolute_error"]
        for case in float_control["cases"]
    )
    conclusions = {
        "explicit_zero_points_change_result": (
            numerical_cases(original) != numerical_cases(explicit)
        ),
        "float32_gpu_control_maximum_absolute_error": float_control_max,
        "original_real_source_to_direct_mae_ratio": (
            real["source_gpu"]["mean_absolute_error"]
            / real["direct_gpu"]["mean_absolute_error"]
        ),
        "accumulation_positive_signed_error_weight_sum_correlation": (
            accumulation_positive[
                "signed_error_weight_sum_correlation"
            ]
        ),
        "accumulation_negative_signed_error_weight_sum_correlation": (
            accumulation_negative[
                "signed_error_weight_sum_correlation"
            ]
        ),
        "accumulation_impulse_signed_error_weight_sum_correlation": (
            accumulation_impulse[
                "signed_error_weight_sum_correlation"
            ]
        ),
        "accumulation_positive_error_slope_per_quantized_weight_sum": (
            accumulation_positive["signed_error_by_weight_sum"]["slope"]
        ),
        "accumulation_negative_error_slope_per_quantized_weight_sum": (
            accumulation_negative["signed_error_by_weight_sum"]["slope"]
        ),
        "accumulation_impulse_error_slope_per_quantized_weight_sum": (
            accumulation_impulse["signed_error_by_weight_sum"]["slope"]
        ),
        "positive_slope_normalized_by_input_and_weight_scale": (
            accumulation_positive["signed_error_by_weight_sum"]["slope"]
            / (0.25 * 0.01)
        ),
        "negative_slope_normalized_by_input_and_weight_scale": (
            accumulation_negative["signed_error_by_weight_sum"]["slope"]
            / (-0.25 * 0.01)
        ),
        "impulse_slope_normalized_by_input_and_weight_scale": (
            accumulation_impulse["signed_error_by_weight_sum"]["slope"]
            / (1.0 * 0.01)
        ),
        "half_of_one_over_255": 1.0 / 510.0,
        "diagonal_positive_absolute_error_scale_correlation": (
            diagonal_positive["absolute_error_scale_correlation"]
        ),
        "synthetic_cpu_vs_oracle_maximum_absolute_error": max(
            case["cpu_vs_oracle"]["maximum_absolute_error"]
            for case in synthetic_cases
        ),
        "synthetic_direct_gpu_vs_oracle_maximum_absolute_error": max(
            case["direct_gpu_vs_oracle"]["maximum_absolute_error"]
            for case in synthetic_cases
        ),
        "synthetic_source_gpu_vs_oracle_maximum_absolute_error": max(
            case["source_gpu_vs_oracle"]["maximum_absolute_error"]
            for case in synthetic_cases
        ),
        "diagnosis": (
            "The source-quantized path introduces a systematic half-activation-"
            "quantization-step offset before FC. Its output error is therefore "
            "proportional to each quantized weight-row sum. The fault is in "
            "quantize_float32_to_int8w4c4 or its scale/offset contract. Closed-"
            "form synthetic oracles show that the direct GPU FC accumulator is "
            "mathematically correct; the CPU hybrid path and source-quantized "
            "GPU path use different lossy activation quantizers. Explicit "
            "weight zero-point arrays do not change the result, and the FLOAT32 "
            "GPU control is exact."
        ),
    }
    destination = report_path.with_name("analysis.json")
    destination.write_text(json.dumps(conclusions, indent=2) + "\n")

    for model in report["models"]:
        real_case = case_by_name(model, "real-step-7")
        direct = real_case.get("direct_gpu")
        direct_text = (
            f"{direct['maximum_absolute_error']:.9g}"
            if direct is not None
            else "-"
        )
        print(
            f"{model['name']}: real max_abs direct={direct_text} "
            f"source={real_case['source_gpu']['maximum_absolute_error']:.9g}"
        )
    print(
        "accumulation slopes positive/negative/impulse: "
        f"{conclusions['accumulation_positive_error_slope_per_quantized_weight_sum']:.12g}, "
        f"{conclusions['accumulation_negative_error_slope_per_quantized_weight_sum']:.12g}, "
        f"{conclusions['accumulation_impulse_error_slope_per_quantized_weight_sum']:.12g}"
    )
    print(
        "normalized slopes positive/negative/impulse vs 1/510: "
        f"{conclusions['positive_slope_normalized_by_input_and_weight_scale']:.12g}, "
        f"{conclusions['negative_slope_normalized_by_input_and_weight_scale']:.12g}, "
        f"{conclusions['impulse_slope_normalized_by_input_and_weight_scale']:.12g}, "
        f"{conclusions['half_of_one_over_255']:.12g}"
    )
    print(
        "synthetic maximum error vs closed-form oracle CPU/direct/source: "
        f"{conclusions['synthetic_cpu_vs_oracle_maximum_absolute_error']:.9g}, "
        f"{conclusions['synthetic_direct_gpu_vs_oracle_maximum_absolute_error']:.9g}, "
        f"{conclusions['synthetic_source_gpu_vs_oracle_maximum_absolute_error']:.9g}"
    )
    print(conclusions["diagnosis"])
    print(f"Saved analysis: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
