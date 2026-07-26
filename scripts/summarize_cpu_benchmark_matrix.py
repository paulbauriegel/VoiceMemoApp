#!/usr/bin/env python3
"""Validate and summarize VoiceMemo CPU benchmark reports.

The reports are deliberately content-free. Comparability is established from
the audio duration, segment/VAD counts, and device/runtime fields rather than
memo titles, transcript text, or audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


class ReportValidationError(ValueError):
    """Raised when a report is unsafe to include in a comparison."""


FINGERPRINT_FIELDS = (
    "audio_duration_ms",
    "segment_count",
    "vad_skipped_segments",
    "device",
    "soc",
    "android_sdk",
    "backend",
)

CSV_FIELDS = (
    "config",
    "workers",
    "threads_per_worker",
    "outcome",
    "audio_duration_ms",
    "segment_count",
    "vad_skipped_segments",
    "model_load_wall_ms",
    "transcription_wall_ms",
    "transcription_process_cpu_ms",
    "total_job_wall_ms",
    "real_time_factor",
    "audio_speed_x",
    "mean_active_inference_lanes",
    "start_pss_kb",
    "end_pss_kb",
    "start_thermal_status",
    "maximum_thermal_status",
    "end_thermal_status",
    "timestamp_millis",
    "device",
    "soc",
    "android_sdk",
    "backend",
)

THERMAL_STATUS_NAMES = {
    "NONE",
    "LIGHT",
    "MODERATE",
    "SEVERE",
    "CRITICAL",
    "EMERGENCY",
    "SHUTDOWN",
    "UNAVAILABLE",
}


def load_report(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportValidationError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(report, dict):
        raise ReportValidationError(f"{path}: top-level JSON value must be an object")
    return report


def _require_int(report: dict[str, Any], key: str, minimum: int = 0) -> int:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReportValidationError(
            f"{key} must be an integer greater than or equal to {minimum}; got {value!r}",
        )
    return value


def _require_number(
    report: dict[str, Any],
    key: str,
    minimum: float = 0.0,
) -> float:
    value = report.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ReportValidationError(
            f"{key} must be a number greater than or equal to {minimum}; got {value!r}",
        )
    return float(value)


def validate_report(
    report: dict[str, Any],
    expected_workers: int,
    expected_threads: int,
    baseline: dict[str, Any] | None = None,
) -> None:
    """Reject incomplete, resumed, non-CPU, or non-comparable reports."""

    errors: list[str] = []

    def expect(key: str, expected: Any) -> None:
        actual = report.get(key)
        if actual != expected:
            errors.append(f"{key} is {actual!r}; expected {expected!r}")

    expect("schema_version", 1)
    expect("outcome", "SUCCESS")
    expect("backend", "CPU_XNNPACK")
    expect("gpu_delegate", False)
    expect("xnnpack", True)
    expect("workers", expected_workers)
    expect("threads_per_worker", expected_threads)
    expect("maximum_inference_threads", expected_workers * expected_threads)
    expect("fresh_full_run", True)
    expect("precompleted_segments", 0)
    expect("start_thermal_status", "NONE")

    for key in (
        "start_thermal_status",
        "maximum_thermal_status",
        "end_thermal_status",
    ):
        value = report.get(key)
        if value not in THERMAL_STATUS_NAMES:
            errors.append(
                f"{key} must be a recognized thermal status; got {value!r}",
            )

    try:
        _require_int(report, "timestamp_millis", 1)
        _require_int(report, "audio_duration_ms", 1)
        segment_count = _require_int(report, "segment_count", 1)
        completed_segments = _require_int(report, "completed_segments", 0)
        vad_skipped_segments = _require_int(report, "vad_skipped_segments", 0)
        if completed_segments != segment_count:
            errors.append(
                f"completed_segments is {completed_segments}; expected {segment_count}",
            )
        if vad_skipped_segments > segment_count:
            errors.append(
                f"vad_skipped_segments is {vad_skipped_segments}; "
                f"segment_count is only {segment_count}",
            )
        _require_int(report, "model_load_wall_ms", 0)
        _require_int(report, "transcription_wall_ms", 1)
        _require_int(report, "transcription_process_cpu_ms", 0)
        _require_int(report, "total_job_wall_ms", 1)
        _require_int(report, "start_pss_kb", 0)
        _require_int(report, "end_pss_kb", 0)
        _require_number(report, "real_time_factor", 0.0)
        _require_number(report, "audio_speed_x", 0.0)
        _require_number(report, "mean_active_inference_lanes", 0.0)
    except ReportValidationError as error:
        errors.append(str(error))

    for key in ("device", "soc", "backend"):
        value = report.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"{key} must be a non-empty string; got {value!r}")
    try:
        _require_int(report, "android_sdk", 1)
    except ReportValidationError as error:
        errors.append(str(error))

    if baseline is not None:
        for key in FINGERPRINT_FIELDS:
            if report.get(key) != baseline.get(key):
                errors.append(
                    f"{key} is {report.get(key)!r}; "
                    f"first accepted report used {baseline.get(key)!r}",
                )

    if errors:
        raise ReportValidationError("; ".join(errors))


def report_timestamp(path: Path) -> int:
    return _require_int(load_report(path), "timestamp_millis", 1)


def _milliseconds_to_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value / 1000.0:.2f}"


def _number(value: Any, digits: int) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    return f"{value:.{digits}f}"


def _pss_mebibytes(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    return f"{value / 1024.0:.1f}"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _csv_content(rows: Sequence[dict[str, Any]]) -> str:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _markdown_content(rows: Sequence[dict[str, Any]]) -> str:
    first = rows[0]
    fastest = min(rows, key=lambda row: row["transcription_wall_ms"])
    default = next(
        (
            row
            for row in rows
            if row["workers"] == 2 and row["threads_per_worker"] == 3
        ),
        None,
    )
    speedup_text = ""
    if default is not None:
        speedup = default["transcription_wall_ms"] / fastest["transcription_wall_ms"]
        speedup_text = (
            f" It is {speedup:.2f}x as fast as the 2x3 transcription wall time."
        )

    lines = [
        "# VoiceMemo CPU benchmark matrix",
        "",
        (
            f"Fastest accepted run: **{fastest['config']}** at "
            f"{_number(fastest.get('audio_speed_x'), 2)}x real time."
            f"{speedup_text}"
        ),
        "",
        (
            "Comparable content-free fingerprint: "
            f"{first['audio_duration_ms'] / 1000.0:.2f} s audio, "
            f"{first['segment_count']} segments, "
            f"{first['vad_skipped_segments']} VAD-skipped; "
            f"{first['device']} / {first['soc']} / Android {first['android_sdk']}."
        ),
        "",
        (
            "| Config | Model load (s) | Transcribe (s) | Process CPU (s) | "
            "RTF | Speed | Mean lanes | VAD skipped | End PSS (MiB) | Thermal |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        thermal = (
            f"{row.get('start_thermal_status', '-')} -> "
            f"{row.get('maximum_thermal_status', '-')} -> "
            f"{row.get('end_thermal_status', '-')}"
        )
        lines.append(
            "| {config} | {model} | {wall} | {cpu} | {rtf} | {speed}x | "
            "{lanes} | {vad}/{segments} | {pss} | {thermal} |".format(
                config=row["config"],
                model=_milliseconds_to_seconds(row.get("model_load_wall_ms")),
                wall=_milliseconds_to_seconds(row.get("transcription_wall_ms")),
                cpu=_milliseconds_to_seconds(row.get("transcription_process_cpu_ms")),
                rtf=_number(row.get("real_time_factor"), 3),
                speed=_number(row.get("audio_speed_x"), 2),
                lanes=_number(row.get("mean_active_inference_lanes"), 2),
                vad=row.get("vad_skipped_segments", "-"),
                segments=row.get("segment_count", "-"),
                pss=_pss_mebibytes(row.get("end_pss_kb")),
                thermal=thermal,
            ),
        )
    lines.extend(
        [
            "",
            (
                "Only successful, fresh full runs that started at thermal status "
                "NONE are accepted. No memo title, transcript text, or audio is "
                "included in these reports."
            ),
            "",
        ],
    )
    return "\n".join(lines)


def summarize(report_paths: Iterable[Path], output_directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    seen_configs: set[str] = set()

    for path in report_paths:
        report = load_report(path)
        workers = _require_int(report, "workers", 1)
        threads = _require_int(report, "threads_per_worker", 1)
        validate_report(report, workers, threads, baseline)
        config = f"{workers}x{threads}"
        if config in seen_configs:
            raise ReportValidationError(f"duplicate configuration: {config}")
        seen_configs.add(config)
        baseline = baseline or report
        rows.append({"config": config, **report})

    if not rows:
        raise ReportValidationError("at least one report is required")

    output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(output_directory / "summary.csv", _csv_content(rows))
    markdown = _markdown_content(rows)
    _atomic_write_text(output_directory / "summary.md", markdown)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    timestamp_parser = subparsers.add_parser(
        "timestamp",
        help="Print timestamp_millis from one report.",
    )
    timestamp_parser.add_argument("report", type=Path)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate one candidate report.",
    )
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--workers", type=int, required=True)
    validate_parser.add_argument("--threads", type=int, required=True)
    validate_parser.add_argument("--baseline", type=Path)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Write summary.csv and summary.md for accepted reports.",
    )
    summarize_parser.add_argument("--output-dir", type=Path, required=True)
    summarize_parser.add_argument("reports", nargs="+", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "timestamp":
            print(report_timestamp(arguments.report))
            return 0

        if arguments.command == "validate":
            report = load_report(arguments.report)
            baseline = load_report(arguments.baseline) if arguments.baseline else None
            validate_report(report, arguments.workers, arguments.threads, baseline)
            print(
                f"Accepted {arguments.workers}x{arguments.threads}: "
                f"{report['audio_duration_ms'] / 1000.0:.2f} s audio, "
                f"{report['segment_count']} segments, "
                f"{report['vad_skipped_segments']} VAD-skipped.",
            )
            return 0

        rows = summarize(arguments.reports, arguments.output_dir)
        for row in rows:
            print(
                f"{row['config']:>3}  "
                f"wall={row['transcription_wall_ms'] / 1000.0:8.2f}s  "
                f"speed={row['audio_speed_x']:5.2f}x  "
                f"lanes={row['mean_active_inference_lanes']:4.2f}  "
                f"thermal={row['maximum_thermal_status']}",
            )
        return 0
    except ReportValidationError as error:
        print(f"Benchmark report rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
