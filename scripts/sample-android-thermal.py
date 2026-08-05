#!/usr/bin/env python3
"""Continuously capture Android thermal HAL readings over ADB.

The collector is deliberately independent of a particular VoiceMemo benchmark.
Start it immediately before a workload and stop it with Ctrl-C, or provide a
fixed duration. Raw samples are written as NDJSON and a compact JSON summary is
written next to them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = 1
THERMAL_STATUS_RE = re.compile(r"^Thermal Status:\s*(-?\d+)\s*$", re.MULTILINE)
TEMPERATURE_RE = re.compile(
    r"Temperature\{mValue=([^,]+),\s*mType=(-?\d+),\s*"
    r"mName=(.*),\s*mStatus=(-?\d+)\}"
)


class SamplingError(RuntimeError):
    """Raised when a trustworthy thermal sample cannot be collected."""


@dataclass(frozen=True)
class Temperature:
    name: str
    sensor_type: int
    value_celsius: float
    status: int

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.sensor_type,
            "value_celsius": self.value_celsius,
            "status": self.status,
        }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def parse_thermalservice(text: str) -> tuple[int, list[Temperature]]:
    """Parse only the live `Current temperatures from HAL` section."""
    status_match = THERMAL_STATUS_RE.search(text)
    if status_match is None:
        raise SamplingError("thermalservice did not expose a numeric thermal status")

    section_start = text.find("Current temperatures from HAL:")
    section_end = text.find("Current cooling devices from HAL:", section_start)
    if section_start < 0 or section_end < 0:
        raise SamplingError("thermalservice did not expose the live HAL temperature section")

    temperatures: list[Temperature] = []
    for line in text[section_start:section_end].splitlines()[1:]:
        match = TEMPERATURE_RE.search(line.strip())
        if match is None:
            continue
        value = float(match.group(1))
        if not math.isfinite(value):
            continue
        temperatures.append(
            Temperature(
                name=match.group(3),
                sensor_type=int(match.group(2)),
                value_celsius=value,
                status=int(match.group(4)),
            )
        )

    if not temperatures:
        raise SamplingError("thermalservice live HAL section contained no temperatures")
    return int(status_match.group(1)), temperatures


def parse_battery(text: str) -> dict[str, Any]:
    def integer(name: str) -> int | None:
        match = re.search(rf"^\s*{re.escape(name)}:\s*(-?\d+)\s*$", text, re.MULTILINE)
        return int(match.group(1)) if match else None

    temperature = integer("temperature")
    return {
        "temperature_celsius": temperature / 10 if temperature is not None else None,
        "level_percent": integer("level"),
        "ac_powered": parse_boolean_field(text, "AC powered"),
        "usb_powered": parse_boolean_field(text, "USB powered"),
        "wireless_powered": parse_boolean_field(text, "Wireless powered"),
    }


def parse_boolean_field(text: str, name: str) -> bool | None:
    match = re.search(
        rf"^\s*{re.escape(name)}:\s*(true|false)\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).lower() == "true" if match else None


def adb_command(adb_path: str, serial: str, *arguments: str) -> list[str]:
    return [adb_path, "-s", serial, *arguments]


def run_adb(adb_path: str, serial: str, *arguments: str) -> str:
    command = adb_command(adb_path, serial, *arguments)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown ADB error"
        raise SamplingError(f"{' '.join(command)} failed: {detail}")
    return result.stdout


def collect_sample(adb_path: str, serial: str, started: float) -> dict[str, Any]:
    thermal_text = run_adb(adb_path, serial, "shell", "dumpsys", "thermalservice")
    battery_text = run_adb(adb_path, serial, "shell", "dumpsys", "battery")
    thermal_status, temperatures = parse_thermalservice(thermal_text)
    return {
        "kind": "sample",
        "captured_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "thermal_status": thermal_status,
        "temperatures": [item.as_json() for item in temperatures],
        "battery": parse_battery(battery_text),
    }


def update_summary(summary: dict[str, Any], sample: dict[str, Any]) -> None:
    summary["sample_count"] += 1
    elapsed = sample["elapsed_seconds"]
    summary["elapsed_seconds"] = elapsed

    status_key = str(sample["thermal_status"])
    summary["thermal_status_first_seen_seconds"].setdefault(status_key, elapsed)

    for temperature in sample["temperatures"]:
        name = temperature["name"]
        value = temperature["value_celsius"]
        sensor = summary["sensors"].setdefault(
            name,
            {
                "type": temperature["type"],
                "sample_count": 0,
                "first_celsius": value,
                "last_celsius": value,
                "minimum_celsius": value,
                "maximum_celsius": value,
            },
        )
        sensor["sample_count"] += 1
        sensor["last_celsius"] = value
        sensor["minimum_celsius"] = min(sensor["minimum_celsius"], value)
        sensor["maximum_celsius"] = max(sensor["maximum_celsius"], value)


def parse_metadata(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SamplingError(f"Metadata must use KEY=VALUE syntax: {item!r}")
        if key in result:
            raise SamplingError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def output_paths(raw_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    summary_path = raw_path.with_name(raw_path.name + ".summary.json")
    return raw_path, summary_path


def record(args: argparse.Namespace) -> int:
    if args.interval_seconds < 0.25:
        raise SamplingError("--interval-seconds must be at least 0.25")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise SamplingError("--duration-seconds must be greater than zero")

    raw_path, summary_path = output_paths(args.output.resolve())
    if raw_path.exists() or summary_path.exists():
        raise SamplingError(
            f"Refusing to overwrite an existing output: {raw_path} or {summary_path}"
        )
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = parse_metadata(args.metadata)
    header = {
        "kind": "session",
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": utc_now(),
        "serial": args.serial,
        "interval_seconds": args.interval_seconds,
        "requested_duration_seconds": args.duration_seconds,
        "label": args.label,
        "metadata": metadata,
    }
    summary: dict[str, Any] = {
        **header,
        "raw_file": str(raw_path),
        "sample_count": 0,
        "elapsed_seconds": 0.0,
        "stop_reason": None,
        "thermal_status_first_seen_seconds": {},
        "sensors": {},
    }

    started = time.monotonic()
    next_sample = started
    try:
        with raw_path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(header, sort_keys=True) + "\n")
            while True:
                now = time.monotonic()
                if args.duration_seconds is not None and now - started >= args.duration_seconds:
                    summary["stop_reason"] = "duration_completed"
                    break
                if now < next_sample:
                    time.sleep(next_sample - now)
                sample = collect_sample(args.adb, args.serial, started)
                output.write(json.dumps(sample, sort_keys=True) + "\n")
                output.flush()
                update_summary(summary, sample)
                next_sample += args.interval_seconds
                if next_sample < time.monotonic():
                    next_sample = time.monotonic()
    except KeyboardInterrupt:
        summary["stop_reason"] = "interrupted"
    except Exception:
        summary["stop_reason"] = "error"
        raise
    finally:
        summary["finished_at_utc"] = utc_now()
        with summary_path.open("x", encoding="utf-8") as output:
            json.dump(summary, output, indent=2, sort_keys=True)
            output.write("\n")

    print(f"Thermal samples: {raw_path}")
    print(f"Thermal summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously sample Android thermal HAL temperatures over ADB."
    )
    parser.add_argument("--serial", required=True, help="Exact ADB device serial")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="NDJSON output")
    parser.add_argument("--adb", default="adb", help="ADB executable")
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="Stop after this duration; otherwise run until Ctrl-C",
    )
    parser.add_argument("--label", help="Human-readable run label")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable provenance field such as audio_sha256=...",
    )
    return parser


def main() -> int:
    try:
        return record(build_parser().parse_args())
    except SamplingError as error:
        print(f"Thermal sampling failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
