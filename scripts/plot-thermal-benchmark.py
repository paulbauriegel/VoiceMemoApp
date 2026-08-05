#!/usr/bin/env python3
"""Render publication PNGs from the saved VoiceMemo thermal campaigns."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = pathlib.Path(__file__).resolve().parents[1]
GPU_ROOT = ROOT / "build/reports/thermal/20260731T203853Z-gpu/campaign"
CPU_ROOT = ROOT / "build/reports/thermal/20260731T155545Z/campaign/throttle"
FRESH_CPU_ROOT = ROOT / "build/reports/thermal/20260731T212131Z-cpu/campaign/throttle"
FONT = "/System/Library/Fonts/SFNS.ttf"

WIDTH = 1600
HEIGHT = 900
BG = "#FAFAF7"
FG = "#202124"
MUTED = "#606368"
GRID = "#D9DADD"
NONE = "#4C78A8"
LIGHT = "#F2A541"
MODERATE = "#E26D35"
SEVERE = "#C73E3A"
CRITICAL = "#7C2D5A"
G3D = "#7A5195"
SKIN = "#D1495B"
BATTERY = "#2A9D8F"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, fnt, fill=FG):
    box = draw.textbbox((0, 0), value, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, font=fnt, fill=fill)


def text_right(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, fnt, fill=FG):
    box = draw.textbbox((0, 0), value, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]), xy[1]), value, font=fnt, fill=fill)


def load_samples(path: pathlib.Path):
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") != "sample":
            continue
        temperatures = {
            item["name"]: float(item["value_celsius"])
            for item in row["temperatures"]
        }
        values.append(
            {
                "seconds": float(row["elapsed_seconds"]),
                "status": int(row["thermal_status"]),
                **temperatures,
            }
        )
    return values


def line_points(rows, x_map, y_map, key: str, every: int = 1):
    selected = rows[::every]
    if rows[-1] is not selected[-1]:
        selected = [*selected, rows[-1]]
    return [(x_map(row["seconds"] / 60), y_map(row[key])) for row in selected]


def save(image: Image.Image, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    print(path)


def gpu_timeline(rows, output: pathlib.Path):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    title = font(42)
    subtitle = font(24)
    label = font(22)
    tick = font(18)
    draw.text((100, 44), "Pixel 7 Pro Mali GPU thermal throttling", font=title, fill=FG)
    draw.text((100, 94), "LiteRT OpenCL GPU · repeated 170.6 s transcription · 1 Hz thermal sampling", font=subtitle, fill=MUTED)

    left, right = 110, 1500
    top1, bottom1 = 165, 445
    top2, bottom2 = 515, 795
    maximum_minutes = rows[-1]["seconds"] / 60
    x_map = lambda value: left + value / maximum_minutes * (right - left)
    y_g3d = lambda value: bottom1 - (value - 25) / 75 * (bottom1 - top1)
    y_skin = lambda value: bottom2 - (value - 25) / 25 * (bottom2 - top2)
    transitions = [(0, NONE, "NONE"), (174.132 / 60, LIGHT, "LIGHT"), (554.185 / 60, MODERATE, "MODERATE"), (995.84 / 60, SEVERE, "SEVERE")]
    phase_fills = ["#E9F1F8", "#FFF1D6", "#FBE4D8", "#F7DDDB"]
    for index, (start, color, name) in enumerate(transitions):
        end = transitions[index + 1][0] if index + 1 < len(transitions) else maximum_minutes
        draw.rectangle((x_map(start), top1, x_map(end), bottom2), fill=phase_fills[index])
        text_center(draw, ((x_map(start) + x_map(end)) / 2, 145), name, tick, color)

    for y in range(25, 101, 15):
        if y <= 100:
            py = y_g3d(y)
            draw.line((left, py, right, py), fill=GRID, width=1)
            text_right(draw, (left - 14, py - 10), f"{y}°", tick, MUTED)
    for y in range(25, 51, 5):
        py = y_skin(y)
        draw.line((left, py, right, py), fill=GRID, width=1)
        text_right(draw, (left - 14, py - 10), f"{y}°", tick, MUTED)
    for minute in range(0, 21, 2):
        px = x_map(minute)
        draw.line((px, bottom2, px, bottom2 + 8), fill=FG, width=2)
        text_center(draw, (px, bottom2 + 30), f"{minute}", tick, MUTED)
    text_center(draw, ((left + right) / 2, 860), "Elapsed GPU load (minutes)", label, FG)
    draw.text((20, 280), "G3D °C", font=label, fill=G3D)
    draw.text((20, 625), "Device °C", font=label, fill=SKIN)

    for threshold, color, name in [(39, LIGHT, "LIGHT 39°"), (43, MODERATE, "MODERATE 43°"), (45, SEVERE, "SEVERE 45°"), (46.5, CRITICAL, "CRITICAL 46.5°")]:
        py = y_skin(threshold)
        draw.line((left, py, right, py), fill=color, width=2)
        draw.text((left + 12, py - 24), name, font=tick, fill=color)

    draw.line(line_points(rows, x_map, y_g3d, "G3D", every=2), fill=G3D, width=5, joint="curve")
    draw.line(line_points(rows, x_map, y_skin, "VIRTUAL-SKIN", every=2), fill=SKIN, width=5, joint="curve")
    draw.line(line_points(rows, x_map, y_skin, "battery", every=2), fill=BATTERY, width=4, joint="curve")
    draw.text((right - 70, y_g3d(rows[-1]["G3D"]) - 14), "G3D", font=label, fill=G3D)
    draw.text((right - 85, y_skin(rows[-1]["VIRTUAL-SKIN"]) - 18), "Skin", font=label, fill=SKIN)
    draw.text((right - 105, y_skin(rows[-1]["battery"]) + 8), "Battery", font=label, fill=BATTERY)
    save(image, output)


def time_to_states(output: pathlib.Path):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((110, 55), "Pixel 7 Pro GPU: time to thermal throttling states", font=font(44), fill=FG)
    draw.text((110, 112), "Mali GPU via LiteRT OpenCL · reached states during the bounded transcription run", font=font(24), fill=MUTED)
    left, right = 280, 1450
    top, gap = 220, 135
    maximum = 21
    x_map = lambda minute: left + minute / maximum * (right - left)
    milestones = [
        ("LIGHT", 174.132 / 60, LIGHT, "2:54"),
        ("MODERATE", 554.185 / 60, MODERATE, "9:14"),
        ("SEVERE", 995.84 / 60, SEVERE, "16:36"),
    ]
    axis_bottom = top + gap * (len(milestones) - 1) + 70
    for minute in range(0, 22, 2):
        px = x_map(minute)
        draw.line((px, top - 50, px, axis_bottom), fill=GRID, width=1)
        text_center(draw, (px, axis_bottom + 35), f"{minute}m", font(18), MUTED)
    for index, (name, minute, color, value) in enumerate(milestones):
        y = top + index * gap
        text_right(draw, (left - 35, y - 16), name, font(26), FG)
        if minute is not None:
            draw.line((left, y, x_map(minute), y), fill=color, width=18)
            draw.ellipse((x_map(minute) - 14, y - 14, x_map(minute) + 14, y + 14), fill=color)
            draw.text((x_map(minute) + 25, y - 18), value, font=font(26), fill=FG)
    save(image, output)


def cpu_timeline(rows, output: pathlib.Path):
    rows = [{**row, "CPU-MAX": max(row["BIG"], row["MID"], row["LITTLE"])} for row in rows]
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((100, 44), "Pixel 7 Pro CPU thermal throttling", font=font(42), fill=FG)
    draw.text((100, 94), "LiteRT XNNPACK CPU 4×3 · repeated 28 s transcription · 1 Hz thermal sampling", font=font(24), fill=MUTED)

    left, right = 165, 1500
    top1, bottom1 = 165, 445
    top2, bottom2 = 515, 795
    maximum_minutes = rows[-1]["seconds"] / 60
    x_map = lambda value: left + value / maximum_minutes * (right - left)
    y_cpu = lambda value: bottom1 - (value - 35) / 50 * (bottom1 - top1)
    y_skin = lambda value: bottom2 - (value - 30) / 15 * (bottom2 - top2)
    light_seconds = next(row["seconds"] for row in rows if row["status"] >= 1)
    transitions = [(0, NONE, "NONE"), (light_seconds / 60, LIGHT, "LIGHT")]
    phase_fills = ["#E9F1F8", "#FFF1D6"]
    for index, (start, color, name) in enumerate(transitions):
        end = transitions[index + 1][0] if index + 1 < len(transitions) else maximum_minutes
        draw.rectangle((x_map(start), top1, x_map(end), bottom2), fill=phase_fills[index])
        text_center(draw, ((x_map(start) + x_map(end)) / 2, 145), name, font(18), color)

    for y in range(40, 86, 10):
        py = y_cpu(y)
        draw.line((left, py, right, py), fill=GRID, width=1)
        text_right(draw, (left - 14, py - 10), f"{y}°", font(18), MUTED)
    for y in range(30, 46, 3):
        py = y_skin(y)
        draw.line((left, py, right, py), fill=GRID, width=1)
        text_right(draw, (left - 14, py - 10), f"{y}°", font(18), MUTED)
    for minute in range(0, 21, 2):
        px = x_map(minute)
        draw.line((px, bottom2, px, bottom2 + 8), fill=FG, width=2)
        text_center(draw, (px, bottom2 + 30), f"{minute}", font(18), MUTED)
    text_center(draw, ((left + right) / 2, 860), "Elapsed CPU load (minutes)", font(22), FG)
    draw.text((12, 280), "CPU max °C", font=font(20), fill=G3D)
    draw.text((20, 625), "Device °C", font=font(22), fill=SKIN)

    for threshold, color, name in [(39, LIGHT, "LIGHT 39°"), (43, MODERATE, "MODERATE 43°")]:
        py = y_skin(threshold)
        draw.line((left, py, right, py), fill=color, width=2)
        draw.text((left + 12, py - 24), name, font=font(18), fill=color)

    draw.line(line_points(rows, x_map, y_cpu, "CPU-MAX", every=2), fill=G3D, width=5, joint="curve")
    draw.line(line_points(rows, x_map, y_skin, "VIRTUAL-SKIN", every=2), fill=SKIN, width=5, joint="curve")
    draw.line(line_points(rows, x_map, y_skin, "battery", every=2), fill=BATTERY, width=4, joint="curve")
    draw.text((right - 125, y_cpu(rows[-1]["CPU-MAX"]) - 20), "CPU max", font=font(20), fill=G3D)
    draw.text((right - 85, y_skin(rows[-1]["VIRTUAL-SKIN"]) - 18), "Skin", font=font(20), fill=SKIN)
    draw.text((right - 105, y_skin(rows[-1]["battery"]) + 8), "Battery", font=font(20), fill=BATTERY)
    save(image, output)


def cpu_time_to_states(rows, output: pathlib.Path):
    first_light = next(row["seconds"] for row in rows if row["status"] >= 1)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((110, 55), "Pixel 7 Pro CPU: time to thermal throttling states", font=font(44), fill=FG)
    draw.text((110, 112), "LiteRT XNNPACK CPU 4×3 · reached states during the matched 20:52 run", font=font(24), fill=MUTED)
    left, right, y = 280, 1450, 360
    maximum = 21
    x_map = lambda minute: left + minute / maximum * (right - left)
    for minute in range(0, 22, 2):
        px = x_map(minute)
        draw.line((px, y - 110, px, y + 70), fill=GRID, width=1)
        text_center(draw, (px, y + 105), f"{minute}m", font(18), MUTED)
    text_right(draw, (left - 35, y - 16), "LIGHT", font(26), FG)
    minute = first_light / 60
    draw.line((left, y, x_map(minute), y), fill=LIGHT, width=18)
    draw.ellipse((x_map(minute) - 14, y - 14, x_map(minute) + 14, y + 14), fill=LIGHT)
    draw.text((x_map(minute) + 25, y - 18), f"{int(first_light // 60)}:{int(first_light % 60):02d}", font=font(26), fill=FG)
    save(image, output)


def fresh_cpu_units():
    manifest = json.loads((FRESH_CPU_ROOT / "manifest.json").read_text(encoding="utf-8"))
    samples = load_samples(FRESH_CPU_ROOT / "thermal.ndjson")
    units = []
    for item in manifest["units"]:
        selected = [row for row in samples if item["start_offset_seconds"] <= row["seconds"] <= item["end_offset_seconds"]]
        units.append({
            "unit": item["unit"],
            "seconds": item["candidate"]["total_wall_millis"] / 1000,
            "peak_skin": max(row["VIRTUAL-SKIN"] for row in selected),
            "status": max(row["status"] for row in selected),
        })
    baseline = units[0]["seconds"]
    for item in units:
        item["slowdown"] = (item["seconds"] / baseline - 1) * 100
    return units


def fresh_cpu_performance_scatter(units, output: pathlib.Path):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((105, 50), "Pixel 7 Pro CPU slowdown during sustained heat", font=font(44), fill=FG)
    draw.text((105, 107), "LiteRT XNNPACK CPU 4×3 · slowdown versus unit 1 · nine completed units", font=font(24), fill=MUTED)
    left, right, top, bottom = 135, 1490, 180, 790
    x_min, x_max, y_min, y_max = 38.5, 42.3, -3, 45
    x_map = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    y_map = lambda value: bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    for x in (39, 40, 41, 42):
        px = x_map(x)
        draw.line((px, top, px, bottom), fill=GRID, width=1)
        text_center(draw, (px, bottom + 30), f"{x}°", font(18), MUTED)
    for y in range(0, 46, 10):
        py = y_map(y)
        draw.line((left, py, right, py), fill=GRID, width=1)
        text_right(draw, (left - 15, py - 10), f"{y}%", font(18), MUTED)
    draw.line((x_map(39), top, x_map(39), bottom), fill=LIGHT, width=3)
    draw.text((x_map(39) + 8, top + 8), "39 °C LIGHT threshold", font=font(19), fill=LIGHT)
    points = [(x_map(item["peak_skin"]), y_map(item["slowdown"])) for item in units]
    draw.line(points, fill="#777A80", width=3)
    offsets = {1: (12, -28), 2: (12, -28), 3: (12, -28), 4: (-48, -34), 5: (12, -28), 6: (12, 4), 7: (-48, 4), 8: (12, -28), 9: (12, -28)}
    for item, (x, y) in zip(units, points):
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=LIGHT, outline=BG, width=2)
        ox, oy = offsets[item["unit"]]
        draw.text((x + ox, y + oy), f"U{item['unit']}", font=font(16), fill=FG)
    text_center(draw, ((left + right) / 2, 855), "Peak skin temperature during unit", font(24), FG)
    draw.text((20, 440), "Runtime increase\nvs unit 1", font=font(22), fill=FG)
    save(image, output)


def fresh_cpu_phase_bars(units, output: pathlib.Path):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((105, 46), "Pixel 7 Pro CPU performance loss under LIGHT throttling", font=font(40), fill=FG)
    draw.text((105, 108), "LiteRT XNNPACK CPU 4×3 · matched 20:52 heating window · unit 1 = reference", font=font(22), fill=MUTED)
    sustained = statistics.median(item["slowdown"] for item in units[1:])
    slowest = max(units, key=lambda item: item["slowdown"])
    final = units[-1]
    labels = ["First unit\nNONE → LIGHT", "Sustained LIGHT\nUnits 2–9 median", f"Slowest LIGHT\nUnit {slowest['unit']}", f"Final LIGHT\nUnit {final['unit']}"]
    values = [0.0, sustained, slowest["slowdown"], final["slowdown"]]
    colors = [NONE, LIGHT, SEVERE, LIGHT]
    left, right, top, bottom = 160, 1480, 190, 720
    y_map = lambda value: bottom - value / 50 * (bottom - top)
    for y in range(0, 51, 10):
        py = y_map(y)
        draw.line((left, py, right, py), fill=GRID, width=1)
        text_right(draw, (left - 18, py - 10), f"{y}%", font(18), MUTED)
    slot = (right - left) / len(values)
    for index, (name, value, color) in enumerate(zip(labels, values, colors)):
        x0 = left + index * slot + 55
        x1 = left + (index + 1) * slot - 55
        draw.rectangle((x0, y_map(value), x1, bottom), fill=color)
        text_center(draw, ((x0 + x1) / 2, y_map(value) - 32), f"+{value:.1f}%", font(28), FG)
        first, second = name.split("\n")
        text_center(draw, ((x0 + x1) / 2, bottom + 42), first, font(20), FG)
        text_center(draw, ((x0 + x1) / 2, bottom + 72), second, font(18), MUTED)
    draw.text((35, 430), "Runtime increase", font=font(22), fill=FG)
    save(image, output)


def cpu_units():
    manifest = json.loads((CPU_ROOT / "manifest.json").read_text(encoding="utf-8"))
    samples = load_samples(CPU_ROOT / "thermal.ndjson")
    units = []
    for item in manifest["units"]:
        selected = [row for row in samples if item["start_offset_seconds"] <= row["seconds"] <= item["end_offset_seconds"]]
        units.append(
            {
                "unit": item["unit"],
                "seconds": item["candidate"]["total_wall_millis"] / 1000,
                "peak_skin": max(row["VIRTUAL-SKIN"] for row in selected),
                "status": max(row["status"] for row in selected),
            }
        )
    baseline = statistics.median(item["seconds"] for item in units[:5])
    for item in units:
        item["slowdown"] = (item["seconds"] / baseline - 1) * 100
    return units


def performance_scatter(units, output: pathlib.Path):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((105, 50), "Performance loss as phone skin temperature rises", font=font(44), fill=FG)
    draw.text((105, 107), "CPU/XNNPACK 4×3 control · each point is one completed 28 s workload unit", font=font(24), fill=MUTED)
    left, right, top, bottom = 125, 1490, 180, 790
    x_min, x_max, y_min, y_max = 28, 41.2, -2, 15
    x_map = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    y_map = lambda value: bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    for x in range(28, 42, 2):
        px = x_map(x)
        draw.line((px, top, px, bottom), fill=GRID, width=1)
        text_center(draw, (px, bottom + 30), f"{x}°", font(18), MUTED)
    for y in range(0, 16, 3):
        py = y_map(y)
        draw.line((left, py, right, py), fill=GRID, width=1)
        text_right(draw, (left - 15, py - 10), f"{y}%", font(18), MUTED)
    draw.line((x_map(39), top, x_map(39), bottom), fill=LIGHT, width=3)
    draw.text((x_map(39) + 8, top + 8), "39 °C LIGHT threshold", font=font(19), fill=LIGHT)
    points = [(x_map(item["peak_skin"]), y_map(item["slowdown"])) for item in units]
    draw.line(points, fill="#777A80", width=3)
    late_offsets = {11: (-10, -36), 12: (12, -48), 13: (12, 2), 14: (12, -42), 15: (12, -34), 16: (12, 8)}
    for item, (x, y) in zip(units, points):
        color = LIGHT if item["status"] >= 1 else NONE
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=color, outline=BG, width=2)
        ox, oy = late_offsets.get(item["unit"], (11, -25))
        draw.text((x + ox, y + oy), f"U{item['unit']}", font=font(16), fill=FG)
    draw.text((left + 25, top + 25), "● global NONE", font=font(21), fill=NONE)
    draw.text((left + 25, top + 60), "● global LIGHT", font=font(21), fill=LIGHT)
    text_center(draw, ((left + right) / 2, 855), "Peak skin temperature during unit", font(24), FG)
    draw.text((18, 440), "Slowdown vs\nearly-run median", font=font(22), fill=FG)
    save(image, output)


def phase_bars(output: pathlib.Path):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((105, 46), "Measured transcription slowdown by thermal phase", font=font(40), fill=FG)
    draw.text((105, 108), "CPU/XNNPACK 4×3 control · completed units only · early units 1–5 = reference", font=font(22), fill=MUTED)
    labels = ["Early global NONE\nUnits 1–5", "Sensor LIGHT, global NONE\nUnits 6–7", "Global LIGHT\nUnits 8–16", "Slowest unit\nUnit 15"]
    values = [0.0, 4.2858, 9.5984, 12.5389]
    colors = [NONE, LIGHT, LIGHT, SEVERE]
    left, right, top, bottom = 160, 1480, 190, 720
    y_map = lambda value: bottom - value / 15 * (bottom - top)
    for y in range(0, 16, 3):
        py = y_map(y)
        draw.line((left, py, right, py), fill=GRID, width=1)
        text_right(draw, (left - 18, py - 10), f"{y}%", font(18), MUTED)
    slot = (right - left) / len(values)
    for index, (name, value, color) in enumerate(zip(labels, values, colors)):
        x0 = left + index * slot + 55
        x1 = left + (index + 1) * slot - 55
        draw.rectangle((x0, y_map(value), x1, bottom), fill=color)
        text_center(draw, ((x0 + x1) / 2, y_map(value) - 32), f"+{value:.1f}%", font(28), FG)
        first, second = name.split("\n")
        text_center(draw, ((x0 + x1) / 2, bottom + 42), first, font(20), FG)
        text_center(draw, ((x0 + x1) / 2, bottom + 72), second, font(18), MUTED)
    draw.text((35, 430), "Runtime increase", font=font(22), fill=FG)
    save(image, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "blog/images")
    args = parser.parse_args()
    gpu_rows = load_samples(GPU_ROOT / "thermal.ndjson")
    fresh_cpu_rows = load_samples(FRESH_CPU_ROOT / "thermal.ndjson")
    units = cpu_units()
    fresh_units = fresh_cpu_units()
    gpu_timeline(gpu_rows, args.output / "gpu-thermal-throttling-timeline.png")
    time_to_states(args.output / "gpu-time-to-thermal-states.png")
    cpu_timeline(fresh_cpu_rows, args.output / "cpu-thermal-throttling-timeline.png")
    cpu_time_to_states(fresh_cpu_rows, args.output / "cpu-time-to-thermal-states.png")
    fresh_cpu_performance_scatter(fresh_units, args.output / "cpu-performance-loss-vs-temperature.png")
    fresh_cpu_phase_bars(fresh_units, args.output / "cpu-performance-loss-by-state.png")
    performance_scatter(units, args.output / "thermal-performance-loss-vs-temperature.png")
    phase_bars(args.output / "thermal-performance-loss-by-state.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
