#!/usr/bin/env python3
"""Find a visible Android UI Automator node and print its center coordinates."""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


BOUNDS_PATTERN = re.compile(r"^\[(\d+),(\d+)]\[(\d+),(\d+)]$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", type=Path)
    parser.add_argument("--resource-id")
    parser.add_argument("--text")
    parser.add_argument("--content-desc")
    return parser.parse_args()


def resource_id_matches(actual: str, expected: str) -> bool:
    return actual == expected or actual.endswith(f":id/{expected}")


def main() -> int:
    args = parse_args()
    if not any((args.resource_id, args.text, args.content_desc)):
        raise SystemExit("At least one selector is required.")

    root = ET.parse(args.xml).getroot()
    for node in root.iter("node"):
        attributes = node.attrib
        if args.resource_id and not resource_id_matches(
            attributes.get("resource-id", ""),
            args.resource_id,
        ):
            continue
        if args.text is not None and attributes.get("text") != args.text:
            continue
        if (
            args.content_desc is not None
            and attributes.get("content-desc") != args.content_desc
        ):
            continue

        bounds = BOUNDS_PATTERN.match(attributes.get("bounds", ""))
        if bounds is None:
            continue
        left, top, right, bottom = (int(value) for value in bounds.groups())
        if right <= left or bottom <= top:
            continue
        print(f"{(left + right) // 2} {(top + bottom) // 2}")
        return 0
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ET.ParseError, OSError) as error:
        print(f"Could not parse UI hierarchy: {error}", file=sys.stderr)
        raise SystemExit(2)
