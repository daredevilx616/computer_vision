from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from . import stereo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 7 stereo measurement utilities.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--polygon", type=str, required=True, help="Comma-separated x:y pairs, e.g., 10:20,200:20,...")
    parser.add_argument("--focal-mm", type=float, required=True)
    parser.add_argument("--sensor-width-mm", type=float, required=True)
    parser.add_argument("--baseline-mm", type=float, required=True)
    return parser


def parse_polygon(spec: str):
    points = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        x_str, y_str = token.split(":")
        points.append({"x": float(x_str), "y": float(y_str)})
    return points


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    left = cv2.imread(str(args.left))
    right = cv2.imread(str(args.right))
    if left is None or right is None:
        raise FileNotFoundError("Failed to read left/right images.")

    polygon = parse_polygon(args.polygon)
    result = stereo.stereo_measurement(
        left=left,
        right=right,
        polygon=polygon,
        focal_length_mm=args.focal_mm,
        sensor_width_mm=args.sensor_width_mm,
        baseline_mm=args.baseline_mm,
    )
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
