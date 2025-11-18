from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from . import pipelines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 3 image analysis utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    grad = sub.add_parser("gradients", help="Compute gradient magnitude/orientation and LoG.")
    grad.add_argument("--image", type=Path, required=True)

    key = sub.add_parser("keypoints", help="Detect keypoints interpreted as edges or corners.")
    key.add_argument("--image", type=Path, required=True)
    key.add_argument("--mode", choices=["edge", "corner"], default="edge")

    boundary = sub.add_parser("boundary", help="Find dominant object boundary.")
    boundary.add_argument("--image", type=Path, required=True)

    aruco = sub.add_parser("aruco", help="Segment region defined by ArUco markers.")
    aruco.add_argument("--image", type=Path, required=True)
    aruco.add_argument("--dictionary", default="DICT_5X5_100")

    compare = sub.add_parser("compare", help="Compare segmentation masks (e.g., SAM2 vs ours).")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "gradients":
        image = cv2.imread(str(args.image))
        if image is None:
            raise FileNotFoundError(args.image)
        result = pipelines.compute_gradients(image)
    elif args.command == "keypoints":
        image = cv2.imread(str(args.image))
        if image is None:
            raise FileNotFoundError(args.image)
        result = pipelines.detect_keypoints(image, args.mode)
    elif args.command == "boundary":
        image = cv2.imread(str(args.image))
        if image is None:
            raise FileNotFoundError(args.image)
        result = pipelines.segment_boundary(image)
    elif args.command == "aruco":
        image = cv2.imread(str(args.image))
        if image is None:
            raise FileNotFoundError(args.image)
        result = pipelines.aruco_segment(image, args.dictionary)
    elif args.command == "compare":
        ref = cv2.imread(str(args.reference), cv2.IMREAD_GRAYSCALE)
        cand = cv2.imread(str(args.candidate), cv2.IMREAD_GRAYSCALE)
        if ref is None or cand is None:
            raise FileNotFoundError("One of the mask images could not be read.")
        result = pipelines.compare_masks(ref, cand)
    else:
        raise ValueError(f"Unknown command {args.command}")

    json.dump(result, fp=sys.stdout)


if __name__ == "__main__":
    main()
