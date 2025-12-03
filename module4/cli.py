"""
Module 4 CLI for SIFT and panorama stitching.
Usage:
  python -m module4.cli sift --image-a left.png --image-b right.png
  python -m module4.cli stitch --images img1.png img2.png img3.png
Emits JSON with match stats or panorama paths; images saved under module4/output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from . import sift, stitcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 4 panoramas and SIFT tooling.")
    sub = parser.add_subparsers(dest="command", required=True)

    sift_parser = sub.add_parser("sift", help="Run custom SIFT + RANSAC on two images.")
    sift_parser.add_argument("--image-a", type=Path, required=True)
    sift_parser.add_argument("--image-b", type=Path, required=True)

    stitch_parser = sub.add_parser("stitch", help="Stitch multiple images into a panorama.")
    stitch_parser.add_argument("--images", type=Path, nargs="+", required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "sift":
        img_a = cv2.imread(str(args.image_a))
        img_b = cv2.imread(str(args.image_b))
        if img_a is None or img_b is None:
            raise FileNotFoundError("Unable to read input images.")
        sift_a = sift.sift(img_a)
        sift_b = sift.sift(img_b)
        matches = sift.match_descriptors(sift_a.descriptors, sift_b.descriptors)
        H, inliers = sift.ransac_homography(matches, sift_a.keypoints, sift_b.keypoints)
        visual = sift.draw_matches(img_a, img_b, sift_a.keypoints, sift_b.keypoints, matches, inliers)
        output_path = stitcher.OUTPUT_DIR / "sift_matches.png"
        cv2.imwrite(str(output_path), visual)
        payload = {
            "match_count": len(matches),
            "inliers": len(inliers),
            "homography": H.tolist(),
            "visual_path": str(output_path.relative_to(sift.BASE_DIR)),
        }
        json.dump(payload, sys.stdout)
    elif args.command == "stitch":
        images = []
        for path in args.images:
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Unable to read {path}")
            images.append(image)
        result = stitcher.stitch_images(images)
        json.dump(result, sys.stdout)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
