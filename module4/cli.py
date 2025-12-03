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

        # Custom SIFT
        sift_a = sift.sift(img_a)
        sift_b = sift.sift(img_b)
        matches = sift.match_descriptors(sift_a.descriptors, sift_b.descriptors)
        H, inliers = sift.ransac_homography(matches, sift_a.keypoints, sift_b.keypoints)
        visual = sift.draw_matches(img_a, img_b, sift_a.keypoints, sift_b.keypoints, matches, inliers)
        output_path = stitcher.OUTPUT_DIR / "sift_matches.png"
        cv2.imwrite(str(output_path), visual)

        # OpenCV SIFT for comparison
        cv_match_count = 0
        cv_visual_path = None
        try:
            sift_cv = cv2.SIFT_create()
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
            kpa, des_a = sift_cv.detectAndCompute(gray_a, None)
            kpb, des_b = sift_cv.detectAndCompute(gray_b, None)

            if des_a is not None and des_b is not None and len(des_a) > 0 and len(des_b) > 0:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                raw_matches = bf.knnMatch(des_a, des_b, k=2)

                good = []
                for m_n in raw_matches:
                    if len(m_n) >= 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good.append(m)

                cv_match_count = len(good)

                if good:
                    cv_visual = cv2.drawMatches(
                        img_a, kpa, img_b, kpb, good, None,
                        matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0),
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    cv_visual_path = stitcher.OUTPUT_DIR / "opencv_sift_matches.png"
                    cv2.imwrite(str(cv_visual_path), cv_visual)
        except Exception:
            pass

        payload = {
            "match_count": len(matches),
            "inliers": len(inliers),
            "homography": H.tolist(),
            "visual_path": str(output_path),
            "cv_match_count": cv_match_count,
            "cv_visual_path": str(cv_visual_path) if cv_visual_path else None,
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
