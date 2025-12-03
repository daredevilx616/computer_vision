from __future__ import annotations

import base64
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .sift import SIFTResult, draw_matches, match_descriptors, ransac_homography, sift

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_data_url(image: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Unable to encode image.")
    return f"data:image/png;base64,{base64.b64encode(buf).decode('ascii')}"


def find_best_matches(sift_results: List[SIFTResult]) -> List[tuple]:
    """Find best matching pairs to determine image ordering."""
    match_scores = []
    n = len(sift_results)
    for i in range(n):
        for j in range(i + 1, n):
            matches = match_descriptors(sift_results[i].descriptors, sift_results[j].descriptors)
            if len(matches) >= 4:
                match_scores.append((i, j, len(matches)))
    return sorted(match_scores, key=lambda x: x[2], reverse=True)


def alpha_blend(img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Blend two images with feathering at overlaps."""
    overlap = (mask1 > 0) & (mask2 > 0)

    # Distance transform for feathering
    dist1 = cv2.distanceTransform((mask1 * 255).astype(np.uint8), cv2.DIST_L2, 3)
    dist2 = cv2.distanceTransform((mask2 * 255).astype(np.uint8), cv2.DIST_L2, 3)

    # Normalize distances
    total_dist = dist1 + dist2 + 1e-6
    alpha1 = dist1 / total_dist
    alpha2 = dist2 / total_dist

    # Blend
    result = np.zeros_like(img1)
    result[overlap] = (img1[overlap] * alpha1[overlap, None] + img2[overlap] * alpha2[overlap, None]).astype(np.uint8)
    result[mask1 & ~overlap] = img1[mask1 & ~overlap]
    result[mask2 & ~overlap] = img2[mask2 & ~overlap]

    return result


def cylindrical_warp(image: np.ndarray, focal: float | None = None) -> np.ndarray:
    """Warp an image to cylindrical coordinates to reduce bending in panoramas."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    f = focal if focal is not None else 0.8 * max(h, w)
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    theta = (x_idx - cx) / f
    h_ = (y_idx - cy) / f
    X = f * np.tan(theta) + cx
    Y = f * h_ / np.cos(theta) + cy
    map_x = X.astype(np.float32)
    map_y = Y.astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped


def stitch_images(images: List[np.ndarray]) -> dict:
    """Horizontal panorama stitching using OpenCV's built-in Stitcher."""
    if len(images) < 2:
        raise ValueError("Need at least two images for stitching.")

    print(f"[Stitcher] Starting horizontal panorama with {len(images)} images", file=sys.stderr)

    match_visuals: List[str] = []

    # --- (optional) SIFT debug visualizations between consecutive pairs ---
    # This keeps using your own SIFT + draw_matches so you still get
    # nice match visual outputs in the UI.
    sift_results: List[SIFTResult] = [sift(img) for img in images]

    for idx in range(len(images) - 1):
        matches = match_descriptors(
            sift_results[idx].descriptors,
            sift_results[idx + 1].descriptors,
        )
        print(f"[Stitcher] Image {idx} to {idx+1}: {len(matches)} matches", file=sys.stderr)

        inliers = list(range(len(matches)))  # we’re just visualizing, no RANSAC needed here

        match_vis = draw_matches(
            images[idx],
            images[idx + 1],
            sift_results[idx].keypoints,
            sift_results[idx + 1].keypoints,
            matches,
            inliers,
        )
        match_path = OUTPUT_DIR / f"matches_{idx}_{idx + 1}.png"
        cv2.imwrite(str(match_path), match_vis)
        match_visuals.append(_to_data_url(match_vis))

    # --- actual panorama stitching (same way I did it earlier) ---
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed with status code: {status}")

    print(f"[Stitcher] Stitching succeeded, panorama size: {pano.shape[1]}x{pano.shape[0]}", file=sys.stderr)

    pano_path = OUTPUT_DIR / "stitched_panorama.png"
    cv2.imwrite(str(pano_path), pano)
    print(f"[Stitcher] Panorama saved to {pano_path}", file=sys.stderr)

    return {
        "panorama": _to_data_url(pano),
        "panorama_path": str(pano_path),
        "match_visuals": match_visuals,
    }
