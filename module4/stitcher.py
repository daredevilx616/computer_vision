from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"  # unused now but kept for compatibility

MAX_STITCH_DIM = int(os.getenv("MODULE4_STITCH_MAX_DIM", "1600"))


def _to_data_url(image: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Unable to encode image.")
    return f"data:image/png;base64,{base64.b64encode(buf).decode('ascii')}"


def _resize_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale image so the longest side is <= max_dim."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def stitch_images(images: List[np.ndarray]) -> dict:
    """Horizontal panorama stitching using OpenCV's built-in Stitcher."""
    if len(images) < 2:
        raise ValueError("Need at least two images for stitching.")

    print(f"[Stitcher] Starting horizontal panorama with {len(images)} images", file=sys.stderr)

    # Downscale inputs so Stitcher stays within Render's worker timeout.
    resized_for_stitch = [_resize_max_dim(img, MAX_STITCH_DIM) for img in images]

    # Match visuals removed to avoid extra SIFT work in constrained environments.
    match_visuals: List[str] = []

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(resized_for_stitch)

    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed with status code: {status}")

    print(f"[Stitcher] Stitching succeeded, panorama size: {pano.shape[1]}x{pano.shape[0]}", file=sys.stderr)

    return {
        "panorama": _to_data_url(pano),
        "match_visuals": match_visuals,
    }
