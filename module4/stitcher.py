from __future__ import annotations

import base64
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


def stitch_images(images: List[np.ndarray]) -> dict:
    if len(images) < 2:
        raise ValueError("Need at least two images for stitching.")

    sift_results: List[SIFTResult] = [sift(img) for img in images]
    homographies: List[np.ndarray] = [np.eye(3)]
    match_visuals: List[str] = []

    for idx in range(1, len(images)):
        prev = sift_results[idx - 1]
        curr = sift_results[idx]
        matches = match_descriptors(prev.descriptors, curr.descriptors)
        if len(matches) < 4:
            raise RuntimeError(f"Insufficient matches between frames {idx-1} and {idx}.")
        h_prev_to_curr, inliers = ransac_homography(matches, prev.keypoints, curr.keypoints)
        curr_to_base = homographies[-1] @ np.linalg.inv(h_prev_to_curr)
        homographies.append(curr_to_base)
        match_vis = draw_matches(images[idx - 1], images[idx], prev.keypoints, curr.keypoints, matches, inliers)
        match_path = OUTPUT_DIR / f"matches_{idx - 1}_{idx}.png"
        cv2.imwrite(str(match_path), match_vis)
        match_visuals.append(_to_data_url(match_vis))

    corners = []
    for img, h in zip(images, homographies):
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
        corners.append(warped)
    all_pts = np.vstack(corners)
    x_coords, y_coords = all_pts[:, 0], all_pts[:, 1]
    min_x, min_y = np.floor([x_coords.min(), y_coords.min()]).astype(int)
    max_x, max_y = np.ceil([x_coords.max(), y_coords.max()]).astype(int)

    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=float)
    panorama = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)

    for img, h in zip(images, homographies):
        warp_matrix = translation @ h
        warped = cv2.warpPerspective(img, warp_matrix, (panorama.shape[1], panorama.shape[0]))
        mask = (warped > 0).astype(np.uint8)
        panorama = np.where(mask, warped, panorama)

    pano_path = OUTPUT_DIR / "stitched_panorama.png"
    cv2.imwrite(str(pano_path), panorama)
    return {
        "panorama": _to_data_url(panorama),
        "panorama_path": str(pano_path.relative_to(BASE_DIR)),
        "match_visuals": match_visuals,
    }
