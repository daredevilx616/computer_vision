from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(buffer: bytes) -> np.ndarray:
    array = np.frombuffer(buffer, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode uploaded image.")
    return image


def disparity_map(left: np.ndarray, right: np.ndarray, num_disparities: int = 96, block_size: int = 7) -> np.ndarray:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        disp12MaxDiff=1,
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity < 0.5] = np.nan
    return disparity


def disparity_to_data_url(disparity: np.ndarray) -> str:
    valid = np.nan_to_num(disparity, nan=0.0)
    disp_norm = cv2.normalize(valid, None, 0, 255, cv2.NORM_MINMAX)
    disp_uint8 = np.uint8(disp_norm)
    colored = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_VIRIDIS)
    success, buf = cv2.imencode(".png", colored)
    if not success:
        raise RuntimeError("Failed to encode disparity image.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def pixel_to_metric_scale(image_width_px: int, focal_length_mm: float, sensor_width_mm: float) -> float:
    return (focal_length_mm / sensor_width_mm) * image_width_px


def polygon_measurements(
    disparity: np.ndarray,
    polygon: Sequence[Dict[str, float]],
    focal_px: float,
    baseline_mm: float,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    cx = disparity.shape[1] / 2.0
    cy = disparity.shape[0] / 2.0
    vertices_world: List[Dict[str, float]] = []

    for vertex in polygon:
        x = float(vertex["x"])
        y = float(vertex["y"])
        x0 = int(round(x))
        y0 = int(round(y))
        half = 3
        patch = disparity[max(0, y0 - half) : y0 + half + 1, max(0, x0 - half) : x0 + half + 1]
        disp = np.nanmean(patch)
        if np.isnan(disp) or disp <= 0:
            vertices_world.append({"x": x, "y": y, "z_mm": float("nan")})
            continue
        z_mm = (focal_px * baseline_mm) / disp
        X = (x - cx) * z_mm / focal_px
        Y = (y - cy) * z_mm / focal_px
        vertices_world.append({"x_mm": float(X), "y_mm": float(Y), "z_mm": float(z_mm), "pixel_x": x, "pixel_y": y})

    segments: List[Dict[str, float]] = []
    for idx in range(len(vertices_world)):
        start = vertices_world[idx]
        end = vertices_world[(idx + 1) % len(vertices_world)]
        if any(np.isnan([start["z_mm"], end["z_mm"]])):
            length = float("nan")
        else:
            dx = end["x_mm"] - start["x_mm"]
            dy = end["y_mm"] - start["y_mm"]
            dz = end["z_mm"] - start["z_mm"]
            length = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        segments.append(
            {
                "start": {"x": start["pixel_x"], "y": start["pixel_y"], "z_mm": start["z_mm"]},
                "end": {"x": end["pixel_x"], "y": end["pixel_y"], "z_mm": end["z_mm"]},
                "length_mm": length,
            }
        )
    return vertices_world, segments


def stereo_measurement(
    left: np.ndarray,
    right: np.ndarray,
    polygon: Sequence[Dict[str, float]],
    focal_length_mm: float,
    sensor_width_mm: float,
    baseline_mm: float,
) -> Dict[str, object]:
    disparity = disparity_map(left, right)
    focal_px = pixel_to_metric_scale(left.shape[1], focal_length_mm, sensor_width_mm)
    vertices, segments = polygon_measurements(disparity, polygon, focal_px, baseline_mm)
    disparity_url = disparity_to_data_url(disparity)
    valid_lengths = [seg["length_mm"] for seg in segments if not np.isnan(seg["length_mm"])]
    summary = {
        "disparity": disparity_url,
        "vertices": vertices,
        "segments": segments,
        "mean_length_mm": float(np.mean(valid_lengths)) if valid_lengths else float("nan"),
    }
    return summary

