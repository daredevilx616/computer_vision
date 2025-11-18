from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def _write_image(image: np.ndarray, stem: str) -> Path:
    path = OUTPUT_DIR / f"{stem}.png"
    cv2.imwrite(str(path), image)
    return path


def _to_data_url(image: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image buffer.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def compute_gradients(image: np.ndarray) -> Dict[str, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mag_img = mag_norm.astype(np.uint8)

    hsv = np.zeros((angle.shape[0], angle.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV hue range [0,180)
    hsv[..., 1] = 255
    hsv[..., 2] = 255
    angle_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
    log = cv2.Laplacian(blurred, cv2.CV_32F)
    log_norm = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mag_path = _write_image(mag_img, "module3_gradient_magnitude")
    angle_path = _write_image(angle_color, "module3_gradient_orientation")
    log_path = _write_image(log_norm, "module3_log_filter")

    return {
        "magnitude": _to_data_url(mag_img),
        "orientation": _to_data_url(angle_color),
        "log": _to_data_url(log_norm),
        "magnitude_path": str(mag_path.relative_to(BASE_DIR)),
        "orientation_path": str(angle_path.relative_to(BASE_DIR)),
        "log_path": str(log_path.relative_to(BASE_DIR)),
    }


def detect_keypoints(image: np.ndarray, mode: str = "edge") -> Dict[str, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    overlay = image.copy()

    if mode == "edge":
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        threshold = 0.4 * magnitude.max()
        edge_map = (magnitude > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        mask = cv2.cvtColor(edge_map * 255, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(overlay, 0.8, mask, 0.2, 0)
    elif mode == "corner":
        gray_f = np.float32(gray)
        harris = cv2.cornerHarris(gray_f, 2, 3, 0.04)
        harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX)
        points = np.argwhere(harris_norm > 180)
        for y, x in points:
            cv2.circle(overlay, (int(x), int(y)), 4, (255, 0, 0), 1)
        blended = overlay
    else:
        raise ValueError(f"Unknown keypoint mode '{mode}'")

    path = _write_image(blended, f"module3_keypoints_{mode}")
    return {
        "overlay": _to_data_url(blended),
        "overlay_path": str(path.relative_to(BASE_DIR)),
    }


def segment_boundary(image: np.ndarray) -> Dict[str, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No object boundaries detected.")
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 2)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(overlay, 0.85, mask_bgr, 0.15, 0)

    contour_path = _write_image(blended, "module3_boundary")
    mask_path = _write_image(mask, "module3_boundary_mask")
    return {
        "overlay": _to_data_url(blended),
        "mask": _to_data_url(mask),
        "overlay_path": str(contour_path.relative_to(BASE_DIR)),
        "mask_path": str(mask_path.relative_to(BASE_DIR)),
    }


def aruco_segment(image: np.ndarray, dictionary_name: str = "DICT_5X5_100") -> Dict[str, str]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module not available; install opencv-contrib-python.")

    dict_id = getattr(cv2.aruco, dictionary_name, None)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dictionary '{dictionary_name}'")
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No ArUco markers detected.")

    all_points = np.vstack([c.reshape(-1, 2) for c in corners]).astype(np.int32)
    hull = cv2.convexHull(all_points)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    overlay = image.copy()
    cv2.polylines(overlay, [hull], True, (0, 255, 0), 2)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(overlay, 0.8, mask_bgr, 0.2, 0)

    overlay_path = _write_image(blended, "module3_aruco_overlay")
    mask_path = _write_image(mask, "module3_aruco_mask")
    return {
        "overlay": _to_data_url(blended),
        "mask": _to_data_url(mask),
        "overlay_path": str(overlay_path.relative_to(BASE_DIR)),
        "mask_path": str(mask_path.relative_to(BASE_DIR)),
        "marker_count": len(ids),
        "dictionary": dictionary_name,
    }


def compare_masks(reference: np.ndarray, challenger: np.ndarray) -> Dict[str, float]:
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if challenger.ndim == 3:
        challenger = cv2.cvtColor(challenger, cv2.COLOR_BGR2GRAY)
    _, ref_bin = cv2.threshold(reference, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, chal_bin = cv2.threshold(challenger, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    intersection = np.logical_and(ref_bin, chal_bin).sum()
    union = np.logical_or(ref_bin, chal_bin).sum()
    dice = (2 * intersection) / (ref_bin.sum() + chal_bin.sum() + 1e-6)
    iou = intersection / (union + 1e-6)
    return {"dice": float(dice), "iou": float(iou), "reference_pixels": int(ref_bin.sum()), "challenger_pixels": int(chal_bin.sum())}


def load_image_from_bytes(buffer: bytes) -> np.ndarray:
    array = np.frombuffer(buffer, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image buffer.")
    return image


def save_upload(name: str, buffer: bytes) -> Path:
    upload_path = UPLOAD_DIR / name
    with open(upload_path, "wb") as fh:
        fh.write(buffer)
    return upload_path

