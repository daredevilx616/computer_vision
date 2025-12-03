from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_default_base = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("MODULE3_BASE_DIR", _default_base)).resolve()
UPLOAD_DIR = Path(os.getenv("MODULE3_UPLOAD_DIR", BASE_DIR / "uploads"))
OUTPUT_DIR = Path(os.getenv("MODULE3_OUTPUT_DIR", BASE_DIR / "output"))
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
        # Smooth + auto Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        v = np.median(blurred)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, lower, upper, L2gradient=True)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        min_area = 0.01 * h * w
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        contours.sort(key=cv2.contourArea, reverse=True)

        # Sample sparse keypoints along dominant contours
        for cnt in contours[:3]:
            perim = cv2.arcLength(cnt, True)
            step = max(int(perim / 60), 5)
            pts = cnt[::step]
            for p in pts:
                x, y = int(p[0][0]), int(p[0][1])
                cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)

        mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(overlay, 0.9, mask, 0.1, 0)
    elif mode == "corner":
        # Constrain detection to the dominant object region to avoid background noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        v = np.median(blurred)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, lower, upper, L2gradient=True)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_roi = np.zeros_like(gray)
        h, w = gray.shape[:2]
        image_area = h * w

        primary_contour = None
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 0.05 * image_area or area > 0.95 * image_area:
                    continue
                # Avoid full-frame/background by checking border touch
                xs = cnt[:, 0, 0]
                ys = cnt[:, 0, 1]
                if xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
                    continue
                primary_contour = cnt
                break
        if primary_contour is not None:
            cv2.drawContours(mask_roi, [primary_contour], -1, 255, thickness=-1)
        else:
            mask_roi[:, :] = 255  # fallback: full image

        corner_points = []
        if primary_contour is not None:
            perim = cv2.arcLength(primary_contour, True)
            epsilon = 0.01 * perim
            approx = cv2.approxPolyDP(primary_contour, epsilon, True)
            for p in approx:
                corner_points.append((int(p[0][0]), int(p[0][1])))

        # Use Shi-Tomasi corners on the ROI only
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=20,
            qualityLevel=0.05,
            minDistance=20,
            blockSize=7,
            useHarrisDetector=False,
            mask=mask_roi,
        )
        if corners is not None:
            for pt in corners:
                x, y = pt.ravel().astype(int)
                corner_points.append((x, y))

        # Deduplicate close points
        deduped = []
        seen = set()
        for x, y in corner_points:
            key = (int(x // 3), int(y // 3))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((x, y))

        for x, y in deduped:
            cv2.circle(overlay, (x, y), 6, (0, 0, 255), -1)
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

    # Aggregate marker information
    all_marker_points = np.vstack([c.reshape(-1, 2) for c in corners]).astype(np.int32)
    marker_area = cv2.contourArea(cv2.convexHull(all_marker_points))
    marker_centers = [c.reshape(-1, 2).mean(axis=0) for c in corners]
    marker_centers_arr = np.vstack(marker_centers)
    image_area = image.shape[0] * image.shape[1]

    # Contrast-limited grayscale helps edge stability across lighting
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Focus processing near the markers while allowing some context
    hull = cv2.convexHull(all_marker_points)
    bx, by, bw, bh = cv2.boundingRect(hull)
    margin = int(max(bw, bh) * 1.25 + 60)  # allow object offset from marker
    x1 = max(0, bx - margin)
    y1 = max(0, by - margin)
    x2 = min(image.shape[1], bx + bw + margin)
    y2 = min(image.shape[0], by + bh + margin)

    roi = blurred[y1:y2, x1:x2]
    if roi.size == 0:
        roi = blurred
        x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]

    # Adaptive Canny thresholds based on image statistics
    v = np.median(roi)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges_roi = cv2.Canny(roi, lower, upper, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_roi = cv2.morphologyEx(edges_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges_roi = cv2.dilate(edges_roi, kernel, iterations=1)

    # Place ROI edges back into full-size canvas for debugging output
    combined_edges = np.zeros_like(gray)
    combined_edges[y1:y2, x1:x2] = edges_roi

    contours_all, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(marker_area * 1.2, image_area * 0.01)
    max_area = image_area * 0.75

    candidate_contours: List[Tuple[float, float, np.ndarray]] = []
    for cnt in contours_all:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        if w_c == 0 or h_c == 0:
            continue

        # Prefer roughly compact shapes (reject very elongated rectangles like phones)
        aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
        if aspect_ratio > 2.2:
            continue

        marker_center_y = float(marker_centers_arr[:, 1].mean())
        if (y_c + h_c * 0.5) < (marker_center_y - 0.05 * image.shape[0]):
            continue

        # Distance of markers to contour (do not hard reject, use in scoring)
        distances = [cv2.pointPolygonTest(cnt, tuple(center), True) for center in marker_centers_arr]
        min_abs_distance = float(np.min(np.abs(distances)))

        mean_distance = float(np.mean(np.abs(distances)))
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull_area if hull_area > 0 else 0.0
        perimeter = cv2.arcLength(cnt, True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
        if compactness < 0.45:
            continue

        # Higher score favours contours near markers, reasonably solid, and sizable
        score = (
            0.45 * (1.0 / (1.0 + (min_abs_distance / max(image.shape[0], image.shape[1]))))
            + 0.2 * solidity
            + 0.2 * compactness
            + 0.15 * (area / max_area)
        )
        candidate_contours.append((score, area, cnt))

    object_contour: np.ndarray | None = None

    if candidate_contours:
        candidate_contours.sort(key=lambda x: (x[0], x[1]), reverse=True)
        object_contour = candidate_contours[0][2]
    else:
        # Secondary fallback: largest contour not touching image border
        border_safe = [
            c for c in contours_all
            if cv2.contourArea(c) >= min_area
            and not np.any(c[:, 0, 0] == 0)
            and not np.any(c[:, 0, 1] == 0)
            and not np.any(c[:, 0, 0] == image.shape[1] - 1)
            and not np.any(c[:, 0, 1] == image.shape[0] - 1)
        ]
        if border_safe:
            object_contour = max(border_safe, key=cv2.contourArea)
        else:
            # Target is expected below the marker; focus search there
            bottom_start = int(min(image.shape[0] - 1, marker_center_y))
            bottom_roi = enhanced[bottom_start:, :]
            if bottom_roi.size > 0:
                _, thr = cv2.threshold(bottom_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thr = cv2.bitwise_not(thr)  # object is brighter
                kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_b, iterations=2)
                thr = cv2.dilate(thr, kernel_b, iterations=1)
                cont_bottom, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cont_bottom = [c for c in cont_bottom if cv2.contourArea(c) >= min_area * 0.8]
                if cont_bottom:
                    c_sel = max(cont_bottom, key=cv2.contourArea)
                    c_sel[:, 0, 1] += bottom_start  # shift back to image coords
                    object_contour = c_sel
                    combined_edges = np.zeros_like(gray)
                    combined_edges[bottom_start:, :] = thr
            if object_contour is None:
                # Fallback: seed GrabCut with markers to avoid whole-image masks
                grabcut_mask = np.full(image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
                for marker in corners:
                    pts = marker.reshape(-1, 2).astype(int)
                    x_min, y_min = pts.min(axis=0) - 4
                    x_max, y_max = pts.max(axis=0) + 4
                    grabcut_mask[max(0, y_min):min(grabcut_mask.shape[0], y_max),
                                 max(0, x_min):min(grabcut_mask.shape[1], x_max)] = cv2.GC_FGD

                border = 8
                grabcut_mask[:border, :] = cv2.GC_BGD
                grabcut_mask[-border:, :] = cv2.GC_BGD
                grabcut_mask[:, :border] = cv2.GC_BGD
                grabcut_mask[:, -border:] = cv2.GC_BGD

                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
                gc_mask = np.where(
                    (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                    255,
                    0,
                ).astype(np.uint8)

                cont_gc, _ = cv2.findContours(gc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cont_gc:
                    object_contour = max(cont_gc, key=cv2.contourArea)
                    combined_edges = gc_mask
                else:
                    # Final fallback to marker hull
                    object_contour = cv2.convexHull(all_marker_points)

    # Smooth the contour for better visualization
    epsilon = 0.005 * cv2.arcLength(object_contour, True)
    object_contour = cv2.approxPolyDP(object_contour, epsilon, True)

    # Create mask from actual object boundary
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [object_contour], -1, 255, thickness=-1)

    # Optional: fill any holes in the mask
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill)

    # Get bounding rect for ROI visualization
    x, y, w, h = cv2.boundingRect(object_contour)
    margin = max(w, h) // 4
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    # Step 3: Create visualization
    overlay = image.copy()

    # Draw detected markers (green boxes with IDs)
    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)

    # Draw actual object boundary (thick red contour for visibility)
    cv2.drawContours(overlay, [object_contour], -1, (0, 0, 255), 4)

    # Add bounding box for reference (yellow)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Create blended result with mask overlay
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = mask  # Blue channel for mask visualization
    blended = cv2.addWeighted(overlay, 0.75, mask_colored, 0.25, 0)

    overlay_path = _write_image(blended, "module3_aruco_overlay")
    mask_path = _write_image(mask, "module3_aruco_mask")
    edges_path = _write_image(combined_edges, "module3_aruco_edges")

    return {
        "overlay": _to_data_url(blended),
        "mask": _to_data_url(mask),
        "edges": _to_data_url(combined_edges),
        "overlay_path": str(overlay_path.relative_to(BASE_DIR)),
        "mask_path": str(mask_path.relative_to(BASE_DIR)),
        "edges_path": str(edges_path.relative_to(BASE_DIR)),
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
