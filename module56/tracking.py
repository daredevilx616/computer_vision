import base64
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import cv2
import numpy as np


@dataclass
class Marker:
    id: Union[int, str]
    corners: List[List[float]]


def _to_data_url(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode image buffer.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _bytes_to_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image buffer")
    return img


def _resize_max(image: np.ndarray, max_dim: int = 960) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(h, w) / float(max_dim)
    if scale <= 1.0:
        return image
    new_size = (int(w / scale), int(h / scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _clamp_int(value: float, min_val: int, max_val: int) -> int:
    return max(min_val, min(int(round(value)), max_val))


def detect_aruco_backend(frame_bytes: bytes, dict_name: str = "DICT_4X4_50", max_dim: int = 960) -> Dict:
    img = _resize_max(_bytes_to_image(frame_bytes), max_dim=max_dim)
    print(f"[ArUco Backend] Image shape after resize: {img.shape}, dict: {dict_name}")

    aruco = cv2.aruco
    dict_id = getattr(aruco, dict_name, None)
    if dict_id is None:
        print(f"[ArUco Backend] Unknown dictionary {dict_name}, falling back to DICT_4X4_50")
        dict_id = aruco.DICT_4X4_50
        dict_name = "DICT_4X4_50"

    dictionary = aruco.getPredefinedDictionary(dict_id)

    # Fix: Use DetectorParameters() instead of deprecated DetectorParameters_create()
    try:
        params = aruco.DetectorParameters()
        # Tune parameters for better detection
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.03
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.03
        params.minCornerDistanceRate = 0.05
        params.minDistanceToBorder = 3
        params.minMarkerDistanceRate = 0.05
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
    except AttributeError:
        # Fallback for older OpenCV versions
        params = aruco.DetectorParameters_create()

    # Try detection on original image
    corners, ids, _ = aruco.detectMarkers(img, dictionary, parameters=params)
    print(f"[ArUco Backend] Original image detection: {len(ids) if ids is not None else 0} markers")

    # If no markers found, try with contrast enhancement
    if ids is None or len(ids) == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        corners, ids, _ = aruco.detectMarkers(enhanced_bgr, dictionary, parameters=params)
        print(f"[ArUco Backend] Enhanced image detection: {len(ids) if ids is not None else 0} markers")
        if ids is not None and len(ids) > 0:
            img = enhanced_bgr

    markers: List[Marker] = []
    aruco_markers = []  # Separate list for ArUco markers for drawing
    
    if ids is not None:
        for idx, c in enumerate(corners):
            marker_id = int(ids[idx][0])
            marker_corners = c.reshape(-1, 2).tolist()
            markers.append(Marker(id=marker_id, corners=marker_corners))
            aruco_markers.append((marker_id, c))

    # QR fallback/augmentation
    qr_detector = cv2.QRCodeDetector()
    qr_data, qr_points, _ = qr_detector.detectAndDecode(img)
    if qr_points is not None and len(qr_points) > 0:
        pts = qr_points.reshape(-1, 2).tolist()
        markers.append(Marker(id=qr_data if qr_data else "QR", corners=pts))

    annotated = img.copy()
    
    # Draw ArUco markers
    if aruco_markers:
        aruco_corners = [np.array(m[1], dtype=np.float32) for m in aruco_markers]
        aruco_ids = np.array([[m[0]] for m in aruco_markers], dtype=np.int32)
        aruco.drawDetectedMarkers(annotated, aruco_corners, aruco_ids)
    
    # Draw QR codes
    for m in markers:
        if isinstance(m.id, str):
            pts = np.array(m.corners, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(annotated, [pts], True, (0, 200, 255), 2)
            cv2.putText(
                annotated,
                str(m.id),
                (int(pts[0][0]) + 4, int(pts[0][1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

    total = len(markers)
    aruco_count = len(aruco_markers)
    qr_count = 1 if qr_points is not None and len(qr_points) > 0 else 0

    if total == 0:
        message = (
            f"No markers found using {dict_name}. "
            "Try: (1) Different dictionary, (2) Increase screen brightness, "
            "(3) Reduce glare/reflections, (4) Move closer to camera, "
            "(5) Use printed markers instead of screen."
        )
    else:
        parts = []
        if aruco_count > 0:
            marker_ids = [m[0] for m in aruco_markers]
            parts.append(f"{aruco_count} ArUco marker(s) (IDs: {marker_ids})")
        if qr_count > 0:
            parts.append(f"{qr_count} QR code(s)")
        message = f"Detected {' + '.join(parts)} via {dict_name}"

    print(f"[ArUco Backend] Final result: {message}")

    return {
        "count": total,
        "markers": [{"id": m.id, "corners": m.corners} for m in markers],
        "visual": _to_data_url(annotated),
        "dictionary": dict_name,
        "message": message,
        "aruco_count": aruco_count,
        "qr_count": qr_count,
    }


def _color_distance(hsv_a: np.ndarray, hsv_b: np.ndarray) -> np.ndarray:
    """Calculate color distance in HSV space"""
    dh = np.minimum(np.abs(hsv_a[..., 0] - hsv_b[0]), 180 - np.abs(hsv_a[..., 0] - hsv_b[0])) / 90.0
    ds = np.abs(hsv_a[..., 1] - hsv_b[1]) / 255.0
    dv = np.abs(hsv_a[..., 2] - hsv_b[2]) / 255.0
    return dh * 0.6 + ds * 0.3 + dv * 0.1


def markerless_step_backend(
    frame_bytes: bytes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: Optional[List[float]] = None,
    search_pad: float = 40.0,
) -> Dict:
    """
    Markerless tracking using color-based template matching
    
    Args:
        frame_bytes: Current frame as bytes
        x, y: Top-left corner of bounding box
        w, h: Width and height of bounding box
        color: Optional reference color [H, S, V]
        search_pad: Search padding around the box
    """
    img = _bytes_to_image(frame_bytes)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]

    # Clamp initial box coordinates
    x0 = _clamp_int(x, 0, width - 1)
    y0 = _clamp_int(y, 0, height - 1)
    x1 = _clamp_int(x + w, x0 + 1, width)  # Ensure x1 > x0
    y1 = _clamp_int(y + h, y0 + 1, height)  # Ensure y1 > y0

    roi = hsv[y0:y1, x0:x1]
    if roi.size == 0:
        raise ValueError("ROI outside frame or has zero area")

    # Get reference color
    if color is not None:
        ref_color = np.array(color, dtype=np.float32)
    else:
        ref_color = np.mean(roi.reshape(-1, 3), axis=0)

    # Define search region
    search_x0 = _clamp_int(x0 - search_pad, 0, width - 1)
    search_y0 = _clamp_int(y0 - search_pad, 0, height - 1)
    search_x1 = _clamp_int(x1 + search_pad, search_x0 + 1, width)
    search_y1 = _clamp_int(y1 + search_pad, search_y0 + 1, height)

    search_region = hsv[search_y0:search_y1, search_x0:search_x1]
    if search_region.size == 0:
        raise ValueError("Search region empty")

    # Downsample for efficiency
    stride = 2
    small = search_region[::stride, ::stride]
    
    # Calculate color distance
    dist = _color_distance(small.astype(np.float32), ref_color)
    mask = dist < 0.22
    ys, xs = np.where(mask)

    # Update box based on matches
    if len(xs) < 10:
        new_box = {"x": float(x0), "y": float(y0), "w": float(x1 - x0), "h": float(y1 - y0)}
        message = "Low confidence; returning original box"
        confidence = 0.0
    else:
        cx = (xs.mean() * stride) + search_x0
        cy = (ys.mean() * stride) + search_y0
        new_x = _clamp_int(cx - w / 2, 0, width - 1)
        new_y = _clamp_int(cy - h / 2, 0, height - 1)
        new_box = {
            "x": float(new_x),
            "y": float(new_y),
            "w": float(_clamp_int(w, 5, width)),
            "h": float(_clamp_int(h, 5, height)),
        }
        message = "Tracking updated"
        confidence = min(len(xs) / 1000.0, 1.0)

    # Draw visualization
    annotated = img.copy()
    box_x = int(new_box["x"])
    box_y = int(new_box["y"])
    box_w = int(new_box["w"])
    box_h = int(new_box["h"])
    
    # Draw bounding box
    cv2.rectangle(
        annotated,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (0, 255, 255),
        2,
    )
    
    # Draw center point
    center_x = box_x + box_w // 2
    center_y = box_y + box_h // 2
    cv2.circle(annotated, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Draw confidence text
    cv2.putText(
        annotated,
        f"Conf: {confidence:.2f}",
        (box_x, box_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return {
        "box": new_box,
        "hits": int(len(xs)),
        "confidence": float(confidence),
        "ref_color": {"h": float(ref_color[0]), "s": float(ref_color[1]), "v": float(ref_color[2])},
        "message": message,
        "visual": _to_data_url(annotated),
    }