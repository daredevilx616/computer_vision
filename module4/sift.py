from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class KeyPoint:
    pt: Tuple[float, float]
    octave: int
    layer: int
    size: float
    angle: float


@dataclass
class SIFTResult:
    keypoints: List[KeyPoint]
    descriptors: np.ndarray


def _resize_if_needed(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(h, w) / max_size
    if scale <= 1:
        return image
    new_size = (int(w / scale), int(h / scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def gaussian_pyramid(image: np.ndarray, num_octaves: int = 4, scales_per_octave: int = 3, sigma: float = 1.6) -> List[List[np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    base = _resize_if_needed(gray)
    pyramid: List[List[np.ndarray]] = []
    k = 2 ** (1 / scales_per_octave)

    for octave in range(num_octaves):
        gaussian_images: List[np.ndarray] = []
        if octave == 0:
            octave_base = base
        else:
            octave_base = cv2.resize(pyramid[octave - 1][scales_per_octave], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        sigma_prev = 0.5
        sigmas = [sigma * (k ** i) for i in range(scales_per_octave + 3)]
        for idx, sigma_total in enumerate(sigmas):
            sigma_diff = np.sqrt(max(sigma_total**2 - sigma_prev**2, 1e-4))
            blurred = cv2.GaussianBlur(octave_base, (0, 0), sigma_diff)
            gaussian_images.append(blurred)
            sigma_prev = sigma_total
        pyramid.append(gaussian_images)
    return pyramid


def dog_pyramid(gaussian: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    dogs: List[List[np.ndarray]] = []
    for octave in gaussian:
        octave_dogs = []
        for i in range(1, len(octave)):
            octave_dogs.append(octave[i] - octave[i - 1])
        dogs.append(octave_dogs)
    return dogs


def detect_keypoints(gaussian: List[List[np.ndarray]], dogs: List[List[np.ndarray]], contrast_threshold: float = 0.015, edge_threshold: float = 10.0) -> List[KeyPoint]:
    """Detect keypoints with contrast and edge response filtering."""
    keypoints: List[KeyPoint] = []
    for octave_idx, octave_dogs in enumerate(dogs):
        for layer in range(1, len(octave_dogs) - 1):
            prev_img = octave_dogs[layer - 1]
            curr_img = octave_dogs[layer]
            next_img = octave_dogs[layer + 1]
            h, w = curr_img.shape
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    val = curr_img[y, x]
                    # Lower contrast threshold to detect more keypoints
                    if abs(val) < contrast_threshold:
                        continue

                    # Check if local extremum
                    patch = np.concatenate(
                        [
                            prev_img[y - 1 : y + 2, x - 1 : x + 2].ravel(),
                            curr_img[y - 1 : y + 2, x - 1 : x + 2].ravel(),
                            next_img[y - 1 : y + 2, x - 1 : x + 2].ravel(),
                        ]
                    )
                    if val > 0 and val >= patch.max():
                        is_extremum = True
                    elif val < 0 and val <= patch.min():
                        is_extremum = True
                    else:
                        is_extremum = False

                    if not is_extremum:
                        continue

                    # Edge response filtering (remove keypoints on edges)
                    dxx = curr_img[y, x + 1] + curr_img[y, x - 1] - 2 * curr_img[y, x]
                    dyy = curr_img[y + 1, x] + curr_img[y - 1, x] - 2 * curr_img[y, x]
                    dxy = (curr_img[y + 1, x + 1] - curr_img[y + 1, x - 1] - curr_img[y - 1, x + 1] + curr_img[y - 1, x - 1]) / 4.0

                    trace = dxx + dyy
                    det = dxx * dyy - dxy * dxy

                    if det <= 0:
                        continue

                    # Edge threshold check: reject keypoints with large principal curvature ratio
                    if (trace * trace) / det > ((edge_threshold + 1) ** 2) / edge_threshold:
                        continue

                    scale = 2 ** octave_idx
                    kp = KeyPoint(
                        pt=(float(x * scale), float(y * scale)),
                        octave=octave_idx,
                        layer=layer,
                        size=1.6 * (2 ** octave_idx) * (layer + 1),
                        angle=0.0,
                    )
                    keypoints.append(kp)
    return keypoints


def _gradient_at(image: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    h, w = image.shape
    x = min(max(x, 1), w - 2)
    y = min(max(y, 1), h - 2)
    dx = image[y, x + 1] - image[y, x - 1]
    dy = image[y - 1, x] - image[y + 1, x]
    magnitude = np.sqrt(dx * dx + dy * dy)
    angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    return magnitude, angle


def assign_orientations(gaussian: List[List[np.ndarray]], keypoints: List[KeyPoint]) -> None:
    """Assign dominant orientations to keypoints with Gaussian weighting."""
    for kp in keypoints:
        layer_img = gaussian[kp.octave][kp.layer]
        scale = 2 ** kp.octave
        x = int(round(kp.pt[0] / scale))
        y = int(round(kp.pt[1] / scale))

        # Extract region around keypoint
        y_start = max(0, y - 8)
        y_end = min(layer_img.shape[0], y + 9)
        x_start = max(0, x - 8)
        x_end = min(layer_img.shape[1], x + 9)

        region = layer_img[y_start:y_end, x_start:x_end]
        if region.size == 0:
            continue

        # Compute gradients
        gy, gx = np.gradient(region)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0

        # Apply Gaussian weighting (sigma = 1.5 * scale of keypoint)
        sigma = 1.5 * kp.size
        h, w = region.shape
        gauss_weight = np.zeros_like(magnitude)
        for i in range(h):
            for j in range(w):
                dy = i - (h // 2)
                dx = j - (w // 2)
                gauss_weight[i, j] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

        weighted_magnitude = magnitude * gauss_weight

        # Build orientation histogram
        hist, _ = np.histogram(orientation, bins=36, range=(0, 360), weights=weighted_magnitude)

        # Smooth histogram
        hist = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), [1, 1, 1], mode='valid') / 3.0

        # Find dominant orientation
        peak_idx = np.argmax(hist)
        kp.angle = float((peak_idx * 360) / 36)


def compute_descriptors(gaussian: List[List[np.ndarray]], keypoints: List[KeyPoint]) -> np.ndarray:
    """Compute SIFT descriptors with Gaussian weighting and improved normalization."""
    descriptors: List[np.ndarray] = []

    # Create Gaussian weight window (16x16)
    gaussian_window = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        for j in range(16):
            gaussian_window[i, j] = np.exp(-((i - 7.5) ** 2 + (j - 7.5) ** 2) / (2 * (8.0) ** 2))

    for kp in keypoints:
        layer_img = gaussian[kp.octave][kp.layer]
        scale = 2 ** kp.octave
        x = int(round(kp.pt[0] / scale))
        y = int(round(kp.pt[1] / scale))

        # Extract 16x16 patch around keypoint
        y_start = max(0, y - 8)
        y_end = min(layer_img.shape[0], y + 8)
        x_start = max(0, x - 8)
        x_end = min(layer_img.shape[1], x + 8)

        patch = layer_img[y_start:y_end, x_start:x_end]

        # Skip if patch is too small
        if patch.shape[0] < 10 or patch.shape[1] < 10:
            continue

        # Resize to 16x16 if needed
        if patch.shape != (16, 16):
            patch = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_LINEAR)

        # Compute gradients
        gy, gx = np.gradient(patch)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = (np.degrees(np.arctan2(gy, gx)) - kp.angle + 360.0) % 360.0

        # Apply Gaussian weighting
        magnitude = magnitude * gaussian_window

        # Build descriptor: 4x4 grid of 8-bin histograms
        descriptor = []
        for i in range(4):
            for j in range(4):
                cell_mag = magnitude[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4].ravel()
                cell_ori = orientation[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4].ravel()
                hist, _ = np.histogram(cell_ori, bins=8, range=(0, 360), weights=cell_mag)
                descriptor.extend(hist)

        descriptor = np.array(descriptor, dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(descriptor)
        if norm > 1e-6:
            descriptor = descriptor / norm

        # Clip values to 0.2 and renormalize (improves robustness to illumination)
        descriptor = np.clip(descriptor, 0, 0.2)
        norm = np.linalg.norm(descriptor)
        if norm > 1e-6:
            descriptor = descriptor / norm

        descriptors.append(descriptor)

    if not descriptors:
        return np.zeros((0, 128), dtype=np.float32)
    return np.vstack(descriptors)


def sift(image: np.ndarray) -> SIFTResult:
    pyramid = gaussian_pyramid(image)
    dogs = dog_pyramid(pyramid)
    keypoints = detect_keypoints(pyramid, dogs)
    assign_orientations(pyramid, keypoints)
    descriptors = compute_descriptors(pyramid, keypoints)
    return SIFTResult(keypoints=keypoints, descriptors=descriptors)


def match_descriptors(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75) -> List[Tuple[int, int, float]]:
    matches: List[Tuple[int, int, float]] = []
    if desc1.size == 0 or desc2.size == 0:
        return matches
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        if len(distances) < 2:
            continue
        idx = np.argsort(distances)
        best, second = distances[idx[0]], distances[idx[1]]
        if best < ratio * second:
            matches.append((i, idx[0], float(best)))
    return matches


def ransac_homography(matches: Sequence[Tuple[int, int, float]], kp1: Sequence[KeyPoint], kp2: Sequence[KeyPoint], iterations: int = 500, threshold: float = 3.0):
    if len(matches) < 4:
        raise RuntimeError("Not enough matches for homography.")

    pts1 = np.array([kp1[i].pt for i, _, _ in matches], dtype=np.float32)
    pts2 = np.array([kp2[j].pt for _, j, _ in matches], dtype=np.float32)

    best_h = None
    best_inliers: List[int] = []
    for _ in range(iterations):
        sample_idx = np.random.choice(len(matches), 4, replace=False)
        src = pts1[sample_idx]
        dst = pts2[sample_idx]
        h, status = cv2.findHomography(src, dst, 0)
        if h is None:
            continue
        projected = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), h).reshape(-1, 2)
        errors = np.linalg.norm(projected - pts2, axis=1)
        inliers = np.where(errors < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers.tolist()
            best_h = h

    if best_h is None:
        raise RuntimeError("Failed to compute homography.")
    return best_h, best_inliers


def draw_matches(image_a: np.ndarray, image_b: np.ndarray, kp_a: Sequence[KeyPoint], kp_b: Sequence[KeyPoint], matches: Sequence[Tuple[int, int, float]], inliers: Sequence[int]) -> np.ndarray:
    """Draw only inlier matches in green; omit outliers for clarity."""
    h = max(image_a.shape[0], image_b.shape[0])
    w = image_a.shape[1] + image_b.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: image_a.shape[0], : image_a.shape[1]] = image_a
    canvas[: image_b.shape[0], image_a.shape[1] :] = image_b

    offset = image_a.shape[1]
    inlier_set = set(inliers)
    for idx in inlier_set:
        i, j, _ = matches[idx]
        pt1 = tuple(map(int, kp_a[i].pt))
        pt2 = (int(kp_b[j].pt[0] + offset), int(kp_b[j].pt[1]))
        color = (0, 255, 0)
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
    return canvas
