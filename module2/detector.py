"""
Template matching utilities for Module 2.

This module loads a curated template set, performs multi-scale correlation on
scenes, and returns detection results that can be consumed by the CLI, API,
or tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Paths ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = DATA_DIR / "templates"
METADATA_PATH = DATA_DIR / "metadata.json"

# Matching configuration -----------------------------------------------------

DEFAULT_THRESHOLD = 0.4
SCALE_FACTORS: Sequence[float] = (
    0.12,
    0.18,
    0.25,
    0.35,
    0.45,
    0.6,
    0.75,
    0.9,
    1.05,
    1.25,
    1.5,
    1.75,
    2.0,
    2.4,
    2.7,
)


@dataclass(frozen=True)
class Template:
    name: str  # logical label
    image: np.ndarray
    gray: np.ndarray
    edges: np.ndarray
    mask: Optional[np.ndarray]
    width: int
    height: int
    mean_color: np.ndarray
    hist_hs: np.ndarray
    variant: str  # filename used to load
    keypoints: Tuple
    descriptors: Optional[np.ndarray]

    @classmethod
    def from_path(cls, name: str, path: Path) -> "Template":
        orb = cv2.ORB_create(nfeatures=800)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read template at {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 20, 60)
        # Auto-crop to edge activity to drop background and increase specificity.
        ys, xs = np.nonzero(edges)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1] - 1, x2 + pad)
            y2 = min(image.shape[0] - 1, y2 + pad)
            image = image[y1 : y2 + 1, x1 : x2 + 1]
            gray = gray[y1 : y2 + 1, x1 : x2 + 1]
            edges = edges[y1 : y2 + 1, x1 : x2 + 1]
        # Additionally focus on the central content to reduce background leakage.
        h, w = image.shape[:2]
        cx1 = int(w * 0.2)
        cx2 = int(w * 0.8)
        cy1 = int(h * 0.2)
        cy2 = int(h * 0.8)
        image = image[cy1:cy2, cx1:cx2]
        gray = gray[cy1:cy2, cx1:cx2]
        edges = edges[cy1:cy2, cx1:cx2]
        mask = None
        mean_color = image.reshape(-1, 3).mean(axis=0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_hs = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist_hs, hist_hs, 0, 1, cv2.NORM_MINMAX)
        kp, des = orb.detectAndCompute(gray, None)
        return cls(
            name=name,
            image=image,
            gray=gray,
            edges=edges,
            mask=mask,
            width=image.shape[1],
            height=image.shape[0],
            mean_color=mean_color,
            hist_hs=hist_hs,
            variant=path.name,
            keypoints=kp,
            descriptors=des,
        )


@dataclass(frozen=True)
class TemplateConfig:
    """Per-template configuration (optional)."""

    threshold: float = DEFAULT_THRESHOLD
    scales: Optional[Sequence[float]] = None
    color_tolerance: Optional[float] = None


# Default: no special per-template overrides; everything uses DEFAULT_THRESHOLD / SCALE_FACTORS
TEMPLATE_CONFIG: Dict[str, TemplateConfig] = {}


# Metadata helpers -----------------------------------------------------------

def ensure_dataset() -> Dict[str, object]:
    """Load metadata JSON, raising a helpful error if missing."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_templates(metadata: Optional[Dict[str, object]] = None) -> List[Template]:
    """
    Load all PNG templates from the local data/templates directory.

    Each `*.png` file becomes one template whose label is the stem of the
    filename (without extension), e.g. `coke.png` -> label `"coke"`.
    The `metadata` parameter is accepted for API compatibility with other
    modules but is not required here.
    """
    templates: List[Template] = []
    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Template directory not found at {TEMPLATE_DIR}")

    for path in sorted(TEMPLATE_DIR.glob("*.png")):
        label = path.stem  # simple, readable label
        try:
            templates.append(Template.from_path(label, path))
        except FileNotFoundError as exc:
            # Skip unreadable templates but keep going for the others.
            print(f"Warning: {exc}")

    return templates


# Matching internals ---------------------------------------------------------

def _match_template_multiscale(
    scene_color: np.ndarray,
    template: Template,
    scales: Sequence[float],
) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """Return best score, location, and template size for multi-scale matching."""
    best_score = -1.0
    best_loc = (0, 0)
    best_size = (template.image.shape[1], template.image.shape[0])

    scene_h, scene_w = scene_color.shape[:2]

    for scale in scales:
        if scale <= 0:
            continue
        new_w = max(4, int(round(template.image.shape[1] * scale)))
        new_h = max(4, int(round(template.image.shape[0] * scale)))
        if new_w > scene_w or new_h > scene_h:
            continue

        tpl = cv2.resize(template.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(scene_color, tpl, cv2.TM_CCOEFF_NORMED)

        _, score, _, loc = cv2.minMaxLoc(result)
        if not np.isfinite(score):
            continue
        if score > best_score:
            best_score = float(score)
            best_loc = (int(loc[0]), int(loc[1]))
            best_size = (new_w, new_h)

    return best_score, best_loc, best_size


# Public detection functions -------------------------------------------------

def detect_scene(
    scene_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    template_thresholds: Optional[Dict[str, float]] = None,
    scales: Sequence[float] = SCALE_FACTORS,
) -> Dict[str, object]:
    """Detect objects using the built-in template set."""
    templates = load_templates()
    return detect_scene_with_templates(
        scene_path,
        templates,
        threshold,
        template_thresholds,
        scales,
        TEMPLATE_CONFIG,
    )


def detect_scene_with_templates(
    scene_path: Path,
    templates: Iterable[Template],
    threshold: float = DEFAULT_THRESHOLD,
    template_thresholds: Optional[Dict[str, float]] = None,
    scales: Sequence[float] = SCALE_FACTORS,
    template_config: Optional[Dict[str, TemplateConfig]] = None,
) -> Dict[str, object]:
    """Run multi-scale template matching for a single scene image."""
    scene = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
    if scene is None:
        raise FileNotFoundError(f"Unable to read scene image: {scene_path}")
    scene_color = cv2.GaussianBlur(scene, (3, 3), 0)
    scene_gray = cv2.cvtColor(scene_color, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1200)
    scene_kp, scene_des = orb.detectAndCompute(scene_gray, None)

    detections: List[Dict[str, object]] = []
    best_by_label: Dict[str, Tuple[float, Dict[str, object]]] = {}
    config_lookup = template_config or TEMPLATE_CONFIG

    for template in templates:
        config = config_lookup.get(template.name)
        template_scales = config.scales if config and config.scales else scales
        score, (x, y), (w, h) = _match_template_multiscale(scene_color, template, template_scales)

        if template_thresholds and template.name in template_thresholds:
            min_score = template_thresholds[template.name]
        elif config:
            min_score = config.threshold
        else:
            min_score = threshold

        if not np.isfinite(score):
            continue

        # Reject extremely tiny matches relative to the template (helps avoid background hits),
        # but allow smaller scales so that far-away / small objects like the Coke can are still detected.
        min_w = int(template.width * 0.2)
        min_h = int(template.height * 0.2)
        if w < min_w or h < min_h:
            continue

        x_adj = max(0, min(scene.shape[1] - 1, x))
        y_adj = max(0, min(scene.shape[0] - 1, y))
        w_adj = min(scene.shape[1] - x_adj, w)
        h_adj = min(scene.shape[0] - y_adj, h)
        if w_adj <= 0 or h_adj <= 0:
            continue

        region = scene[y_adj : y_adj + h_adj, x_adj : x_adj + w_adj]
        region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        reg_hist = cv2.calcHist([region_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(reg_hist, reg_hist, 0, 1, cv2.NORM_MINMAX)
        hist_score = cv2.compareHist(template.hist_hs, reg_hist, cv2.HISTCMP_CORREL)

        fused = float(score)

        if config and config.color_tolerance is not None:
            region_mean = region.reshape(-1, 3).mean(axis=0)
            diff = float(np.linalg.norm(region_mean - template.mean_color))
            if diff > config.color_tolerance:
                continue

        if fused < min_score:
            continue

        current = best_by_label.get(template.name)
        det_payload = {
            "label": template.name,
            "confidence": fused,
            "box": {"left": x_adj, "top": y_adj, "right": x_adj + w_adj, "bottom": y_adj + h_adj},
            "variant": template.variant,
        }
        if current is None or fused > current[0]:
            best_by_label[template.name] = (fused, det_payload)

    if best_by_label:
        sorted_best = sorted(best_by_label.items(), key=lambda kv: kv[1][0], reverse=True)
        best_label, (best_score, best_det) = sorted_best[0]
        if len(sorted_best) == 1 or best_score - sorted_best[1][1][0] >= 0.05:
            detections.append(best_det)

    try:
        scene_repr = str(scene_path.relative_to(BASE_DIR))
    except ValueError:
        scene_repr = str(scene_path)

    return {
        "scene": scene_repr,
        "detections": detections,
        "threshold": threshold,
    }


def annotate_detections(scene: np.ndarray, detections: Iterable[Dict[str, object]]) -> np.ndarray:
    """Draw detection boxes on a copy of the scene."""
    annotated = scene.copy()
    for det in detections:
        box = det["box"]
        p1 = (int(box["left"]), int(box["top"]))
        p2 = (int(box["right"]), int(box["bottom"]))
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
        label = f"{det['label']} {det.get('confidence', 0):.2f}"
        cv2.putText(
            annotated,
            label,
            (p1[0], max(15, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union helper used by the evaluation CLI."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
