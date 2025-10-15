"""
Reusable detection utilities for Module 2 template matching experiments.

This module centralises template loading and matching so that both the
command-line scripts and the web API can invoke a consistent pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .generate_dataset import (
    BASE_DIR,
    DATA_DIR,
    METADATA_PATH,
    TEMPLATE_DIR,
    generate_dataset,
)

DEFAULT_THRESHOLD = 0.68


@dataclass(frozen=True)
class Template:
    name: str
    image: np.ndarray
    width: int
    height: int


def ensure_dataset() -> Dict[str, object]:
    """Load metadata, generating the synthetic dataset if missing."""
    if not METADATA_PATH.exists():
        generate_dataset()
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        metadata: Dict[str, object] = json.load(fh)
    return metadata


def load_templates(metadata: Dict[str, object]) -> List[Template]:
    """Read template images from disk."""
    templates: List[Template] = []
    for entry in metadata.get("templates", []):
        path = BASE_DIR / entry["path"]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to read template image at {path}")
        templates.append(Template(name=entry["name"], image=image, width=image.shape[1], height=image.shape[0]))
    return templates


def match_template(scene: np.ndarray, template: Template) -> Tuple[float, Tuple[int, int]]:
    """Return best correlation score and top-left coordinate for a template."""
    result = cv2.matchTemplate(scene, template.image, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def detect_scene(
    scene_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, object]:
    metadata = ensure_dataset()
    templates = load_templates(metadata)
    return detect_scene_with_templates(scene_path, templates, threshold)


def detect_scene_with_templates(
    scene_path: Path,
    templates: Iterable[Template],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, object]:
    """
    Run template matching on a single scene image.

    Returns a dictionary describing detections and the path to the scene.
    """
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene image does not exist: {scene_path}")

    scene = cv2.imread(str(scene_path))
    if scene is None:
        raise ValueError(f"Could not load scene image: {scene_path}")

    detections: List[Dict[str, object]] = []
    for template in templates:
        score, (x, y) = match_template(scene, template)
        if score < threshold:
            continue
        detections.append(
            {
                "label": template.name,
                "confidence": score,
                "box": {
                    "left": x,
                    "top": y,
                    "right": x + template.width,
                    "bottom": y + template.height,
                },
            }
        )

    try:
        scene_repr = str(scene_path.relative_to(BASE_DIR))
    except ValueError:
        scene_repr = str(scene_path)

    return {
        "scene": scene_repr,
        "detections": detections,
        "threshold": threshold,
    }


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union helper for evaluation."""
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


def annotate_detections(scene: np.ndarray, detections: Iterable[Dict[str, object]]) -> np.ndarray:
    """Draw detection boxes onto a scene copy."""
    annotated = scene.copy()
    for det in detections:
        box = det["box"]
        cv2.rectangle(
            annotated,
            (int(box["left"]), int(box["top"])),
            (int(box["right"]), int(box["bottom"])),
            (0, 255, 0),
            2,
        )
        label = det["label"]
        score = det.get("confidence", 0.0)
        cv2.putText(
            annotated,
            f"{label} {score:.2f}",
            (int(box["left"]), max(12, int(box["top"] - 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated
