"""
Synthetic dataset generator for Module 2 experiments.

The assignment requires template matching with ten distinct objects
whose templates come from separate scenes. This script procedurally
creates a reproducible miniature dataset: ten template images plus a
few cluttered scenes that contain those objects alongside mild noise.

Running the script populates:
    module2/data/templates/*.png
    module2/data/scenes/*.png
    module2/data/metadata.json

All assets are lightweight (PNG) and deterministic so the dataset can
be regenerated any time without shipping binary files in version
control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = DATA_DIR / "templates"
SCENE_DIR = DATA_DIR / "scenes"
METADATA_PATH = DATA_DIR / "metadata.json"

TEMPLATE_SIZE = 96  # square templates
SCENE_SIZE = 320

# BGR color palette
COLORS: Dict[str, Tuple[int, int, int]] = {
    "blue": (180, 90, 30),
    "green": (60, 160, 60),
    "red": (60, 70, 220),
    "yellow": (40, 215, 240),
    "magenta": (200, 70, 200),
    "cyan": (230, 200, 60),
    "orange": (60, 140, 255),
    "white": (245, 245, 245),
    "purple": (150, 80, 180),
    "aqua": (220, 180, 80),
}

SHAPE_NAMES = [
    "blue_circle",
    "green_square",
    "red_triangle",
    "yellow_diamond",
    "magenta_cross",
    "cyan_ring",
    "orange_pentagon",
    "white_plus",
    "purple_arrow",
    "aqua_halfmoon",
]


def draw_shape(canvas: np.ndarray, shape_name: str) -> None:
    """Render a deterministic geometric glyph into the template canvas."""

    h, w = canvas.shape[:2]
    center = (w // 2, h // 2)

    if shape_name == "blue_circle":
        cv2.circle(canvas, center, 32, COLORS["blue"], -1, lineType=cv2.LINE_AA)
    elif shape_name == "green_square":
        cv2.rectangle(canvas, (20, 20), (w - 20, h - 20), COLORS["green"], -1, lineType=cv2.LINE_AA)
    elif shape_name == "red_triangle":
        pts = np.array([[w // 2, 18], [22, h - 22], [w - 22, h - 22]], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], COLORS["red"], lineType=cv2.LINE_AA)
    elif shape_name == "yellow_diamond":
        pts = np.array([[w // 2, 18], [22, h // 2], [w // 2, h - 18], [w - 22, h // 2]], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], COLORS["yellow"], lineType=cv2.LINE_AA)
    elif shape_name == "magenta_cross":
        cv2.rectangle(canvas, (40, 18), (56, h - 18), COLORS["magenta"], -1, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, (18, 40), (w - 18, 56), COLORS["magenta"], -1, lineType=cv2.LINE_AA)
    elif shape_name == "cyan_ring":
        cv2.circle(canvas, center, 34, COLORS["cyan"], 12, lineType=cv2.LINE_AA)
    elif shape_name == "orange_pentagon":
        pts = np.array(
            [
                [w // 2, 14],
                [20, h // 2 - 12],
                [34, h - 22],
                [w - 34, h - 22],
                [w - 20, h // 2 - 12],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [pts], COLORS["orange"], lineType=cv2.LINE_AA)
    elif shape_name == "white_plus":
        cv2.rectangle(canvas, (38, 12), (58, h - 12), COLORS["white"], -1, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, (12, 38), (w - 12, 58), COLORS["white"], -1, lineType=cv2.LINE_AA)
    elif shape_name == "purple_arrow":
        pts = np.array(
            [
                [22, 22],
                [w - 24, h // 2],
                [22, h - 22],
                [34, h // 2 + 8],
                [34, h // 2 - 8],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [pts], COLORS["purple"], lineType=cv2.LINE_AA)
    elif shape_name == "aqua_halfmoon":
        cv2.circle(canvas, center, 36, COLORS["aqua"], -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (center[0] + 14, center[1]), 32, (10, 10, 10), -1, lineType=cv2.LINE_AA)
    else:
        raise ValueError(f"Unknown shape '{shape_name}'")


def create_template(shape_name: str) -> np.ndarray:
    """Generate one template tile."""
    canvas = np.full((TEMPLATE_SIZE, TEMPLATE_SIZE, 3), 18, dtype=np.uint8)
    draw_shape(canvas, shape_name)
    return canvas


def overlay_shape(scene: np.ndarray, shape_img: np.ndarray, top: int, left: int) -> None:
    """Alpha-compose the shape onto the scene using a hard mask."""
    h, w = shape_img.shape[:2]
    roi = scene[top : top + h, left : left + w]
    gray = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
    mask = gray > 25

    roi[mask] = shape_img[mask]
    scene[top : top + h, left : left + w] = roi


def make_background(rng: np.random.Generator) -> np.ndarray:
    """Create a textured background for a scene."""
    gradient = np.linspace(24, 70, SCENE_SIZE, dtype=np.float32)
    base = np.stack([gradient] * SCENE_SIZE, axis=1)
    background = np.dstack([
        base + rng.uniform(-8, 8, size=(SCENE_SIZE, SCENE_SIZE)),
        base + rng.uniform(-16, 16, size=(SCENE_SIZE, SCENE_SIZE)),
        base + rng.uniform(-6, 6, size=(SCENE_SIZE, SCENE_SIZE)),
    ])
    noise = rng.normal(0, 5, size=background.shape)
    scene = np.clip(background + noise + 40, 0, 255).astype(np.uint8)
    return scene


SCENES = [
    {"name": "scene_boardwalk", "seed": 11, "shapes": ["blue_circle", "yellow_diamond", "white_plus", "purple_arrow"]},
    {"name": "scene_museum", "seed": 29, "shapes": ["green_square", "cyan_ring", "aqua_halfmoon", "red_triangle"]},
    {"name": "scene_plaza", "seed": 47, "shapes": ["magenta_cross", "orange_pentagon", "blue_circle", "cyan_ring"]},
]


def create_scene(scene_spec: Dict[str, object]) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Build a scene image and record placements metadata."""
    rng = np.random.default_rng(scene_spec["seed"])
    scene = make_background(rng)

    placements: List[Dict[str, object]] = []
    occupied: List[Tuple[int, int, int, int]] = []

    def overlaps(box: Tuple[int, int, int, int]) -> bool:
        top, left, bottom, right = box
        for ot, ol, ob, oright in occupied:
            separated = bottom <= ot or top >= ob or right <= ol or left >= oright
            if not separated:
                return True
        return False

    for shape_name in scene_spec["shapes"]:
        template = create_template(shape_name)
        h, w = template.shape[:2]

        for _ in range(100):  # attempt to find a free slot
            top = int(rng.integers(12, SCENE_SIZE - h - 12))
            left = int(rng.integers(12, SCENE_SIZE - w - 12))
            box = (top, left, top + h, left + w)
            if not overlaps(box):
                break
        else:
            raise RuntimeError("Could not place shape without overlap")

        overlay_shape(scene, template, top, left)
        placements.append(
            {
                "shape": shape_name,
                "top": top,
                "left": left,
                "height": h,
                "width": w,
            }
        )
        occupied.append(box)

    return scene, placements


def generate_dataset() -> Dict[str, object]:
    """Create templates, scenes, and metadata on disk."""
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    SCENE_DIR.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, object] = {
        "templates": [],
        "scenes": [],
    }

    for shape_name in SHAPE_NAMES:
        template = create_template(shape_name)
        path = TEMPLATE_DIR / f"{shape_name}.png"
        cv2.imwrite(str(path), template)
        metadata["templates"].append({"name": shape_name, "path": str(path.relative_to(BASE_DIR))})

    for scene_spec in SCENES:
        scene, placements = create_scene(scene_spec)
        path = SCENE_DIR / f"{scene_spec['name']}.png"
        cv2.imwrite(str(path), scene)
        metadata["scenes"].append(
            {
                "name": scene_spec["name"],
                "path": str(path.relative_to(BASE_DIR)),
                "placements": placements,
            }
        )

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


def main() -> None:
    metadata = generate_dataset()
    print(f"Generated {len(metadata['templates'])} templates and {len(metadata['scenes'])} scenes.")
    print(f"Metadata saved to {METADATA_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
