"""
Template matching experiment for Module 2.

Usage:
    python -m module2.run_template_matching                 # evaluate full synthetic dataset
    python -m module2.run_template_matching --scene path    # process a single scene image

The CLI now underpins the web API integration by emitting JSON for a
single-scene invocation. Batch evaluation retains the previous summary
metrics across all generated scenes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2

from .detector import (
    DEFAULT_THRESHOLD,
    annotate_detections,
    compute_iou,
    detect_scene_with_templates,
    ensure_dataset,
    load_templates,
)
from .generate_dataset import BASE_DIR

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate_scene(scene_entry: Dict[str, object], templates, threshold: float) -> Dict[str, object]:
    scene_path = BASE_DIR / scene_entry["path"]
    detection = detect_scene_with_templates(scene_path, templates, threshold)

    scene = cv2.imread(str(scene_path))
    annotated = annotate_detections(scene, detection["detections"])
    output_path = OUTPUT_DIR / f"{scene_entry['name']}_detections.png"
    cv2.imwrite(str(output_path), annotated)

    gt_by_shape = {item["shape"]: item for item in scene_entry["placements"]}
    matches = []
    for det in detection["detections"]:
        shape = det["label"]
        pred_box = (
            det["box"]["left"],
            det["box"]["top"],
            det["box"]["right"],
            det["box"]["bottom"],
        )
        if shape not in gt_by_shape:
            matches.append({"template": shape, "score": det["confidence"], "iou": 0.0})
            continue
        gt = gt_by_shape[shape]
        gt_box = (gt["left"], gt["top"], gt["left"] + gt["width"], gt["top"] + gt["height"])
        matches.append({"template": shape, "score": det["confidence"], "iou": compute_iou(pred_box, gt_box)})

    present_templates = {item["shape"] for item in scene_entry["placements"]}
    tp = sum(1 for m in matches if m["iou"] >= 0.5 and m["template"] in present_templates)
    precision = tp / len(matches) if matches else 0.0
    recall = tp / len(present_templates) if present_templates else 0.0

    return {
        "scene": scene_entry["name"],
        "detections": detection["detections"],
        "metrics": {
            "precision_at_0.5": precision,
            "recall_at_0.5": recall,
        },
        "output_image": str(output_path.relative_to(BASE_DIR)),
    }


def run_dataset_evaluation(threshold: float) -> Dict[str, object]:
    metadata = ensure_dataset()
    templates = load_templates(metadata)

    results = [evaluate_scene(scene_entry, templates, threshold) for scene_entry in metadata["scenes"]]
    summary = {
        "results": results,
        "threshold": threshold,
    }

    result_path = OUTPUT_DIR / "template_matching_results.json"
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote detection renders to {OUTPUT_DIR.relative_to(BASE_DIR)}")
    print(f"Metrics saved to {result_path.relative_to(BASE_DIR)}")
    for item in results:
        metrics = item["metrics"]
        print(
            f"  {item['scene']}: precision@0.5={metrics['precision_at_0.5']:.2f}, "
            f"recall@0.5={metrics['recall_at_0.5']:.2f}"
        )
    return summary


def run_single_scene(scene_path: Path, threshold: float) -> Dict[str, object]:
    metadata = ensure_dataset()
    templates = load_templates(metadata)
    detection = detect_scene_with_templates(scene_path, templates, threshold)

    scene = cv2.imread(str(scene_path))
    if scene is None:
        raise ValueError(f"Failed to load scene: {scene_path}")
    annotated = annotate_detections(scene, detection["detections"])

    OUTPUT_DIR.mkdir(exist_ok=True)
    annotated_name = Path(scene_path).stem + "_detections.png"
    annotated_path = OUTPUT_DIR / annotated_name
    cv2.imwrite(str(annotated_path), annotated)

    try:
        display_path = str(annotated_path.relative_to(BASE_DIR))
    except ValueError:
        display_path = str(annotated_path)
    detection["annotated_image"] = display_path
    return detection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Template matching utilities for Module 2 assignment.")
    parser.add_argument("--scene", type=Path, help="Path to a specific scene image to process.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Correlation threshold for accepting detections (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (useful for programmatic consumption).",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.scene:
        result = run_single_scene(args.scene, args.threshold)
    else:
        result = run_dataset_evaluation(args.threshold)

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
