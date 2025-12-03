"""
Run template matching on a scene image and return JSON results.
This script is called by the web API to detect ALL templates in a scene.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .detector import detect_scene, load_templates, annotate_detections, BASE_DIR

_default_output = BASE_DIR / "output"
if os.getenv("VERCEL"):
    OUTPUT_DIR = Path(os.getenv("MODULE2_OUTPUT_DIR", "/tmp/module2_output"))
else:
    OUTPUT_DIR = Path(os.getenv("MODULE2_OUTPUT_DIR", _default_output))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_detection(scene_path: Path, threshold: float = 0.7, only: list[str] | None = None) -> dict:
    """
    Run template matching on a scene image with ALL available templates.

    Returns a dict with:
    - detections: list of all detected objects
    - threshold: the threshold used
    - annotated_image: path to the annotated output image
    """
    # Load the scene
    scene = cv2.imread(str(scene_path))
    if scene is None:
        raise FileNotFoundError(f"Cannot load scene: {scene_path}")

    # Load all templates
    templates = load_templates()

    # Import the multi-scale matching function
    from .detector import _match_template_multiscale, SCALE_FACTORS

    # Prepare scene
    scene_color = cv2.GaussianBlur(scene, (3, 3), 0)

    # Optional per-label min thresholds (allows leniency for certain objects)
    per_label_min_threshold = {
        "stop-sign": 0.5,    # keep stop easy enough to trigger
        "yield-sign": 0.6,
        "exit-sign": 0.3,
        "no-right-sign": 0.7,
        "one-way-sign": 0.6,
        "heart": 0.7,
        "diamond": 0.7,
        "spades": 0.7,
        "clubs": 0.7,
    }

    def infer_allowed_labels(name: str) -> set[str]:
        n = name.lower()
        if "stop" in n and "yield" in n:
            return {"stop-sign", "yield-sign"}
        if "stop" in n:
            return {"stop-sign"}
        if "yield" in n:
            return {"yield-sign"}
        if "exit" in n:
            return {"exit-sign"}
        if "no-right" in n or "noright" in n:
            return {"no-right-sign"}
        if "one-way" in n or "oneway" in n or "one_way" in n:
            return {"one-way-sign"}
        if "cheerios" in n:
            return {"cheerios-logo"}
        if "ten-of-hearts" in n or "hearts" in n:
            return {"heart"}
        if "diamond" in n:
            return {"diamond"}
        if "spade" in n:
            return {"spades"}
        if "club" in n:
            return {"clubs"}
        if "all-four" in n or "cards" in n:
            return {"heart", "diamond", "spades", "clubs"}
        return set()

    # Run detection for each template and collect all matches
    all_detections = []

    for template in templates:
        label_lc = template.name.lower()
        eff_thresh = min(threshold, per_label_min_threshold.get(label_lc, threshold))

        # Run multi-scale matching
        score, (x, y), (w, h) = _match_template_multiscale(
            scene_color,
            template,
            SCALE_FACTORS
        )

        # Check if detection meets threshold
        if score >= eff_thresh:
            # Ensure coordinates are within bounds
            x_adj = max(0, min(scene.shape[1] - 1, x))
            y_adj = max(0, min(scene.shape[0] - 1, y))
            w_adj = min(scene.shape[1] - x_adj, w)
            h_adj = min(scene.shape[0] - y_adj, h)

            if w_adj > 0 and h_adj > 0:
                # Extract the matched region for validation
                region = scene[y_adj : y_adj + h_adj, x_adj : x_adj + w_adj]

                # Optional histogram guard for certain labels (signs, suits)
                hist_guard_labels = {"heart", "diamond", "spades", "clubs"}
                if label_lc in hist_guard_labels:
                    guard_thresh = 0.1
                    region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    reg_hist = cv2.calcHist([region_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                    cv2.normalize(reg_hist, reg_hist, 0, 1, cv2.NORM_MINMAX)
                    hist_score = cv2.compareHist(template.hist_hs, reg_hist, cv2.HISTCMP_CORREL)
                    if hist_score < guard_thresh:
                        continue

                # Shape validation: Check aspect ratio matches template
                # This helps distinguish hearts (wider) from diamonds (taller)
                template_aspect = template.width / template.height if template.height > 0 else 1.0
                match_aspect = w_adj / h_adj if h_adj > 0 else 1.0
                aspect_diff = abs(template_aspect - match_aspect)

                # If aspect ratio is very different, penalize the score
                # Hearts are ~0.9-1.1 aspect ratio (roughly square)
                # Diamonds are also ~0.8-1.2 (also roughly square)
                # So this helps a bit but not perfect for cards
                if aspect_diff > 0.4:  # Very different shape proportions
                    score = score * 0.8  # Penalize mismatched proportions

                # Use the correlation score with shape validation
                final_score = score

                detection = {
                    "label": template.name,
                    "confidence": float(final_score),
                    "box": {
                        "left": int(x_adj),
                        "top": int(y_adj),
                        "right": int(x_adj + w_adj),
                        "bottom": int(y_adj + h_adj)
                    },
                    "variant": template.variant
                }
                all_detections.append(detection)

    # Optional label filter (e.g., only detect 'yield-sign')
    if only:
        allow = {lbl.lower() for lbl in only}
    else:
        inferred = infer_allowed_labels(scene_path.stem)
        allow = {lbl.lower() for lbl in inferred} if inferred else None

    if allow:
        all_detections = [d for d in all_detections if d["label"].lower() in allow]

    # Sort by confidence (highest first)
    all_detections.sort(key=lambda d: d.get('confidence', 0), reverse=True)

    # If we have an allow list (inferred or provided), keep best per label to reduce clutter
    if allow and all_detections:
        best_per_label = {}
        for det in all_detections:
            lbl = det["label"].lower()
            prev = best_per_label.get(lbl)
            if prev is None or det["confidence"] > prev["confidence"]:
                best_per_label[lbl] = det
        all_detections = list(best_per_label.values())

    # Create annotated image with ALL detections
    annotated = annotate_detections(scene, all_detections)

    # Save annotated image
    scene_name = scene_path.stem
    output_filename = f"{scene_name}_detected.png"
    output_path = OUTPUT_DIR / output_filename
    cv2.imwrite(str(output_path), annotated)

    return {
        "scene": str(scene_path),
        "detections": all_detections,
        "threshold": threshold,
        "annotated_image": str(output_path),
        "num_detections": len(all_detections)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run template matching on a scene with all available templates"
    )
    parser.add_argument("--scene", required=True, help="Path to scene image")
    parser.add_argument("--threshold", type=float, default=0.7, help="Detection threshold (default: 0.7)")
    parser.add_argument("--only", nargs="+", help="Restrict detection to specific template label(s)")
    parser.add_argument("--json", action="store_true", help="Output JSON format")

    args = parser.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_detection(scene_path, threshold=args.threshold, only=args.only)

        if args.json:
            # Output pure JSON for API consumption
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print(f"\nTemplate Matching Results")
            print(f"=" * 70)
            print(f"Scene: {result['scene']}")
            print(f"Threshold: {result['threshold']}")
            print(f"Detections: {result['num_detections']}")
            print(f"\nDetected Objects:")
            for i, det in enumerate(result['detections'], 1):
                print(f"  {i}. {det['label']}: {det['confidence']:.1%} confidence")
                box = det['box']
                print(f"     Box: ({box['left']}, {box['top']}) -> ({box['right']}, {box['bottom']})")
            print(f"\nAnnotated image saved to: {result['annotated_image']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
