"""
Test all template/scene pairs to evaluate template matching performance
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# Add module2 to path
sys.path.insert(0, str(Path(__file__).parent / 'module2'))

from detector import detect_scene, annotate_detections

# Define test pairs
test_pairs = [
    ("cheerios.png", "cheerios-scene.png"),
    ("cheerios-logo.png", "cheerios-scene.png"),
    ("coke.png", "coke-scene.png"),
    ("Coca-Cola-logo.png", "coke-scene.png"),
    ("iphone.png", "iphone-scene.png"),
    ("apple-logo.png", "iphone-scene.png"),
    ("razor.png", "razor-scene.png"),
    ("razor-logo.png", "razor-scene.png"),
    ("stanley.png", "stanely-scene.png"),
    ("stanley-logo.png", "stanely-scene.png"),
    ("heart.png", "ten-of-hearts.png"),
    ("gsu logo.png", "gsu.jpg"),
    ("gsu logo.png", "gsu2.jpg"),
    ("UNO_Logo.png", "uno-box.png"),
    ("coke.png", "coke-bottle.jpg"),
]

BASE_DIR = Path(__file__).parent / "module2"
TEMPLATE_DIR = BASE_DIR / "data" / "templates"
SCENE_DIR = BASE_DIR / "data" / "scenes"
OUTPUT_DIR = BASE_DIR / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

def test_single_pair(template_name, scene_name, threshold=0.4):
    """Test a single template/scene pair"""
    template_path = TEMPLATE_DIR / template_name
    scene_path = SCENE_DIR / scene_name

    # Check if files exist
    if not template_path.exists():
        return {
            "template": template_name,
            "scene": scene_name,
            "status": "SKIP",
            "reason": f"Template not found: {template_path}"
        }

    if not scene_path.exists():
        return {
            "template": template_name,
            "scene": scene_name,
            "status": "SKIP",
            "reason": f"Scene not found: {scene_path}"
        }

    try:
        # Run detection
        result = detect_scene(scene_path, threshold=threshold)

        detections = result.get("detections", [])

        # Read scene for annotation
        scene = cv2.imread(str(scene_path))

        if detections:
            # Annotate detections
            annotated = annotate_detections(scene, detections)

            # Save annotated image
            output_filename = f"{template_path.stem}_on_{scene_path.stem}.png"
            output_path = OUTPUT_DIR / output_filename
            cv2.imwrite(str(output_path), annotated)

            # Get best detection
            best = max(detections, key=lambda d: d.get("confidence", 0))

            return {
                "template": template_name,
                "scene": scene_name,
                "status": "DETECTED",
                "confidence": best.get("confidence", 0),
                "label": best.get("label", "unknown"),
                "num_detections": len(detections),
                "output": str(output_path)
            }
        else:
            return {
                "template": template_name,
                "scene": scene_name,
                "status": "NOT DETECTED",
                "confidence": 0,
                "threshold": threshold
            }

    except Exception as e:
        return {
            "template": template_name,
            "scene": scene_name,
            "status": "ERROR",
            "reason": str(e)
        }

def main():
    print("=" * 80)
    print("TEMPLATE MATCHING TEST REPORT")
    print("=" * 80)
    print()

    results = []

    for template_name, scene_name in test_pairs:
        result = test_single_pair(template_name, scene_name)
        results.append(result)

        # Print result
        status = result["status"]
        if status == "DETECTED":
            print(f"[OK] {template_name:30s} -> {scene_name:30s}")
            print(f"  Confidence: {result['confidence']:.3f} | Label: {result['label']}")
            print(f"  Output: {result['output']}")
        elif status == "NOT DETECTED":
            print(f"[FAIL] {template_name:30s} -> {scene_name:30s}")
            print(f"  No detection (threshold: {result['threshold']})")
        elif status == "SKIP":
            print(f"[SKIP] {template_name:30s} -> {scene_name:30s}")
            print(f"  {result['reason']}")
        elif status == "ERROR":
            print(f"[ERROR] {template_name:30s} -> {scene_name:30s}")
            print(f"  Error: {result['reason']}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    detected = sum(1 for r in results if r["status"] == "DETECTED")
    not_detected = sum(1 for r in results if r["status"] == "NOT DETECTED")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"Total pairs tested: {len(results)}")
    print(f"  Detected:     {detected}")
    print(f"  Not detected: {not_detected}")
    print(f"  Skipped:      {skipped}")
    print(f"  Errors:       {errors}")
    print()

    if detected > 0:
        print(f"Success rate: {detected}/{len(results) - skipped} = {detected/(len(results)-skipped)*100:.1f}%")

    print()
    print(f"Annotated images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
