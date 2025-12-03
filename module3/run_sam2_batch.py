"""
Run SAM2 automatic segmentation on a batch of images and compare to our ArUco masks.

Usage:
  python -m module3.run_sam2_batch \\
    --input-dir module3/uploads/ArUco2 \\
    --aruco-dir module3/output/aruco_batch \\
    --checkpoint module3/sam2_checkpoints/sam2_hiera_small.pt \\
    --config sam2_hiera_s.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

try:  # Support both `python -m module3.run_sam2_batch` and direct execution
    from module3.pipelines import compare_masks
except ImportError:  # pragma: no cover - fallback for script execution
    from pipelines import compare_masks


def find_aruco_mask(aruco_dir: Path, stem: str) -> Optional[Path]:
    """Find the saved ArUco mask that corresponds to the given image stem."""
    candidates = sorted(aruco_dir.glob(f"*{stem}_mask.png"))
    return candidates[0] if candidates else None


def run_sam2_batch(
    input_dir: Path,
    aruco_dir: Path,
    checkpoint: Path,
    config: str,
    device: str = "cpu",
    points_per_side: int = 8,
    max_images: int = 0,
) -> None:
    output_dir = Path("module3/output/sam2")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(image_files)

    if max_images and max_images > 0:
        image_files = image_files[:max_images]

    if not image_files:
        print(f"[ERROR] No images found in {input_dir}")
        return

    print("=" * 70)
    print("SAM2 Batch Segmentation + Comparison")
    print("=" * 70)
    print(f"Input dir     : {input_dir.resolve()}")
    print(f"ArUco masks   : {aruco_dir.resolve()}")
    print(f"Output dir    : {output_dir.resolve()}")
    print(f"Checkpoint    : {checkpoint.resolve()}")
    print(f"Config        : {config}")
    print(f"Device        : {device}")
    print(f"Images found  : {len(image_files)}")
    print("=" * 70)

    # Build SAM2 model + automatic mask generator (single mask is enough here)
    sam2_model = build_sam2(config, ckpt_path=str(checkpoint), device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.90,
        crop_n_layers=0,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )

    summary = []

    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print("  [ERROR] failed to read image")
            continue

        # Generate masks and pick the largest high-confidence one
        masks = mask_generator.generate(image)
        if not masks:
            print("  [ERROR] SAM2 produced no masks")
            continue
        masks_sorted = sorted(
            masks,
            key=lambda m: (m.get("stability_score", 0.0), m.get("area", 0)),
            reverse=True,
        )
        chosen = masks_sorted[0]
        sam_mask = (chosen["segmentation"].astype(np.uint8)) * 255

        # Save SAM2 outputs
        stem = img_path.stem
        sam_mask_path = output_dir / f"{stem}_sam2_mask.png"
        sam_overlay_path = output_dir / f"{stem}_sam2_overlay.png"

        cv2.imwrite(str(sam_mask_path), sam_mask)
        overlay = image.copy()
        contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(str(sam_overlay_path), overlay)

        record = {
            "image": img_path.name,
            "sam_mask": str(sam_mask_path.resolve()),
            "sam_overlay": str(sam_overlay_path.resolve()),
            "sam_area": int(chosen.get("area", int((sam_mask > 0).sum()))),
        }

        # Compare with our ArUco-based mask if available
        aruco_mask_path = find_aruco_mask(aruco_dir, stem)
        if aruco_mask_path and aruco_mask_path.exists():
            aruco_mask = cv2.imread(str(aruco_mask_path), cv2.IMREAD_GRAYSCALE)
            if aruco_mask is not None:
                scores = compare_masks(aruco_mask, sam_mask)
                record.update(scores)
                record["aruco_mask"] = str(aruco_mask_path.resolve())
                print(f"  Dice: {scores['dice']:.4f}  IoU: {scores['iou']:.4f}")
            else:
                print("  [WARN] Could not read ArUco mask for comparison.")
        else:
            print("  [WARN] No matching ArUco mask found; skipping comparison.")

        summary.append(record)

    # Save summary JSON
    report_path = output_dir / "sam2_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Print aggregate metrics
    dice_vals = [r["dice"] for r in summary if "dice" in r]
    iou_vals = [r["iou"] for r in summary if "iou" in r]
    if dice_vals and iou_vals:
        print("\n=== Aggregate Metrics ===")
        print(f"Mean Dice: {np.mean(dice_vals):.4f}")
        print(f"Mean IoU : {np.mean(iou_vals):.4f}")
        print(f"Report   : {report_path.resolve()}")
    else:
        print("\nNo comparisons were computed (missing ArUco masks?).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM2 on a batch of images and compare to ArUco masks.")
    parser.add_argument("--input-dir", type=Path, default=Path("module3/uploads/ArUco2"), help="Input image directory")
    parser.add_argument("--aruco-dir", type=Path, default=Path("module3/output/aruco_batch"), help="Directory with ArUco masks")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("module3/sam2_checkpoints/sam2_hiera_small.pt"),
        help="Path to SAM2 checkpoint",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2/sam2_hiera_s.yaml",
        help="SAM2 config name (e.g., configs/sam2/sam2_hiera_s.yaml)",
    )
    parser.add_argument("--device", default="cpu", help="Device string, e.g., cpu or cuda:0")
    parser.add_argument("--points-per-side", type=int, default=8, help="Sampling density for SAM2 (lower = faster)")
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit on number of images to process (0 = all)")
    args = parser.parse_args()

    run_sam2_batch(
        args.input_dir,
        args.aruco_dir,
        args.checkpoint,
        args.config,
        device=args.device,
        points_per_side=args.points_per_side,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
