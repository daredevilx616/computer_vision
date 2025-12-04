"""
Generate SAM2 segmentation mask for a single image - for use with real-time tracker.

This creates a mask that you can upload to the SAM2 tracker in module 5-6.

Usage:
  1. Capture a reference frame from the webcam UI
  2. Run this script on that frame:
     python -m module56.run_sam2_single --input path/to/frame.jpg --output mask.png
  3. Upload mask.png to the SAM2 tracker UI
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("[WARNING] SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")


def run_sam2_single(
    input_path: Path,
    output_path: Path,
    checkpoint: Path,
    config: str = "configs/sam2/sam2_hiera_s.yaml",
    device: str = "cpu",
    points_per_side: int = 16,
) -> None:
    """Run SAM2 on a single image and save the largest mask."""

    if not SAM2_AVAILABLE:
        print("[ERROR] SAM2 is not installed. Cannot proceed.")
        return

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return

    if not checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        print(f"Download from: https://github.com/facebookresearch/segment-anything-2")
        return

    print("=" * 70)
    print("SAM2 Single Image Segmentation")
    print("=" * 70)
    print(f"Input         : {input_path.resolve()}")
    print(f"Output        : {output_path.resolve()}")
    print(f"Checkpoint    : {checkpoint.resolve()}")
    print(f"Config        : {config}")
    print(f"Device        : {device}")
    print("=" * 70)

    # Load image
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"[ERROR] Failed to read image: {input_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Build SAM2 model
    print("Loading SAM2 model...")
    sam2_model = build_sam2(config, ckpt_path=str(checkpoint), device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.88,
        crop_n_layers=0,
        min_mask_region_area=100,  # Filter tiny masks
        output_mode="binary_mask",
    )

    # Generate masks
    print("Generating masks...")
    masks = mask_generator.generate(image)

    if not masks:
        print("[ERROR] SAM2 produced no masks. Try different parameters or a clearer image.")
        return

    print(f"Generated {len(masks)} masks")

    # Sort by stability score and area, pick the best one
    masks_sorted = sorted(
        masks,
        key=lambda m: (m.get("stability_score", 0.0), m.get("area", 0)),
        reverse=True,
    )

    # Show top 3 candidates
    print("\nTop mask candidates:")
    for i, m in enumerate(masks_sorted[:3], 1):
        area = m.get("area", 0)
        score = m.get("stability_score", 0.0)
        print(f"  {i}. Area: {area:6d} pixels, Stability: {score:.3f}")

    # Use the best mask
    chosen = masks_sorted[0]
    sam_mask = (chosen["segmentation"].astype(np.uint8)) * 255

    # Save mask
    cv2.imwrite(str(output_path), sam_mask)
    print(f"\n✓ Mask saved to: {output_path.resolve()}")

    # Save overlay for visualization
    overlay_path = output_path.with_stem(output_path.stem + "_overlay")
    overlay = image.copy()

    # Draw mask overlay
    colored_mask = np.zeros_like(image)
    colored_mask[sam_mask > 0] = [0, 255, 255]  # Cyan
    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

    # Draw contour
    contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(str(overlay_path), overlay)
    print(f"✓ Overlay saved to: {overlay_path.resolve()}")

    print("\n" + "=" * 70)
    print("Done! Upload the mask to the SAM2 tracker UI.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM2 on a single image for real-time tracker initialization"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image (captured reference frame)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sam2_mask.png"),
        help="Output mask file (default: sam2_mask.png)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("module3/sam2_checkpoints/sam2_hiera_small.pt"),
        help="Path to SAM2 checkpoint"
    )
    parser.add_argument(
        "--config",
        default="configs/sam2/sam2_hiera_s.yaml",
        help="SAM2 config (default: configs/sam2/sam2_hiera_s.yaml)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu or cuda (default: cpu)"
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=16,
        help="SAM2 sampling density (higher = slower but more accurate)"
    )

    args = parser.parse_args()

    run_sam2_single(
        args.input,
        args.output,
        args.checkpoint,
        args.config,
        args.device,
        args.points_per_side,
    )


if __name__ == "__main__":
    main()
