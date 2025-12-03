"""
Batch process all ArUco marker images for object segmentation.
Default input: module3/uploads/ArUco
Usage: python module3/process_aruco_batch.py [--input-dir module3/uploads/ArUco2]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipelines import aruco_segment

def process_aruco_batch(input_dir: Path):
    """Process all ArUco images and generate segmentation results"""

    # Setup paths
    aruco_dir = input_dir
    output_dir = Path("module3/output/aruco_batch")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("ArUco Object Segmentation - Batch Processing")
    print("="*70)
    print(f"\nInput directory: {aruco_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")

    # Find all images
    image_files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.update(aruco_dir.glob(ext))
    image_files = sorted(image_files)

    if not image_files:
        print(f"\n[ERROR] No images found in {aruco_dir}")
        return

    print(f"\nFound {len(image_files)} images to process\n")

    # Process each image
    results = []
    successful = 0
    failed = 0

    for idx, img_path in enumerate(sorted(image_files), 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        print("-" * 70)

        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [ERROR] Could not load image")
                failed += 1
                continue

            print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

            # Run ArUco segmentation
            result = aruco_segment(image)

            # Save individual results with descriptive names
            stem = f"image_{idx:02d}_{img_path.stem}"

            # Save overlay
            overlay_img = cv2.imread(str(Path("module3") / result['overlay_path']))
            overlay_path = output_dir / f"{stem}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay_img)

            # Save mask
            mask_img = cv2.imread(str(Path("module3") / result['mask_path']), cv2.IMREAD_GRAYSCALE)
            mask_path = output_dir / f"{stem}_mask.png"
            cv2.imwrite(str(mask_path), mask_img)

            # Save edges
            edges_img = cv2.imread(str(Path("module3") / result['edges_path']), cv2.IMREAD_GRAYSCALE)
            edges_path = output_dir / f"{stem}_edges.png"
            cv2.imwrite(str(edges_path), edges_img)

            # Store result info
            result_info = {
                'image_name': img_path.name,
                'image_index': idx,
                'markers_detected': result['marker_count'],
                'overlay_path': str(overlay_path.absolute()),
                'mask_path': str(mask_path.absolute()),
                'edges_path': str(edges_path.absolute()),
                'status': 'success'
            }
            results.append(result_info)

            print(f"  [SUCCESS]")
            print(f"    - Markers detected: {result['marker_count']}")
            print(f"    - Overlay saved: {overlay_path.name}")
            print(f"    - Mask saved: {mask_path.name}")
            print(f"    - Edges saved: {edges_path.name}")
            successful += 1

        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            result_info = {
                'image_name': img_path.name,
                'image_index': idx,
                'status': 'failed',
                'error': str(e)
            }
            results.append(result_info)
            failed += 1

        print()

    # Save summary report
    summary = {
        'total_images': len(image_files),
        'successful': successful,
        'failed': failed,
        'results': results
    }

    report_path = output_dir / "processing_report.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"Report saved to: {report_path.absolute()}")
    print("="*70)

    # Create summary visualization
    if successful > 0:
        print("\nCreating summary visualization...")
        create_summary_grid(output_dir, results)

def create_summary_grid(output_dir: Path, results: list):
    """Create a grid visualization showing all results"""

    # Filter successful results
    successful_results = [r for r in results if r['status'] == 'success']

    if not successful_results:
        return

    # Load first image to get size
    first_overlay = cv2.imread(successful_results[0]['overlay_path'])
    if first_overlay is None:
        return

    # Resize for grid (max 500 pixels wide)
    target_width = 500
    h, w = first_overlay.shape[:2]
    scale = target_width / w
    target_height = int(h * scale)

    # Create grid (2 columns)
    cols = 2
    rows = (len(successful_results) + cols - 1) // cols

    grid_images = []
    for row in range(rows):
        row_images = []
        for col in range(cols):
            idx = row * cols + col
            if idx < len(successful_results):
                img_path = successful_results[idx]['overlay_path']
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, (target_width, target_height))

                    # Add label
                    label = f"Image {successful_results[idx]['image_index']}"
                    cv2.putText(img_resized, label, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    row_images.append(img_resized)
            else:
                # Fill empty space
                empty = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
                row_images.append(empty)

        if row_images:
            row_concat = np.hstack(row_images)
            grid_images.append(row_concat)

    if grid_images:
        grid = np.vstack(grid_images)
        grid_path = output_dir / "summary_grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"[SUCCESS] Summary grid saved: {grid_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ArUco segmentation for Module 3")
    parser.add_argument("--input-dir", type=Path, default=Path("module3/uploads/ArUco"), help="Directory containing input images")
    args = parser.parse_args()
    process_aruco_batch(args.input_dir)
