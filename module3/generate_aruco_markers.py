"""
Generate ArUco markers for printing
Run this to create printable ArUco markers for your assignment
"""
import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("module3/aruco_markers")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def generate_aruco_marker(marker_id, size_pixels=200, dictionary_name="DICT_5X5_100"):
    """Generate a single ArUco marker"""
    if not hasattr(cv2, 'aruco'):
        print("[ERROR] OpenCV ArUco module not available")
        print("Install opencv-contrib-python: pip install opencv-contrib-python")
        return None

    # Get dictionary
    dict_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    # Generate marker image
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, size_pixels)

    # Add white border for better detection
    border_size = 40
    bordered = cv2.copyMakeBorder(
        marker_image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return bordered

def main():
    print("="*60)
    print("ArUco Marker Generator for Module 3 Assignment")
    print("="*60)

    # Generate markers 0-9
    num_markers = 10
    marker_size = 200  # pixels

    print(f"\nGenerating {num_markers} ArUco markers...")
    print(f"Dictionary: DICT_5X5_100")
    print(f"Marker size: {marker_size}x{marker_size} pixels")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    for marker_id in range(num_markers):
        marker = generate_aruco_marker(marker_id, marker_size)

        if marker is not None:
            output_path = OUTPUT_DIR / f"aruco_marker_{marker_id}.png"
            cv2.imwrite(str(output_path), marker)
            print(f"  [OK] Generated marker {marker_id}: {output_path.name}")

    # Create a sheet with multiple markers
    print("\nCreating printable sheet with all markers...")

    markers_per_row = 5
    rows = 2
    sheet_markers = []

    for i in range(num_markers):
        marker = generate_aruco_marker(i, 180)
        if marker is not None:
            # Add label
            labeled = cv2.copyMakeBorder(marker, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=255)
            cv2.putText(labeled, f"ID: {i}", (10, labeled.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
            sheet_markers.append(labeled)

    # Arrange in grid
    if sheet_markers:
        rows_of_markers = []
        for row_idx in range(rows):
            start = row_idx * markers_per_row
            end = start + markers_per_row
            row_markers = sheet_markers[start:end]
            if row_markers:
                row = np.hstack(row_markers)
                rows_of_markers.append(row)

        if rows_of_markers:
            sheet = np.vstack(rows_of_markers)
            sheet_path = OUTPUT_DIR / "aruco_markers_sheet.png"
            cv2.imwrite(str(sheet_path), sheet)
            print(f"  [OK] Created printable sheet: {sheet_path.name}")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Open the 'aruco_markers_sheet.png' file")
    print("2. Print it on regular paper")
    print("3. Cut out individual markers")
    print("4. Find a NON-RECTANGULAR object (bottle, book, mug, etc.)")
    print("5. Stick markers around the object's boundary")
    print("6. Take 10+ photos from different angles/distances")
    print("7. Upload through the web app Assignment 3 > Section 4")
    print("="*60)

if __name__ == "__main__":
    main()
