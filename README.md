# Computer Vision Modules - Complete Guide

A comprehensive collection of 7 computer vision modules demonstrating fundamental and advanced CV techniques, from template matching to 3D stereo reconstruction.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Module 1: Perspective Projection & Measurement](#module-1-perspective-projection--measurement)
- [Module 2: Template Matching & Fourier Deblurring](#module-2-template-matching--fourier-deblurring)
- [Module 3: Image Processing & Segmentation](#module-3-image-processing--segmentation)
- [Module 4: SIFT & Image Stitching](#module-4-sift--image-stitching)
- [Module 5: ArUco Marker Tracking](#module-5-aruco-marker-tracking)
- [Module 6: Markerless Object Tracking](#module-6-markerless-object-tracking)
- [Module 7: Stereo Vision & 3D Measurement](#module-7-stereo-vision--3d-measurement)
- [Web Application](#web-application)
- [API Reference](#api-reference)

---

## Overview

This project contains **7 distinct computer vision modules**, each implementing different CV algorithms and techniques:

| Module | Focus | Techniques |
|--------|-------|------------|
| **Module 1** | Perspective Projection | Camera calibration, pinhole camera model, real-world measurement |
| **Module 2a** | Template Matching | Multi-scale matching, ORB features, histogram correlation |
| **Module 2b** | Fourier Deblur | Frequency domain filtering, Wiener deconvolution |
| **Module 3** | Image Processing | Gradients, edge detection, keypoints, ArUco segmentation |
| **Module 4** | Feature Matching | SIFT implementation, RANSAC, panorama stitching |
| **Module 5** | ArUco Tracking | Marker detection, QR code detection, real-time tracking |
| **Module 6** | Markerless Tracking | Color-based tracking, SAM2 AI segmentation |
| **Module 7** | Stereo Vision | Disparity mapping, 3D reconstruction, depth measurement |

**Tech Stack:**
- **Backend**: Python 3.x, Flask, OpenCV, NumPy
- **Frontend**: Next.js 15, TypeScript, Tailwind CSS
- **AI/ML**: SAM2 (Segment Anything Model 2) for intelligent segmentation
- **Deployment**: Docker support, Vercel-ready

---

## Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Node.js 18 or higher
node --version
```

### Setup Instructions

**1. Clone the repository:**
```bash
git clone <repository-url>
cd computer_vision
```

**2. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-contrib-python>=4.8.0
- numpy>=1.24.0
- flask>=2.3.0
- Pillow>=10.0.0

**3. Install Node.js dependencies:**
```bash
npm install
```

**4. (Optional) Install SAM2 for Module 6:**
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

Download SAM2 checkpoint:
```bash
# Create checkpoint directory
mkdir -p module3/sam2_checkpoints

# Download checkpoint (example - adjust URL)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt \
  -O module3/sam2_checkpoints/sam2_hiera_small.pt
```

---

## Module 1: Perspective Projection & Measurement

**Location:** [`measure.py`](measure.py), [`app/measure/`](app/measure/), Flask endpoints in [`app.py`](app.py)

### Purpose
Measure real-world dimensions of objects using perspective projection equations and camera calibration. Implements the pinhole camera model for converting pixel measurements to physical units.

### Theory: Pinhole Camera Model

The fundamental relationship in perspective projection:

```
real_size / distance = pixel_size / focal_length_pixels
```

Rearranging for real-world size:

```
real_size = (pixel_size × distance) / focal_length_pixels
```

**Calibration Formula:**
```
focal_length_pixels = (pixel_width × distance) / real_width
```

Where:
- `pixel_width`: Measured width in pixels of a known object
- `real_width`: Actual physical width of the reference object
- `distance`: Distance from camera to object (same units as real_width)
- `focal_length_pixels`: Computed focal length in pixels

**Measurement Formula:**
```
real_size = (pixel_width × distance) / focal_length_pixels
```

### Algorithm

**Two-Phase Process:**

#### Phase 1: Calibration
1. Place a reference object of known dimensions at a known distance
2. Capture image and measure object width in pixels
3. Compute focal length in pixels using calibration formula
4. Store calibration parameters

#### Phase 2: Measurement
1. Place target object at a known distance (can be different from calibration)
2. Capture image and measure object width in pixels
3. Apply measurement formula using stored focal length
4. Return real-world size in calibrated units

### Implementation

**Utility Functions:** ([measure.py](measure.py))

```python
def compute_focal_pixels(known_real_width, known_distance, measured_pixel_width):
    """
    Compute focal length in pixels:
    f = (pixel_width * distance) / real_width
    """
    return (measured_pixel_width * known_distance) / known_real_width

def estimate_real_size(measured_pixel_width, distance, focal_pixels):
    """
    Estimate real-world size from pixel width:
    real_size = (pixel_width * distance) / focal_pixels
    """
    return (measured_pixel_width * distance) / focal_pixels
```

**Flask Backend:** ([app.py:78-110](app.py#L78-L110))

```python
@app.route("/calibrate", methods=["POST"])
def calibrate():
    """Calibrate using reference object"""
    data = request.get_json()
    pixel_width = float(data.get("pixel_width"))
    real_width = float(data.get("real_width"))
    distance = float(data.get("distance"))
    units = data.get("units", "cm")

    focal_length = (pixel_width * distance) / real_width
    CALIBRATION.update({
        "focal_length_pixels": focal_length,
        "units": units
    })

    return jsonify({
        "status": "success",
        "focal_length_pixels": focal_length,
        "units": units
    })

@app.route("/measure", methods=["POST"])
def measure():
    """Measure real-world size of object"""
    data = request.get_json()

    if CALIBRATION["focal_length_pixels"] is None:
        return jsonify({"status": "error", "message": "Calibrate first"}), 400

    pixel_width = float(data.get("pixel_width"))
    distance = float(data.get("distance"))

    real_size = (pixel_width * distance) / CALIBRATION["focal_length_pixels"]

    return jsonify({
        "pixel_width": pixel_width,
        "distance": distance,
        "real_size": real_size,
        "units": CALIBRATION["units"]
    })
```

### Web Application

**Interactive Measurement Tool:** [`app/measure/page.tsx`](app/measure/page.tsx)

**Features:**
- Real-time webcam access
- Click to mark two points on object
- Automatic pixel distance calculation
- Camera parameter input (focal length, sensor width, distance)
- Instant real-world measurement display

**Workflow:**
1. **Start Camera** - Access webcam via MediaDevices API
2. **Capture Photo** - Freeze frame for measurement
3. **Mark Points** - Click two points on object to measure
4. **Enter Parameters:**
   - Distance to object (Z in meters)
   - Focal length (mm)
   - Sensor width (mm)
5. **Calculate** - Get real-world dimension in mm

**Calculation in Browser:**
```typescript
const imageWidthPx = photo.naturalWidth;
const focalPx = (parseFloat(focalMM) / parseFloat(sensorWidthMM)) * imageWidthPx;
const Z = parseFloat(Zmeters);
const realMM = (pixelDist * Z * 1000) / focalPx;
```

### Usage Example

**Scenario:** Measure the width of a book

**Step 1: Calibration**
```bash
# Use a credit card as reference (85.60mm × 53.98mm)
curl -X POST http://localhost:5000/calibrate \
  -H "Content-Type: application/json" \
  -d '{
    "pixel_width": 428,
    "real_width": 85.6,
    "distance": 50,
    "units": "mm"
  }'

# Response:
{
  "status": "success",
  "focal_length_pixels": 250.0,
  "units": "mm"
}
```

**Step 2: Measurement**
```bash
# Measure book at same distance
curl -X POST http://localhost:5000/measure \
  -H "Content-Type: application/json" \
  -d '{
    "pixel_width": 600,
    "distance": 50
  }'

# Response:
{
  "pixel_width": 600,
  "distance": 50,
  "real_size": 120.0,
  "units": "mm"
}
```

**Python API:**
```python
from measure import compute_focal_pixels, estimate_real_size

# Calibration with credit card
focal_px = compute_focal_pixels(
    known_real_width=85.6,    # mm
    known_distance=500,        # mm
    measured_pixel_width=428   # px
)
print(f"Focal length: {focal_px:.2f} pixels")

# Measurement
book_width = estimate_real_size(
    measured_pixel_width=600,  # px
    distance=500,              # mm
    focal_pixels=focal_px
)
print(f"Book width: {book_width:.2f} mm")
```

### Typical Camera Parameters

**Common Values:**

| Device | Focal Length | Sensor Width | Typical f_pixels (at 1280px) |
|--------|--------------|--------------|-------------------------------|
| Smartphone | 4.5 mm | 5.6 mm | ~1030 px |
| Webcam | 3.6 mm | 4.8 mm | ~960 px |
| Laptop Camera | 2.8 mm | 3.2 mm | ~1120 px |
| DSLR (50mm lens) | 50 mm | 36 mm (full frame) | ~1778 px |

**Finding Your Camera Parameters:**
1. Check device specifications online
2. Use calibration with known object
3. Use camera metadata (EXIF data for focal length)
4. Measure sensor physically (advanced)

### Accuracy Considerations

**Error Sources:**
1. **Distance measurement error**: ±1cm can cause ±2% error at 50cm
2. **Pixel measurement error**: ±1px significant for small objects
3. **Lens distortion**: Barrel/pincushion distortion near edges
4. **Sensor size uncertainty**: Varies by device
5. **Perspective error**: Object must be perpendicular to camera

**Improving Accuracy:**
1. Use precise distance measurement (laser distance meter)
2. Zoom in / use higher resolution
3. Calibrate at same distance as measurement
4. Measure in center of frame (less distortion)
5. Use undistortion if camera calibration matrix available
6. Take multiple measurements and average

**Expected Accuracy:**
- Careful setup: ±2-5% error
- Casual measurement: ±10-15% error
- With lens calibration: <1% error possible

### Limitations

1. **Assumes pinhole camera model** - Real lenses have distortion
2. **Requires known distance** - Cannot measure without Z
3. **2D measurement only** - Cannot measure depth directly
4. **Calibration drift** - Zoom/focus changes invalidate calibration
5. **Perspective effects** - Object must be planar and perpendicular

**When to Use Module 7 Instead:**
- Module 7 (Stereo Vision) can measure 3D distances without knowing Z
- Better for depth measurement and 3D object dimensions
- Module 1 is simpler but requires manual distance input

---

## Module 2: Template Matching & Fourier Deblurring

**Location:** [`module2/`](module2/)

### Part A: Template Matching

#### Purpose
Detect and localize objects in scenes using template matching with multi-scale search and feature verification.

#### Key Features
- **Multi-scale matching**: Tests 15 different scale factors (0.12x to 2.7x)
- **Edge-based refinement**: Auto-crops templates to focus on content
- **Histogram correlation**: HSV color verification
- **ORB feature extraction**: Additional verification with keypoints
- **Smart thresholding**: Per-template configurable confidence thresholds

### Architecture

```python
Template Loading → Multi-scale Search → Feature Verification → Best Match Selection
```

**Core Components:**
1. **Template Preprocessing** ([detector.py:64-113](module2/detector.py#L64-L113))
   - Load template image
   - Convert to grayscale and extract edges
   - Auto-crop to content using Canny edge detection
   - Extract ORB keypoints and descriptors
   - Compute color histograms (HSV)

2. **Multi-scale Matching** ([detector.py:165-196](module2/detector.py#L165-L196))
   - Resize template to multiple scales
   - Run normalized cross-correlation (TM_CCOEFF_NORMED)
   - Track best score, location, and scale

3. **Verification** ([detector.py:270-284](module2/detector.py#L270-L284))
   - Histogram comparison (HSV color matching)
   - Optional color tolerance checks
   - Reject extremely small/large matches

### Usage

**Command Line:**
```bash
# Detect objects in a scene
python module2/run_template_matching.py \
  --scene path/to/scene.jpg \
  --threshold 0.7 \
  --json

# Detect specific templates only
python module2/run_template_matching.py \
  --scene path/to/scene.jpg \
  --only stop-sign yield-sign
```

**Python API:**
```python
from module2.detector import detect_scene, load_templates
from pathlib import Path

# Detect with default templates
result = detect_scene(
    scene_path=Path("scene.jpg"),
    threshold=0.7
)

# Access detections
for detection in result["detections"]:
    print(f"Found {detection['label']} at {detection['box']}")
    print(f"Confidence: {detection['confidence']:.2f}")
```

**Templates:**
Place template images in `module2/data/templates/` as PNG files. Filename (without extension) becomes the label.

Example:
```
module2/data/templates/
├── coke.png          # Label: "coke"
├── stop-sign.png     # Label: "stop-sign"
└── cheerios-logo.png # Label: "cheerios-logo"
```

### Output Format
```json
{
  "scene": "path/to/scene.jpg",
  "detections": [
    {
      "label": "stop-sign",
      "confidence": 0.87,
      "box": {
        "left": 120,
        "top": 80,
        "right": 280,
        "bottom": 240
      },
      "variant": "stop-sign.png"
    }
  ],
  "threshold": 0.7,
  "annotated_image": "module2/output/scene_detected.png"
}
```

---

### Part B: Fourier Transform & Deblurring

**Script:** [`module2/fourier_deblur.py`](module2/fourier_deblur.py)

#### Purpose
Demonstrate frequency domain image processing with Gaussian blur simulation and Wiener deconvolution restoration.

#### Algorithm

**Forward Process (Blurring):**
1. Convert image to float [0,1]
2. Apply Gaussian blur (13×13 kernel, σ=2.4)
3. Result: Blurred image

**Inverse Process (Restoration):**
1. Pad image to optimal DFT size (reduces ringing)
2. Compute FFT of blurred image and PSF
3. Apply Wiener filter: `F̂ = (H* / (|H|² + K)) × G`
   - H = PSF frequency response
   - G = Blurred image FFT
   - K = Wiener constant (10⁻³)
4. Inverse FFT and clip to [0,1]

#### Mathematical Foundation

**Blur Model:**
```
g(x,y) = f(x,y) ⊗ h(x,y) + n(x,y)
```
Where:
- `f` = original image
- `h` = point spread function (PSF)
- `⊗` = convolution
- `n` = noise

**Wiener Filter (Frequency Domain):**
```
F̂(u,v) = [H*(u,v) / (|H(u,v)|² + K)] × G(u,v)
```

#### Usage

**Command Line:**
```bash
python -m module2.fourier_deblur \
  --input path/to/image.jpg \
  --output-dir module2/output \
  --json
```

**Output:**
- `{stem}_gaussian_blur.png` - Blurred version
- `{stem}_fourier_restore.png` - Restored version
- `{stem}_fourier_montage.png` - Side-by-side comparison (Original | Blurred | Restored)

**Python API:**
```python
from module2.fourier_deblur import process_image
from pathlib import Path

result = process_image(
    input_path=Path("image.jpg"),
    output_dir=Path("module2/output")
)

print(f"PSNR (blur): {result['psnr_blur']:.2f} dB")
print(f"PSNR (restore): {result['psnr_restore']:.2f} dB")
```

#### Performance Metrics

The module computes **Peak Signal-to-Noise Ratio (PSNR)**:
```
PSNR = 10 × log₁₀(MAX² / MSE)
```

**Typical Results:**
- Blur PSNR: 20-25 dB (degraded)
- Restore PSNR: 30-35 dB (improved, but not perfect)

---

## Module 3: Image Processing & Segmentation

**Location:** [`module3/`](module3/)

### Purpose
Comprehensive image processing toolkit covering gradients, edge detection, keypoint extraction, and AI-powered segmentation with ArUco markers.

### Features

#### 1. Gradient Computation
**Function:** `compute_gradients()` ([pipelines.py:41-72](module3/pipelines.py#L41-L72))

Computes:
- **Sobel gradients** (∂I/∂x, ∂I/∂y)
- **Magnitude**: `√(gₓ² + gᵧ²)`
- **Orientation**: Color-coded direction map (HSV)
- **Laplacian of Gaussian (LoG)**: Second derivative for blob detection

```python
from module3.pipelines import compute_gradients
import cv2

image = cv2.imread("image.jpg")
result = compute_gradients(image)

# result contains:
# - "magnitude": gradient strength visualization
# - "orientation": gradient direction (color-coded)
# - "log": Laplacian of Gaussian response
```

#### 2. Keypoint Detection
**Function:** `detect_keypoints()` ([pipelines.py:75-178](module3/pipelines.py#L75-L178))

**Two Modes:**

**Edge Mode:**
- Automatic Canny edge detection (adaptive thresholds)
- Finds dominant contours
- Samples keypoints along contour perimeters
- Best for: Objects with clear boundaries

**Corner Mode:**
- Constrains detection to foreground object
- Uses Shi-Tomasi corner detection
- Polygon approximation for geometric shapes
- Best for: Structured objects with corners

```python
from module3.pipelines import detect_keypoints
import cv2

image = cv2.imread("object.jpg")

# Edge-based keypoints
edge_result = detect_keypoints(image, mode="edge")

# Corner-based keypoints
corner_result = detect_keypoints(image, mode="corner")
```

#### 3. Boundary Segmentation
**Function:** `segment_boundary()` ([pipelines.py:181-203](module3/pipelines.py#L181-L203))

Simple contour-based segmentation:
1. Gaussian blur → Canny edges
2. Find external contours
3. Select largest contour as object
4. Generate binary mask

#### 4. ArUco-Based Segmentation
**Function:** `aruco_segment()` ([pipelines.py:206-423](module3/pipelines.py#L206-L423))

**Advanced object segmentation using ArUco markers as spatial anchors.**

**Algorithm:**
1. **Marker Detection**
   - Detect ArUco markers with tuned parameters
   - Create region of interest (ROI) around markers
   - Marker serves as "this is the object area" hint

2. **Edge Processing**
   - CLAHE contrast enhancement
   - Adaptive Canny thresholds based on median intensity
   - Morphological closing to connect edges

3. **Contour Scoring** (Multi-criteria)
   ```python
   score = 0.45 × distance_to_marker
         + 0.20 × solidity
         + 0.20 × compactness
         + 0.15 × relative_area
   ```
   - **Distance**: Proximity to markers
   - **Solidity**: `area / convex_hull_area`
   - **Compactness**: `4πA / P²` (circularity)

4. **Fallback Strategy**
   - If edge-based fails → Otsu threshold in bottom region
   - If still fails → GrabCut with marker seeds
   - Final fallback → Marker convex hull

**Usage:**
```python
from module3.pipelines import aruco_segment
import cv2

image = cv2.imread("bottle_with_marker.jpg")

result = aruco_segment(
    image,
    dictionary_name="DICT_5X5_100"
)

# result contains:
# - "overlay": Visualization with markers + contour
# - "mask": Binary segmentation mask
# - "edges": Edge detection visualization
# - "marker_count": Number of detected markers
```

**Supported Dictionaries:**
- `DICT_4X4_50`, `DICT_4X4_100`
- `DICT_5X5_50`, `DICT_5X5_100`
- `DICT_6X6_50`, `DICT_6X6_100`
- `DICT_7X7_50`, `DICT_7X7_100`

#### 5. SAM2 Batch Segmentation
**Script:** [`run_sam2_batch.py`](module3/run_sam2_batch.py)

Uses Meta's Segment Anything Model 2 for AI-powered segmentation.

```bash
python -m module3.run_sam2_batch \
  --input-dir module3/uploads/ArUco \
  --output-dir module3/output \
  --checkpoint module3/sam2_checkpoints/sam2_hiera_small.pt \
  --device cpu
```

### Batch Processing

**ArUco Batch:**
```bash
python -m module3.process_aruco_batch \
  --input-dir module3/uploads/ArUco \
  --output-dir module3/output \
  --dictionary DICT_5X5_100
```

---

## Module 4: SIFT & Image Stitching

**Location:** [`module4/`](module4/)

### Purpose
From-scratch implementation of Scale-Invariant Feature Transform (SIFT) and panorama stitching with RANSAC homography.

### SIFT Implementation

**Full Pipeline:** ([sift.py](module4/sift.py))

#### 1. Gaussian Pyramid Construction
**Function:** `gaussian_pyramid()` ([sift.py:40-60](module4/sift.py#L40-L60))

Builds a multi-scale representation:
- **4 octaves** (default)
- **3 scales per octave** (plus 3 extra for DoG)
- **σ = 1.6** base scale

```
Octave 0: [σ₀, σ₁, σ₂, σ₃, σ₄, σ₅]
Octave 1: [2σ₀, 2σ₁, ...]  (½ resolution)
Octave 2: [4σ₀, 4σ₁, ...]  (¼ resolution)
...
```

**Scale progression:**
```
σᵢ = σ₀ × k^i, where k = 2^(1/scales_per_octave)
```

#### 2. Difference-of-Gaussian (DoG) Pyramid
**Function:** `dog_pyramid()` ([sift.py:63-70](module4/sift.py#L63-L70))

```
DoG(x,y,σ) = G(x,y,kσ) - G(x,y,σ)
```

Approximates scale-normalized Laplacian of Gaussian.

#### 3. Keypoint Detection
**Function:** `detect_keypoints()` ([sift.py:73-131](module4/sift.py#L73-L131))

**Extrema Detection:**
- Check 3×3×3 neighborhood (spatial + scale)
- Must be local maximum or minimum
- Reject low-contrast points (threshold = 0.015)

**Edge Response Filter:**
```
(Tr(H)² / Det(H)) < ((r+1)² / r)
```
Where H is Hessian matrix, r = edge_threshold (10.0)

Removes keypoints on edges (where principal curvature ratio is high).

#### 4. Orientation Assignment
**Function:** `assign_orientations()` ([sift.py:145-189](module4/sift.py#L145-L189))

For rotation invariance:
1. Extract 16×16 patch around keypoint
2. Compute gradient magnitude and orientation
3. Apply Gaussian weighting (σ = 1.5 × scale)
4. Build 36-bin orientation histogram
5. Smooth histogram (3-tap filter)
6. Assign dominant orientation to keypoint

#### 5. Descriptor Computation
**Function:** `compute_descriptors()` ([sift.py:191-258](module4/sift.py#L191-L258))

**128-dimensional descriptor:**
- 4×4 grid of cells
- 8-bin orientation histogram per cell
- Rotate gradients by keypoint orientation
- Apply Gaussian window for smooth weighting

**Normalization:**
1. L2 normalize
2. Clip values to 0.2 (illumination invariance)
3. Renormalize

```python
descriptor = [hist₁, hist₂, ..., hist₁₆]  # 16 cells × 8 bins = 128D
```

### Feature Matching

**Function:** `match_descriptors()` ([sift.py:269-281](module4/sift.py#L269-L281))

**Lowe's Ratio Test:**
```
if d₁ < 0.75 × d₂:
    accept match
```
Where d₁ = distance to nearest neighbor, d₂ = distance to 2nd nearest.

Reduces false matches by requiring distinctive matches.

### RANSAC Homography

**Function:** `ransac_homography()` ([sift.py:284-309](module4/sift.py#L284-L309))

**Algorithm:**
1. Randomly sample 4 match pairs
2. Compute homography H (DLT algorithm)
3. Project all points: `p'ᵢ = H × pᵢ`
4. Count inliers: `||p'ᵢ - p''ᵢ|| < threshold` (3.0 pixels)
5. Repeat 500 iterations
6. Keep H with most inliers

**Homography:**
```
[x']   [h₀ h₁ h₂]   [x]
[y'] = [h₃ h₄ h₅] × [y]
[1 ]   [h₆ h₇ h₈]   [1]
```

### Panorama Stitching

**Script:** [`stitcher.py`](module4/stitcher.py)

Uses OpenCV's built-in stitcher (SIFT under the hood) for production-quality results.

**Usage:**
```python
from module4.stitcher import stitch_images
import cv2

images = [
    cv2.imread("left.jpg"),
    cv2.imread("center.jpg"),
    cv2.imread("right.jpg")
]

result = stitch_images(images)

panorama_data_url = result["panorama"]
```

**CLI:**
```bash
python module4/cli.py stitch \
  --images left.jpg center.jpg right.jpg \
  --output panorama.png
```

**Features:**
- Automatic feature detection and matching
- Multi-band blending for seamless transitions
- Exposure compensation
- Automatic cropping of black borders

**Parameters:**
- `MAX_STITCH_DIM=1600`: Maximum dimension for processing (prevents timeouts)

### Performance

**SIFT Detection:**
- ~100-500 keypoints per image
- Processing time: 2-5 seconds per image

**Stitching:**
- 2-3 images: ~5-10 seconds
- 4-5 images: ~15-30 seconds

---

## Module 5: ArUco Marker Tracking

**Location:** [`module56/tracking.py`](module56/tracking.py) (detect_aruco_backend)

### Purpose
Real-time tracking using fiducial ArUco markers and QR codes for robust object localization.

### Features

#### 1. Multi-Dictionary Support
Supports all standard ArUco dictionaries:
- `DICT_4X4_50`, `DICT_5X5_100`, `DICT_6X6_250`, `DICT_7X7_1000`
- Automatically falls back to `DICT_4X4_50` if unknown

#### 2. Adaptive Detection
**Enhancement Pipeline:**
1. Try detection on original image
2. If no markers → Apply histogram equalization (CLAHE)
3. Retry detection on enhanced image

**Detection Parameters:** ([tracking.py:58-73](module56/tracking.py#L58-L73))
```python
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 23
params.cornerRefinementMethod = CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
```

Optimized for varying lighting and image quality.

#### 3. QR Code Fallback
If ArUco detection fails, attempts QR code detection:
```python
qr_detector = cv2.QRCodeDetector()
qr_data, qr_points, _ = qr_detector.detectAndDecode(img)
```

### Usage

**Python API:**
```python
from module56.tracking import detect_aruco_backend

with open("frame.jpg", "rb") as f:
    frame_bytes = f.read()

result = detect_aruco_backend(
    frame_bytes=frame_bytes,
    dict_name="DICT_5X5_100",
    max_dim=960
)

print(f"Detected {result['count']} markers")
for marker in result['markers']:
    print(f"  ID: {marker['id']}, Corners: {marker['corners']}")
```

**Output:**
```json
{
  "count": 2,
  "markers": [
    {
      "id": 42,
      "corners": [[100, 100], [200, 100], [200, 200], [100, 200]]
    },
    {
      "id": 17,
      "corners": [[300, 150], [400, 150], [400, 250], [300, 250]]
    }
  ],
  "visual": "data:image/png;base64,...",
  "dictionary": "DICT_5X5_100",
  "message": "Detected 2 ArUco marker(s) (IDs: [42, 17]) via DICT_5X5_100",
  "aruco_count": 2,
  "qr_count": 0
}
```

### Generate ArUco Markers

**Script:** [`generate_aruco_markers.py`](module3/generate_aruco_markers.py)

```bash
python -m module3.generate_aruco_markers \
  --dictionary DICT_5X5_100 \
  --ids 0 1 2 3 4 \
  --size 200 \
  --output module3/aruco_markers
```

**Output:**
- Individual marker images: `marker_0.png`, `marker_1.png`, ...
- Test image with all markers: `test_marker_DICT_5X5_100.png`

### Troubleshooting

**No markers detected:**
1. Increase screen brightness
2. Reduce glare/reflections (matte surface)
3. Move marker closer to camera
4. Try different dictionary (smaller markers work better close-up)
5. Print markers instead of displaying on screen

**Best Practices:**
- Marker size: At least 3% of frame width
- Lighting: Even, diffuse (avoid harsh shadows)
- Surface: Flat, non-reflective
- Distance: 20cm - 2m from camera

---

## Module 6: Markerless Object Tracking

**Location:** [`module56/tracking.py`](module56/tracking.py) (markerless_step_backend), [`module56/run_sam2_single.py`](module56/run_sam2_single.py)

### Purpose
Track objects without markers using color-based tracking or AI-powered SAM2 segmentation.

### Tracking Methods

#### 1. Color-Based Tracking
**Function:** `markerless_step_backend()` ([tracking.py:174-291](module56/tracking.py#L174-L291))

**Algorithm:**
1. Convert frame to HSV color space
2. Extract reference color from initial bounding box
3. Define search region (padded around previous box)
4. Compute color distance in HSV space
5. Find pixels matching reference color
6. Update bounding box to match cluster centroid

**Color Distance Metric:**
```python
distance = 0.6 × hue_distance + 0.3 × saturation_dist + 0.1 × value_dist
```

**Hue distance** handles circular wrapping:
```python
dh = min(|h₁ - h₂|, 180 - |h₁ - h₂|) / 90.0
```

**Parameters:**
- `search_pad`: Expansion around box (default: 40 pixels)
- `color_threshold`: Match threshold (default: 0.22)
- `stride`: Downsampling for speed (default: 2)

**Confidence Estimation:**
```python
confidence = min(matching_pixels / 1000.0, 1.0)
```

**Usage:**
```python
from module56.tracking import markerless_step_backend

with open("frame.jpg", "rb") as f:
    frame_bytes = f.read()

result = markerless_step_backend(
    frame_bytes=frame_bytes,
    x=100, y=100, w=50, h=50,  # Initial box
    color=[120, 200, 180],      # Optional HSV reference
    search_pad=40.0
)

new_box = result["box"]  # Updated position
confidence = result["confidence"]
```

#### 2. SAM2 Segmentation
**Script:** [`run_sam2_single.py`](module56/run_sam2_single.py)

Uses Meta's Segment Anything Model 2 for one-shot object segmentation.

**Features:**
- Automatic multi-object detection
- Stability scoring for best mask selection
- No manual annotation required

**Usage:**
```bash
python -m module56.run_sam2_single \
  --input bottle_frame.jpg \
  --output bottle_mask.png \
  --checkpoint module3/sam2_checkpoints/sam2_hiera_small.pt \
  --device cpu \
  --points-per-side 16
```

**Algorithm:**
1. Load SAM2 model with checkpoint
2. Generate automatic masks (grid-based prompts)
3. Score masks by:
   - Predicted IoU (model confidence)
   - Stability score (consistent across thresholds)
   - Area (prefer substantial objects)
4. Select best mask
5. Save binary mask + overlay visualization

**Output:**
- `bottle_mask.png` - Binary mask (white=object, black=background)
- `bottle_mask_overlay.png` - Visual preview

**Parameters:**
- `--points-per-side`: Density of sampling grid (8, 16, 32)
  - Higher = more accurate but slower
- `--device`: `cpu`, `cuda`, `mps`

### SAM2 Tracking Workflow

**Complete Guide:** See [`module56/SAM2_TRACKER_GUIDE.md`](module56/SAM2_TRACKER_GUIDE.md)

**Quick Start:**
1. Capture reference frame from webcam
2. Run SAM2 segmentation offline
3. Upload mask to tracker
4. Track in real-time using color-based tracking

**Why SAM2?**
- No manual bounding box needed
- Better than pure color tracking for complex shapes
- Combines AI (initialization) + real-time (tracking)

---

## Module 7: Stereo Vision & 3D Measurement

**Location:** [`module7/stereo.py`](module7/stereo.py)

### Purpose
Compute 3D depth and measure real-world distances using stereo image pairs.

### Algorithm

#### 1. Disparity Map Computation
**Function:** `disparity_map()` ([stereo.py:31-47](module7/stereo.py#L31-L47))

Uses **Semi-Global Block Matching (SGBM)**:
```python
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=96,      # Max disparity search range
    blockSize=7,            # Matching block size
    P1=8 * 3 * 7²,         # Smoothness penalty (small diff)
    P2=32 * 3 * 7²,        # Smoothness penalty (large diff)
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=2,
    disp12MaxDiff=1
)
```

**Parameters Explained:**
- **numDisparities**: Must be divisible by 16. Larger = more depth range.
- **blockSize**: Odd number. Larger = smoother but less detail.
- **P1, P2**: Regularization. Enforces smooth disparity changes.
- **uniquenessRatio**: Margin by which best match must win (%)
- **speckleWindow/Range**: Noise filtering

**Output:** Disparity map (pixels), where:
```
disparity(x,y) = x_left - x_right
```

#### 2. Depth from Disparity
**Function:** `polygon_measurements()` ([stereo.py:66-111](module7/stereo.py#L66-L111))

**Depth Formula:**
```
Z = (f × B) / d
```
Where:
- `Z` = depth (mm)
- `f` = focal length (pixels)
- `B` = baseline (mm) - distance between cameras
- `d` = disparity (pixels)

**3D Reconstruction:**
```
X = (x - cₓ) × Z / f
Y = (y - cᵧ) × Z / f
Z = (f × B) / d
```

#### 3. Polygon Measurement
For each vertex of a polygon:
1. Extract local 7×7 patch from disparity map
2. Take median disparity (robust to noise)
3. Compute 3D coordinates (X, Y, Z)

For each edge:
1. Compute 3D distance between vertices:
   ```
   length = √(ΔX² + ΔY² + ΔZ²)
   ```

### Camera Calibration

**Focal Length in Pixels:**
```python
f_pixels = (f_mm / sensor_width_mm) × image_width_pixels
```

**Typical Values:**
- Smartphone: f=4.5mm, sensor=5.6mm
- Webcam: f=3.6mm, sensor=4.8mm
- DSLR: f=50mm, sensor=36mm (full frame)

### Usage

**Python API:**
```python
from module7.stereo import stereo_measurement, load_image

# Load stereo pair
with open("left.jpg", "rb") as f:
    left = load_image(f.read())
with open("right.jpg", "rb") as f:
    right = load_image(f.read())

# Define polygon (e.g., rectangle)
polygon = [
    {"x": 100, "y": 100},
    {"x": 200, "y": 100},
    {"x": 200, "y": 200},
    {"x": 100, "y": 200}
]

result = stereo_measurement(
    left=left,
    right=right,
    polygon=polygon,
    focal_length_mm=4.5,
    sensor_width_mm=5.6,
    baseline_mm=65.0  # Distance between cameras
)

print(f"Mean edge length: {result['mean_length_mm']:.2f} mm")

for i, seg in enumerate(result['segments']):
    print(f"Edge {i}: {seg['length_mm']:.2f} mm")
```

**Output:**
```json
{
  "disparity": "data:image/png;base64,...",
  "vertices": [
    {"x_mm": 10.5, "y_mm": 8.2, "z_mm": 450.3, "pixel_x": 100, "pixel_y": 100},
    ...
  ],
  "segments": [
    {
      "start": {"x": 100, "y": 100, "z_mm": 450.3},
      "end": {"x": 200, "y": 100, "z_mm": 448.7},
      "length_mm": 52.3
    },
    ...
  ],
  "mean_length_mm": 51.8
}
```

### Stereo Rig Setup

**Requirements:**
1. Two cameras with identical intrinsic parameters
2. Cameras mounted in parallel (rectified setup)
3. Known baseline distance
4. Synchronized capture

**Software Rectification:**
If cameras aren't perfectly aligned, use OpenCV calibration:
```python
# Compute rectification transforms
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    imageSize, R, T
)

# Apply rectification
left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
```

### Accuracy Considerations

**Depth Uncertainty:**
```
σ_Z = (Z² / (f × B)) × σ_d
```

Where σ_d ≈ 0.5-1.0 pixels (disparity uncertainty)

**Practical Limits:**
- **Minimum depth**: Limited by max disparity
- **Maximum depth**: Limited by matching accuracy
- **Typical range**: 0.3m - 10m for stereo rig

**Improving Accuracy:**
1. Increase baseline (but harder to match)
2. Use higher resolution cameras
3. Better lighting (improves matching)
4. Calibrate cameras properly
5. Use textured backgrounds (helps matching)

---

## Web Application

### Architecture

**Frontend:** Next.js 15 + TypeScript
- Modern React with App Router
- Real-time camera access via MediaDevices API
- Canvas-based visualization
- Tailwind CSS styling

**Backend:** Flask + Python
- RESTful API endpoints
- CORS-enabled for Next.js integration
- File upload handling
- Real-time processing

### Running the Application

**Development Mode:**

1. **Start Flask backend:**
```bash
python app.py
# Runs on http://localhost:5000
```

2. **Start Next.js frontend:**
```bash
npm run dev
# Runs on http://localhost:3000
```

**Production Mode:**
```bash
npm run build
npm start
```

### Available Pages

| Route | Module | Description |
|-------|--------|-------------|
| `/` | - | Home page with module overview |
| `/measure` | 1 | Real-world measurement using perspective projection |
| `/template-matching` | 2a | Upload scene, detect objects |
| `/fourier` | 2b | Image deblurring demo |
| `/assignment3` | 3 | Gradients, keypoints, segmentation |
| `/assignment4` | 4 | SIFT features, panorama stitching |
| `/assignment56` | 5-6 | ArUco + markerless tracking |
| `/assignment7` | 7 | Stereo 3D measurement |

### API Endpoints

#### Module 1: Camera Calibration & Measurement

**Calibrate Camera**
```http
POST /calibrate
Content-Type: application/json

{
  "pixel_width": 428,
  "real_width": 85.6,
  "distance": 50,
  "units": "mm"
}

Response:
{
  "status": "success",
  "focal_length_pixels": 250.0,
  "units": "mm"
}
```

**Measure Object**
```http
POST /measure
Content-Type: application/json

{
  "pixel_width": 600,
  "distance": 50
}

Response:
{
  "pixel_width": 600,
  "distance": 50,
  "real_size": 120.0,
  "units": "mm"
}
```

#### Module 2a: Template Matching
```http
POST /api/template-matching
Content-Type: multipart/form-data

scene: <file>
threshold: 0.7
only: ["stop-sign", "yield-sign"]  # optional
```

#### Fourier Deblur
```http
POST /api/fourier
Content-Type: multipart/form-data

image: <file>
```

#### Module 3 Operations
```http
POST /api/assignment3/gradients
Content-Type: multipart/form-data

image: <file>
```

```http
POST /api/assignment3/keypoints
Content-Type: multipart/form-data

image: <file>
mode: "edge" | "corner"
```

```http
POST /api/assignment3/aruco-segment
Content-Type: multipart/form-data

image: <file>
dictionary: "DICT_5X5_100"
```

#### SIFT Matching
```http
POST /api/assignment4/match
Content-Type: multipart/form-data

image1: <file>
image2: <file>
```

#### Panorama Stitching
```http
POST /api/assignment4/stitch
Content-Type: multipart/form-data

images: <file[]>
```

#### ArUco Detection
```http
POST /api/assignment56/detect-aruco
Content-Type: application/json

{
  "frame": "data:image/jpeg;base64,...",
  "dictionary": "DICT_5X5_100"
}
```

#### Markerless Tracking
```http
POST /api/assignment56/markerless-step
Content-Type: application/json

{
  "frame": "data:image/jpeg;base64,...",
  "x": 100, "y": 100, "w": 50, "h": 50,
  "color": [120, 200, 180]  # optional
}
```

#### Stereo Measurement
```http
POST /api/assignment7/measure
Content-Type: multipart/form-data

left_image: <file>
right_image: <file>
polygon: [[100,100], [200,100], [200,200], [100,200]]
focal_length_mm: 4.5
sensor_width_mm: 5.6
baseline_mm: 65.0
```

---

## Docker Deployment

**Dockerfile:** [`Dockerfile`](Dockerfile)

```bash
# Build image
docker build -t cv-modules .

# Run container
docker run -p 3000:3000 -p 5000:5000 cv-modules
```

**Environment Variables:**
```bash
# Module output directories
MODULE2_OUTPUT_DIR=/tmp/module2_output
MODULE3_OUTPUT_DIR=/tmp/module3_output
MODULE3_UPLOAD_DIR=/tmp/module3_uploads

# Stitching limits
MODULE4_STITCH_MAX_DIM=1600

# Vercel deployment flag
VERCEL=1
```

---

## Project Structure

```
computer_vision/
├── app/                          # Next.js frontend
│   ├── page.tsx                 # Home page
│   ├── measure/                 # Module 1 UI
│   ├── template-matching/       # Module 2a UI
│   ├── fourier/                 # Module 2b UI
│   ├── assignment3/             # Module 3 UI
│   ├── assignment4/             # Module 4 UI
│   ├── assignment56/            # Module 5-6 UI
│   └── assignment7/             # Module 7 UI
│
├── measure.py                    # Module 1: Perspective projection utilities
│
├── module2/                      # Module 2: Template Matching + Fourier
│   ├── detector.py              # Core matching logic
│   ├── detector_core.py         # Low-level utilities
│   ├── fourier_deblur.py        # Deblurring pipeline
│   ├── run_template_matching.py # CLI script
│   ├── data/
│   │   ├── templates/           # Template images
│   │   └── metadata.json        # Template metadata
│   └── output/                  # Results
│
├── module3/                      # Image Processing
│   ├── pipelines.py             # Gradients, keypoints, segmentation
│   ├── generate_aruco_markers.py
│   ├── process_aruco_batch.py
│   ├── run_sam2_batch.py
│   ├── cli.py                   # CLI interface
│   ├── aruco_markers/           # Generated markers
│   ├── sam2_checkpoints/        # SAM2 model weights
│   ├── uploads/                 # Input images
│   └── output/                  # Results
│
├── module4/                      # SIFT + Stitching
│   ├── sift.py                  # SIFT implementation
│   ├── stitcher.py              # Panorama stitching
│   ├── cli.py                   # CLI interface
│   ├── uploads/                 # Input images
│   └── output/                  # Results
│
├── module56/                     # Tracking
│   ├── tracking.py              # ArUco + markerless tracking
│   ├── run_sam2_single.py       # SAM2 segmentation script
│   └── SAM2_TRACKER_GUIDE.md    # Complete SAM2 guide
│
├── module7/                      # Stereo Vision
│   ├── stereo.py                # Disparity + 3D measurement
│   ├── cli.py                   # CLI interface
│   ├── uploads/                 # Input stereo pairs
│   └── output/                  # Results
│
├── components/                   # React components
├── public/                       # Static assets
├── app.py                        # Flask backend
├── requirements.txt              # Python dependencies
├── package.json                  # Node dependencies
├── Dockerfile                    # Container build
└── README.md                     # This file
```

---

## Development Notes

### Adding New Templates (Module 2)
1. Create PNG file in `module2/data/templates/`
2. Filename becomes label (e.g., `apple.png` → "apple")
3. Template auto-crops to content
4. No metadata required

### Adjusting Detection Thresholds
Edit `module2/detector.py`:
```python
DEFAULT_THRESHOLD = 0.4  # Global threshold

TEMPLATE_CONFIG = {
    "coke": TemplateConfig(threshold=0.5),
    "stop-sign": TemplateConfig(threshold=0.6),
}
```

### Custom Scale Ranges
```python
SCALE_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)  # Fewer scales = faster
```

### SAM2 Checkpoints
Download from: https://github.com/facebookresearch/segment-anything-2/tree/main/checkpoints

Available models:
- `sam2_hiera_tiny.pt` - Fastest, least accurate
- `sam2_hiera_small.pt` - **Recommended** balance
- `sam2_hiera_base_plus.pt` - Most accurate, slowest

---

## Performance Benchmarks

**Hardware:** i5-10400, 16GB RAM, integrated GPU

| Module | Operation | Time | Notes |
|--------|-----------|------|-------|
| 1 | Calibration calculation | <0.001s | Simple arithmetic |
| 1 | Measurement calculation | <0.001s | Simple arithmetic |
| 2a | Template matching (1 scene, 10 templates) | ~2s | Multi-scale search |
| 2b | Fourier deblur (1024×768) | ~0.5s | Per-channel FFT |
| 3 | Gradient computation | ~0.1s | Sobel + LoG |
| 3 | ArUco segmentation | ~0.3s | With GrabCut fallback |
| 3 | SAM2 segmentation | ~5-15s | CPU, small model |
| 4 | SIFT detection (per image) | ~3s | ~300 keypoints |
| 4 | Panorama stitch (3 images) | ~8s | With blending |
| 5 | ArUco detection | ~0.05s | Real-time |
| 6 | Color tracking step | ~0.03s | Real-time |
| 7 | Stereo disparity (640×480) | ~0.8s | SGBM |

---

## Troubleshooting

### Common Issues

**ImportError: No module named 'cv2'**
```bash
pip install opencv-contrib-python
```

**SAM2 not found**
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

**Next.js build fails**
```bash
rm -rf .next node_modules
npm install
npm run build
```

**Flask CORS errors**
Check that Flask includes CORS headers (already configured in `app.py`).

**Module import errors**
Ensure working directory is project root:
```bash
cd computer_vision
python -m module2.detector  # Correct
```

### Performance Issues

**Template matching too slow:**
- Reduce `SCALE_FACTORS` count
- Resize scene to lower resolution
- Use fewer templates

**Stitching timeout:**
- Reduce `MODULE4_STITCH_MAX_DIM`
- Use fewer images (≤3)
- Ensure sufficient overlap (30-50%)

**SAM2 out of memory:**
- Use `--device cpu` instead of GPU
- Reduce `--points-per-side`
- Resize input image

---

## Contributing

### Code Style
- Python: PEP 8, type hints preferred
- TypeScript: ESLint + Prettier
- Comments: Docstrings for all public functions

### Testing
```bash
# Python
pytest tests/

# JavaScript
npm test
```

### Pull Requests
1. Fork repository
2. Create feature branch
3. Add tests
4. Submit PR with description

---

## License

MIT License - See LICENSE file for details.

---

## References

### Academic Papers
1. Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints." IJCV.
2. Bradski, G. (1998). "The OpenCV Library." Dr. Dobb's Journal.
3. Hirschmuller, H. (2008). "Stereo Processing by Semiglobal Matching." PAMI.
4. Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos." Meta AI.
5. Garrido-Jurado, S. et al. (2014). "Automatic generation and detection of highly reliable fiducial markers under occlusion." Pattern Recognition.

### Libraries
- OpenCV: https://opencv.org/
- NumPy: https://numpy.org/
- Flask: https://flask.palletsprojects.com/
- Next.js: https://nextjs.org/
- SAM2: https://github.com/facebookresearch/segment-anything-2

---

## Acknowledgments

- Meta AI Research for SAM2
- OpenCV contributors
- ArUco marker system developers
- Computer vision community

---

**Built with ❤️ for computer vision education and research**
