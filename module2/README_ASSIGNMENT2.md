# Assignment 2 Part 1: Template Matching using Correlation

This module implements correlation-based template matching for object detection as required by Assignment 2, Part 1.

## Requirements

- Demonstrate detection of objects in images using template matching with correlation method
- Templates must be taken from completely different scenes (not cropped from the test image)
- Evaluate 10 objects (can be in the same or different images)

## Implementation

The implementation uses **normalized cross-correlation** (`cv2.TM_CCOEFF_NORMED`) for robust template matching that is invariant to brightness and contrast variations.

### Key Components

1. **`detector_core.py`**: Core implementation with:
   - `match_template_correlation()`: Performs correlation-based matching
   - `detect_single_template()`: Detects a single template in a scene
   - `detect_scene_with_templates()`: Detects multiple templates in a scene
   - `evaluate_all_objects()`: Evaluates all 10 objects across scenes
   - `visualize_detection()`: Creates visualization images
   - `generate_evaluation_report()`: Generates text reports

2. **`detector.py`**: Main evaluation script with CLI interface

3. **`test_assignment2.py`**: Test script to verify the implementation

## Usage

### Quick Test

```bash
python module2/test_assignment2.py
```

### Full Evaluation

```bash
# Evaluate on all scene images with visualization
python -m module2.detector --scenes module2/data/scenes/*.png --visualize --threshold 0.6

# Evaluate on specific scenes
python -m module2.detector --scenes scene1.jpg scene2.jpg --visualize

# Custom output directory
python -m module2.detector --scenes *.jpg --output-dir results/my_evaluation
```

### Command Line Options

- `--scenes`: One or more scene image paths (required)
- `--threshold`: Correlation threshold (default: 0.6)
- `--output-dir`: Directory to save results (default: results/assignment2)
- `--visualize`: Generate visualization images for each detection

## Output

The evaluation generates:

1. **`evaluation_report.txt`**: Text report with detection statistics
2. **`results.json`**: JSON file with all detection results
3. **`visualizations/`**: Directory with visualization images (if `--visualize` is used)

## Template Requirements

The assignment requires 10 objects. Currently, the dataset includes:
- Real object templates (cheerios, iphone, stanley, razor, coke, logos)
- Synthetic shape templates can be regenerated with `generate_dataset.py` if you want extra fillers

The code now warns if fewer than 10 real templates exist; add more real templates to `module2/data/templates/` and update `module2/data/metadata.json` to reach 10.

## How It Works

1. **Template Loading**: Loads template images from metadata
2. **Correlation Matching**: Uses `cv2.matchTemplate()` with `TM_CCOEFF_NORMED` method
3. **Threshold Filtering**: Only accepts matches above the correlation threshold
4. **Bounding Box Extraction**: Extracts location of best match
5. **Visualization**: Draws bounding boxes and labels on scene images

## Correlation Method

The implementation uses **Normalized Cross-Correlation** which:
- Is robust to brightness variations
- Is robust to contrast variations
- Returns values between -1 and 1 (1 = perfect match)
- Is computed as: `correlation = (template · scene_window) / (||template|| × ||scene_window||)`

## Example Output

```
======================================================================
ASSIGNMENT 2: TEMPLATE MATCHING EVALUATION
======================================================================
Testing 3 scene(s)
Correlation threshold: 0.6
Output directory: results/assignment2
======================================================================

Running template matching on all 10 objects...

Template: adidas_cap
  Detections: 2/3 (66.7%)
  Avg Confidence: 0.723
  Max Confidence: 0.856
  Min Confidence: 0.612
...
```

## Files

- `detector_core.py`: Core detection implementation
- `detector.py`: Main evaluation script
- `test_assignment2.py`: Test script
- `generate_dataset.py`: Dataset generator for synthetic templates

