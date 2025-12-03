"""
Core detector implementation for Assignment 2 - Template Matching using Correlation

This module implements correlation-based template matching for object detection.
Uses normalized cross-correlation (cv2.TM_CCOEFF_NORMED) for robust matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .generate_dataset import BASE_DIR

# Default correlation threshold
DEFAULT_THRESHOLD = 0.6

# Paths
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = DATA_DIR / "templates"
SCENE_DIR = DATA_DIR / "scenes"
METADATA_PATH = DATA_DIR / "metadata.json"


@dataclass
class Template:
    """Represents a template image for matching."""
    name: str
    path: Path
    image: np.ndarray
    width: int
    height: int

    @classmethod
    def from_path(cls, name: str, path: Path) -> Template:
        """Load a template from a file path."""
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load template: {path}")
        h, w = image.shape[:2]
        return cls(name=name, path=path, image=image, width=w, height=h)


@dataclass
class Detection:
    """Represents a detected object in a scene."""
    template_name: str
    confidence: float
    box: Tuple[int, int, int, int]  # (left, top, right, bottom)
    scene_name: str = ""


@dataclass
class DetectionResult:
    """Result of template matching on a single template."""
    template_name: str
    detected: bool
    confidence: float
    box: Optional[Tuple[int, int, int, int]]  # (left, top, right, bottom)
    scene_name: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "template_name": self.template_name,
            "detected": self.detected,
            "confidence": float(self.confidence),
            "box": list(self.box) if self.box else None,
            "scene_name": self.scene_name,
        }


def ensure_dataset() -> Dict:
    """Ensure dataset exists, generate if needed."""
    if not METADATA_PATH.exists():
        from .generate_dataset import generate_dataset
        return generate_dataset()
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_templates(metadata: Optional[Dict] = None, required_only: bool = False) -> List[Template]:
    """
    Load all templates from metadata.
    
    Args:
        metadata: Dataset metadata dict. If None, loads from file.
        required_only: If True, only load templates needed for 10-object evaluation.
    
    Returns:
        List of Template objects.
    """
    if metadata is None:
        metadata = ensure_dataset()
    
    templates = []
    template_entries = metadata.get("templates", [])
    
    # For assignment, we need 10 objects
    # Use available templates and supplement with generated ones if needed
    for entry in template_entries:
        template_path = BASE_DIR / entry["path"]
        if template_path.exists():
            try:
                template = Template.from_path(entry["name"], template_path)
                templates.append(template)
            except Exception as e:
                print(f"Warning: Failed to load template {entry['name']}: {e}")
    
    missing = max(0, 10 - len(templates))
    if missing > 0:
        print(
            f"Warning: Only {len(templates)} templates available (need 10). "
            f"Add more real templates under {TEMPLATE_DIR}"
        )

    return templates[:10] if required_only else templates


def match_template_correlation(scene: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Perform correlation-based template matching.
    
    Uses normalized cross-correlation (TM_CCOEFF_NORMED) which is robust to
    brightness and contrast variations.
    
    Args:
        scene: Scene image (BGR)
        template: Template image (BGR)
    
    Returns:
        Tuple of (correlation map, max correlation value, location of max)
    """
    # Convert to grayscale for matching (more robust)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Perform normalized cross-correlation
    result = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Find maximum correlation value and location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    return result, max_val, max_loc


def detect_single_template(
    scene_path: Path,
    template: Template,
    threshold: float = DEFAULT_THRESHOLD
) -> DetectionResult:
    """
    Detect a single template in a scene image.
    
    Args:
        scene_path: Path to scene image
        template: Template object to search for
        threshold: Correlation threshold for detection
    
    Returns:
        DetectionResult with detection information
    """
    scene = cv2.imread(str(scene_path))
    if scene is None:
        raise ValueError(f"Failed to load scene: {scene_path}")
    
    scene_name = scene_path.stem
    
    # Resize template if larger than scene to avoid OpenCV assertion
    tpl_img = template.image
    sh, sw = scene.shape[:2]
    th, tw = tpl_img.shape[:2]
    if tw > sw or th > sh:
        scale = min(sw / max(1, tw), sh / max(1, th), 1.0)
        new_w = max(4, int(tw * scale))
        new_h = max(4, int(th * scale))
        tpl_img = cv2.resize(tpl_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        tw, th = new_w, new_h

    # Perform correlation matching
    correlation_map, max_corr, max_loc = match_template_correlation(scene, tpl_img)
    
    detected = max_corr >= threshold
    
    box = None
    if detected:
        # Calculate bounding box
        left = max_loc[0]
        top = max_loc[1]
        right = left + tw
        bottom = top + th
        box = (left, top, right, bottom)
    
    return DetectionResult(
        template_name=template.name,
        detected=detected,
        confidence=float(max_corr),
        box=box,
        scene_name=scene_name,
    )


def detect_scene_with_templates(
    scene_path: Path,
    templates: List[Template],
    threshold: float = DEFAULT_THRESHOLD
) -> Dict:
    """
    Detect all templates in a scene image.
    
    Args:
        scene_path: Path to scene image
        templates: List of Template objects to search for
        threshold: Correlation threshold for detection
    
    Returns:
        Dictionary with detections and metadata
    """
    scene = cv2.imread(str(scene_path))
    if scene is None:
        raise ValueError(f"Failed to load scene: {scene_path}")
    
    detections = []
    
    for template in templates:
        correlation_map, max_corr, max_loc = match_template_correlation(scene, template.image)
        
        if max_corr >= threshold:
            left = max_loc[0]
            top = max_loc[1]
            right = left + template.width
            bottom = top + template.height
            
            detections.append({
                "label": template.name,
                "confidence": float(max_corr),
                "box": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                }
            })
    
    return {
        "detections": detections,
        "threshold": threshold,
    }


def annotate_detections(scene: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on scene image.
    
    Args:
        scene: Scene image
        detections: List of detection dictionaries
    
    Returns:
        Annotated image
    """
    annotated = scene.copy()
    
    for det in detections:
        box = det["box"]
        left = box["left"]
        top = box["top"]
        right = box["right"]
        bottom = box["bottom"]
        confidence = det["confidence"]
        label = det["label"]
        
        # Draw bounding box
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw label with confidence
        label_text = f"{label}: {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (left, top - text_height - baseline - 5),
            (left + text_width, top),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            annotated,
            label_text,
            (left, top - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    
    return annotated


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: (left, top, right, bottom)
        box2: (left, top, right, bottom)
    
    Returns:
        IoU value between 0 and 1
    """
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    
    # Calculate intersection
    inter_left = max(left1, left2)
    inter_top = max(top1, top2)
    inter_right = min(right1, right2)
    inter_bottom = min(bottom1, bottom2)
    
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0
    
    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    
    # Calculate union
    box1_area = (right1 - left1) * (bottom1 - top1)
    box2_area = (right2 - left2) * (bottom2 - top2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def non_max_suppression(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_threshold: float = 0.3,
) -> List[int]:
    """
    Basic NMS to keep highest-score boxes and drop heavy overlaps.
    Returns indices of boxes to keep.
    """
    if not boxes:
        return []
    
    idxs = list(np.argsort(scores)[::-1])
    keep: List[int] = []
    
    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [
            j for j in idxs
            if compute_iou(boxes[current], boxes[j]) <= iou_threshold
        ]
    return keep


def detect_all_template_matches(
    scene_path: Path,
    template: Template,
    threshold: float = DEFAULT_THRESHOLD,
    iou_threshold: float = 0.3,
    scales: Optional[Tuple[float, ...]] = None,
    guard_templates: Optional[List[Template]] = None,
    guard_margin: float = 0.02,
) -> List[Detection]:
    """
    Detect all matches of a template in a scene using correlation + NMS.
    Useful for repeated symbols (e.g., card pips).
    
    If guard_templates are provided, each candidate box is re-scored against
    those alternatives and kept only if the target template wins by margin.
    """
    scene = cv2.imread(str(scene_path))
    if scene is None:
        raise ValueError(f"Failed to load scene: {scene_path}")

    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template.image, cv2.COLOR_BGR2GRAY)

    if scales is None:
        # Default scales cover small repeated marks up to original size.
        scales = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)

    locs: List[Tuple[int, int]] = []
    sizes: List[Tuple[int, int]] = []
    scores_at_locs: List[float] = []

    for scale in scales:
        if scale <= 0:
            continue
        w = max(4, int(round(template.width * scale)))
        h = max(4, int(round(template.height * scale)))
        if w > scene_gray.shape[1] or h > scene_gray.shape[0]:
            continue

        tpl_scaled = cv2.resize(template_gray, (w, h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(scene_gray, tpl_scaled, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):  # (x, y)
            locs.append((int(pt[0]), int(pt[1])))
            sizes.append((w, h))
            scores_at_locs.append(float(result[pt[1], pt[0]]))

    boxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []

    for (left, top), (w, h), score in zip(locs, sizes, scores_at_locs):
        right = left + w
        bottom = top + h
        boxes.append((left, top, right, bottom))
        scores.append(score)

    keep_idx = non_max_suppression(boxes, scores, iou_threshold)

    detections: List[Detection] = []
    for idx in keep_idx:
        box = boxes[idx]
        score = scores[idx]

        if guard_templates:
            # Re-score this ROI against target and guards; keep only if target wins.
            l, t, r, b = box
            roi = scene_gray[t:b, l:r]
            if roi.size == 0:
                continue
            def score_tpl(tpl: Template) -> float:
                tpl_img = cv2.cvtColor(tpl.image, cv2.COLOR_BGR2GRAY)
                tpl_resized = cv2.resize(tpl_img, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, tpl_resized, cv2.TM_CCOEFF_NORMED)
                return float(cv2.minMaxLoc(res)[1])

            target_score = score_tpl(template)
            alt_scores = [score_tpl(gt) for gt in guard_templates]
            best_alt = max(alt_scores) if alt_scores else -1.0
            if best_alt + guard_margin >= target_score:
                continue  # another template matches this ROI as well or better
            score = target_score  # use refined score

        detections.append(
            Detection(
                template_name=template.name,
                confidence=score,
                box=box,
                scene_name=scene_path.stem,
            )
        )

    return detections


def evaluate_all_objects(
    scene_paths: List[Path],
    threshold: float = DEFAULT_THRESHOLD
) -> List[DetectionResult]:
    """
    Evaluate all 10 objects across given scene images.
    
    Args:
        scene_paths: List of paths to scene images
        threshold: Correlation threshold for detection
    
    Returns:
        List of DetectionResult objects
    """
    metadata = ensure_dataset()
    templates = load_templates(metadata, required_only=True)
    
    # Ensure we have exactly 10 templates
    if len(templates) < 10:
        print(f"Warning: Only {len(templates)} templates available, need 10")
    
    results = []
    
    for scene_path in scene_paths:
        scene_name = scene_path.stem
        
        for template in templates[:10]:  # Use first 10 templates
            result = detect_single_template(scene_path, template, threshold)
            result.scene_name = scene_name
            results.append(result)
    
    return results


def visualize_detection(
    scene_path: Path,
    template: Template,
    result: DetectionResult,
    output_path: Path
) -> None:
    """
    Create a visualization showing template, scene, and detection.
    
    Args:
        scene_path: Path to scene image
        template: Template object
        result: DetectionResult
        output_path: Where to save visualization
    """
    scene = cv2.imread(str(scene_path))
    if scene is None:
        raise ValueError(f"Failed to load scene: {scene_path}")
    
    # Create visualization
    h, w = scene.shape[:2]
    template_h, template_w = template.image.shape[:2]
    
    # Create a side-by-side visualization
    vis_width = w + template_w + 40
    vis_height = max(h, template_h) + 100
    vis = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 255
    
    # Place scene on left
    vis[:h, :w] = scene
    
    # Place template on right
    vis[:template_h, w + 20:w + 20 + template_w] = template.image
    
    # Draw detection box if detected
    if result.detected and result.box:
        left, top, right, bottom = result.box
        cv2.rectangle(vis, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Add label
        label_text = f"{result.template_name}: {result.confidence:.3f}"
        cv2.putText(
            vis,
            label_text,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    
    # Add title
    title = f"Template: {template.name} | Scene: {scene_path.stem}"
    cv2.putText(
        vis,
        title,
        (10, vis_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    
    # Add status
    status = f"Detected: {result.detected} | Confidence: {result.confidence:.3f}"
    cv2.putText(
        vis,
        status,
        (10, vis_height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255) if not result.detected else (0, 255, 0),
        2,
    )
    
    cv2.imwrite(str(output_path), vis)


def generate_evaluation_report(results: List[DetectionResult]) -> str:
    """
    Generate a text report from evaluation results.
    
    Args:
        results: List of DetectionResult objects
    
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("TEMPLATE MATCHING EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Group by template
    from collections import defaultdict
    by_template = defaultdict(list)
    for result in results:
        by_template[result.template_name].append(result)
    
    for template_name in sorted(by_template.keys()):
        template_results = by_template[template_name]
        detected_count = sum(1 for r in template_results if r.detected)
        total_count = len(template_results)
        
        lines.append(f"Template: {template_name}")
        lines.append(f"  Detections: {detected_count}/{total_count} ({100*detected_count/total_count:.1f}%)")
        
        if template_results:
            avg_conf = np.mean([r.confidence for r in template_results])
            max_conf = max(r.confidence for r in template_results)
            min_conf = min(r.confidence for r in template_results)
            lines.append(f"  Avg Confidence: {avg_conf:.3f}")
            lines.append(f"  Max Confidence: {max_conf:.3f}")
            lines.append(f"  Min Confidence: {min_conf:.3f}")
        
        lines.append("")
    
    # Overall statistics
    total = len(results)
    detected = sum(1 for r in results if r.detected)
    lines.append("-" * 70)
    lines.append(f"Overall: {detected}/{total} detections ({100*detected/total:.1f}%)")
    lines.append("=" * 70)
    
    return "\n".join(lines)

