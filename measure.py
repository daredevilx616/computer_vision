"""
Module 1 measurement helpers.
How to run the demo app: launch the web UI (or run `python measure.py` as a utility) to capture a frame, click two points,
and compute real-world distance using Z (m), focal length (mm), and sensor width (mm).
"""
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
