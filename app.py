from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import os
from pathlib import Path
import numpy as np

from module2.fourier_deblur import process_image as fourier_process

app = Flask(__name__)

# Calibration and measurement storage
CALIBRATION = {"focal_length_pixels": None, "units": "cm"}
LAST_MEASUREMENT = {}

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/healthz")
def health():
    return "ok"


def _bytes_to_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image buffer")
    return img


def _to_data_url(image: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image buffer.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response("Camera not available", status=503)

@app.route("/calibrate", methods=["POST"])
def calibrate():
    """Calibrate using reference object"""
    data = request.get_json()
    try:
        pixel_width = float(data.get("pixel_width"))
        real_width = float(data.get("real_width"))
        distance = float(data.get("distance"))
        units = data.get("units", "cm")
        focal_length = (pixel_width * distance) / real_width
        CALIBRATION.update({"focal_length_pixels": focal_length, "units": units})
        print(f"[CALIBRATION] pixel_width={pixel_width}, real_width={real_width}, distance={distance}, f_pixels={focal_length}")
        return jsonify({"status":"success", "focal_length_pixels": focal_length, "units": units})
    except Exception as e:
        print("Calibration error:", e)
        return jsonify({"status":"error", "message": str(e)}), 400

@app.route("/measure", methods=["POST"])
def measure():
    """Measure real-world size of object"""
    data = request.get_json()
    try:
        if CALIBRATION["focal_length_pixels"] is None:
            return jsonify({"status":"error","message":"Calibrate first"}), 400
        pixel_width = float(data.get("pixel_width"))
        distance = float(data.get("distance"))
        real_size = (pixel_width * distance) / CALIBRATION["focal_length_pixels"]
        LAST_MEASUREMENT.update({"pixel_width": pixel_width, "distance": distance, "real_size": real_size, "units": CALIBRATION["units"]})
        print(f"[MEASUREMENT] pixel_width={pixel_width}, distance={distance}, real_size={real_size}")
        return jsonify(LAST_MEASUREMENT)
    except Exception as e:
        print("Measurement error:", e)
        return jsonify({"status":"error", "message": str(e)}), 400

@app.route("/api/fourier", methods=["POST"])
def api_fourier():
    """Fourier blur/deblur pipeline for Module 2 (backend endpoint)."""
    if "image" not in request.files:
        return jsonify({"error": "Missing image upload."}), 400

    up = request.files["image"]
    if up.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Use /tmp by default in hosted environments
    base_tmp = Path(os.getenv("MODULE2_OUTPUT_DIR", "/tmp/module2_output"))
    uploads_dir = base_tmp / "uploads"
    out_dir = base_tmp / "fourier"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = uploads_dir / up.filename
    up.save(save_path)

    try:
        result = fourier_process(save_path, out_dir)
        payload = {
            "psnr_blur": result["psnr_blur"],
            "psnr_restore": result["psnr_restore"],
            "blur_path": str(result["blur_path"]),
            "restore_path": str(result["restore_path"]),
            "montage_path": str(result["montage_path"]),
            "blur_image": _to_data_url(cv2.imread(result["blur_path"])),
            "restore_image": _to_data_url(cv2.imread(result["restore_path"])),
            "montage_image": _to_data_url(cv2.imread(result["montage_path"])),
        }
        resp = jsonify(payload)
    except Exception as e:
        resp = jsonify({"error": str(e)})
        resp.status_code = 500

    # Allow cross-origin for the frontend domain
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp


# ---------------- Module 2: Template Matching -----------------
@app.route("/api/template-matching", methods=["POST"])
def api_template_matching():
    """Run template matching on the uploaded scene image."""
    if "scene" not in request.files:
        return jsonify({"error": "Missing scene file upload."}), 400

    file = request.files["scene"]
    threshold = float(request.form.get("threshold", "0.7"))

    base_tmp = Path(os.getenv("MODULE2_BASE_DIR", "/tmp/module2"))
    uploads_dir = base_tmp / "uploads"
    output_dir = base_tmp / "output"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = uploads_dir / file.filename
    file.save(save_path)

    # Ensure module2 uses tmp output
    os.environ["MODULE2_OUTPUT_DIR"] = str(output_dir)
    from module2 import run_template_matching as tm
    tm.OUTPUT_DIR = output_dir
    tm.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = tm.run_detection(save_path, threshold=threshold)
        annotated_path = Path(result["annotated_image"])
        if not annotated_path.is_absolute():
            annotated_path = tm.OUTPUT_DIR / annotated_path.name
        annotated_img = cv2.imread(str(annotated_path))
        annotated_data_url = _to_data_url(annotated_img) if annotated_img is not None else None
        payload = {
            "detections": result.get("detections", []),
            "threshold": result.get("threshold", threshold),
            "annotatedImage": annotated_data_url,
            "num_detections": result.get("num_detections", 0),
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Module 3: Pipelines -----------------
@app.route("/api/assignment3", methods=["POST"])
def api_assignment3():
    """Handle gradients, keypoints, boundary, aruco, compare."""
    form = request.form
    operation = form.get("operation")
    if not operation:
        return jsonify({"error": "Missing operation"}), 400

    base_tmp = Path(os.getenv("MODULE3_BASE_DIR", "/tmp/module3"))
    uploads_dir = base_tmp / "uploads"
    output_dir = base_tmp / "output"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import pipelines and override paths to tmp
    import module3.pipelines as p
    p.UPLOAD_DIR = uploads_dir
    p.OUTPUT_DIR = output_dir

    if operation in ("gradients", "keypoints", "boundary", "aruco"):
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "Image upload required"}), 400
        img = _bytes_to_image(file.read())

        if operation == "gradients":
            res = p.compute_gradients(img)
            return jsonify(res)
        if operation == "keypoints":
            mode = form.get("mode", "edge")
            res = p.detect_keypoints(img, mode)
            return jsonify(res)
        if operation == "boundary":
            res = p.segment_boundary(img)
            return jsonify(res)
        if operation == "aruco":
            dictionary = form.get("dictionary", "DICT_5X5_100")
            res = p.aruco_segment(img, dictionary)
            return jsonify(res)

    if operation == "compare":
        ref = request.files.get("reference")
        cand = request.files.get("candidate")
        if not ref or not cand:
            return jsonify({"error": "Reference and candidate masks required"}), 400
        ref_img = cv2.imdecode(np.frombuffer(ref.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        cand_img = cv2.imdecode(np.frombuffer(cand.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        scores = p.compare_masks(ref_img, cand_img)
        return jsonify(scores)

    return jsonify({"error": f"Unsupported operation {operation}"}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)


