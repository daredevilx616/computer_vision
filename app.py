from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import os
from pathlib import Path

from module2.fourier_deblur import process_image as fourier_process

app = Flask(__name__)

# Webcam capture
cap = cv2.VideoCapture(0)

# Calibration and measurement storage
CALIBRATION = {"focal_length_pixels": None, "units": "cm"}
LAST_MEASUREMENT = {}

def gen_frames():
    """Generator for webcam frames"""
    while True:
        success, frame = cap.read()
        if not success:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
        def to_data_url(p: str) -> str:
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            return f"data:image/png;base64,{b64}"

        payload = {
            "psnr_blur": result["psnr_blur"],
            "psnr_restore": result["psnr_restore"],
            "blur_path": str(result["blur_path"]),
            "restore_path": str(result["restore_path"]),
            "montage_path": str(result["montage_path"]),
            "blur_image": to_data_url(result["blur_path"]),
            "restore_image": to_data_url(result["restore_path"]),
            "montage_image": to_data_url(result["montage_path"]),
        }
        resp = jsonify(payload)
    except Exception as e:
        resp = jsonify({"error": str(e)})
        resp.status_code = 500

    # Allow cross-origin for the frontend domain
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)


