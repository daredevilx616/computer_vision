from flask import Flask, render_template, Response, request, jsonify
import cv2

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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)


