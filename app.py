from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import os
from pathlib import Path
import numpy as np

from module2.fourier_deblur import process_image as fourier_process
from module2 import run_template_matching as tm

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


def _side_by_side(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Place two images side by side on a black canvas."""
    h = max(img_a.shape[0], img_b.shape[0])
    w = img_a.shape[1] + img_b.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: img_a.shape[0], : img_a.shape[1]] = img_a
    canvas[: img_b.shape[0], img_a.shape[1] :] = img_b
    return canvas

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
            "psnrBlur": result["psnr_blur"],
            "psnrRestore": result["psnr_restore"],
            "blurPath": str(result["blur_path"]),
            "restorePath": str(result["restore_path"]),
            "montagePath": str(result["montage_path"]),
            "blurImage": _to_data_url(cv2.imread(result["blur_path"])),
            "restoredImage": _to_data_url(cv2.imread(result["restore_path"])),
            "montageImage": _to_data_url(cv2.imread(result["montage_path"])),
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

    try:
        # Ensure module2 uses tmp output
        os.environ["MODULE2_OUTPUT_DIR"] = str(output_dir)
        tm.OUTPUT_DIR = output_dir
        tm.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


# ---------------- Module 4: SIFT & Stitching -----------------
@app.route("/api/assignment4/sift", methods=["POST"])
def api_sift():
    file_a = request.files.get("imageA")
    file_b = request.files.get("imageB")
    if not file_a or not file_b:
        return jsonify({"error": "Two image uploads are required."}), 400

    # Use module4 directory in project root (works on Windows and Linux)
    base_dir = Path(__file__).resolve().parent / "module4"
    uploads_dir = base_dir / "uploads"
    output_dir = base_dir / "output"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_a = uploads_dir / file_a.filename
    path_b = uploads_dir / file_b.filename
    file_a.save(path_a)
    file_b.save(path_b)

    import module4.sift as msift
    import module4.stitcher as mstitch

    msift.OUTPUT_DIR = output_dir
    mstitch.OUTPUT_DIR = output_dir

    try:
        img_a = cv2.imread(str(path_a))
        img_b = cv2.imread(str(path_b))
        res_a = msift.sift(img_a)
        res_b = msift.sift(img_b)
        matches = msift.match_descriptors(res_a.descriptors, res_b.descriptors)
        H, inliers = msift.ransac_homography(matches, res_a.keypoints, res_b.keypoints)
        visual = msift.draw_matches(img_a, img_b, res_a.keypoints, res_b.keypoints, matches, inliers)
        visual_path = output_dir / "sift_matches.png"
        cv2.imwrite(str(visual_path), visual)

        # OpenCV SIFT + BF matcher for comparison (robust even if no good matches)
        cv_match_count = 0
        cv_visual = _side_by_side(img_a, img_b)
        cv_error = None
        try:
            sift_cv = cv2.SIFT_create()
            kpa, des_a_cv = sift_cv.detectAndCompute(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), None)
            kpb, des_b_cv = sift_cv.detectAndCompute(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), None)

            print(f"[OpenCV SIFT] Detected keypoints: A={len(kpa)}, B={len(kpb)}")
            print(f"[OpenCV SIFT] Descriptors: A={des_a_cv.shape if des_a_cv is not None else None}, B={des_b_cv.shape if des_b_cv is not None else None}")

            if des_a_cv is not None and des_b_cv is not None and len(des_a_cv) > 0 and len(des_b_cv) > 0:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                raw_matches = bf.knnMatch(des_a_cv, des_b_cv, k=2)
                print(f"[OpenCV SIFT] Raw matches: {len(raw_matches)}")

                good = []
                for m_n in raw_matches:
                    if len(m_n) >= 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good.append(m)

                cv_match_count = len(good)
                print(f"[OpenCV SIFT] Good matches after ratio test: {cv_match_count}")

                if good:
                    cv_visual = cv2.drawMatches(
                        img_a, kpa, img_b, kpb, good, None,
                        matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0),
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    print("[OpenCV SIFT] Successfully drew matches")
                else:
                    print("[OpenCV SIFT] No good matches found after ratio test")
            else:
                print("[OpenCV SIFT] No descriptors computed")

        except Exception as e:
            cv_error = str(e)
            print(f"[OpenCV SIFT] Error: {cv_error}")
            import traceback
            traceback.print_exc()
            cv_match_count = 0
            cv_visual = _side_by_side(img_a, img_b)

        payload = {
            "match_count": len(matches),
            "inliers": len(inliers),
            "homography": H.tolist(),
            "visual": _to_data_url(visual),
            "cv_match_count": cv_match_count,
            "cv_visual": _to_data_url(cv_visual),
            "cv_error": cv_error,
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assignment4/stitch", methods=["POST"])
def api_stitch():
    images = [f for f in request.files.getlist("images") if f.filename]
    if len(images) < 2:
        return jsonify({"error": "Upload at least two images for stitching."}), 400

    # Use module4 directory in project root (works on Windows and Linux)
    base_dir = Path(__file__).resolve().parent / "module4"
    uploads_dir = base_dir / "uploads"
    output_dir = base_dir / "output"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for idx, f in enumerate(images):
        path = uploads_dir / f"{idx}_{f.filename}"
        f.save(path)
        paths.append(path)

    import module4.stitcher as mstitch
    mstitch.OUTPUT_DIR = output_dir
    try:
        imgs = [cv2.imread(str(p)) for p in paths]
        result = mstitch.stitch_images(imgs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)


