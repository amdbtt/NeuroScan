import os
import uuid
from flask import Flask, request, jsonify, render_template
from numpy_inference import NumPyVGG16Model, CLASS_LABELS

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BrainTumor_VGG16.h5")
if not os.path.exists(model_path):
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "BrainTumor_VGG16.keras",
        "model.weights.h5",
    )

print(f"Initializing VGG16 model using weights at: {model_path}")
try:
    tumor_model = NumPyVGG16Model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    tumor_model = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/labels", methods=["GET"])
def labels():
    return jsonify({"success": True, "labels": CLASS_LABELS})


@app.route("/predict", methods=["POST"])
def predict():
    if tumor_model is None:
        return jsonify({
            "success": False,
            "error": "Model failed to initialize on the server",
        }), 500

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
        return jsonify({
            "success": False,
            "error": "Unsupported file type. Use JPEG, PNG, TIFF, BMP, or WebP.",
        }), 400

    unique_filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    try:
        file.save(filepath)
        result = tumor_model.predict_detailed(filepath)

        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing temp file: {e}")

        return jsonify({
            "success": True,
            "predictions": result["probabilities"],
            "labels": result["labels"],
            "class_probabilities": result["class_probabilities"],
            "predicted_index": result["predicted_index"],
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "inference_time_ms": result["inference_time_ms"],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}",
        }), 500


if __name__ == "__main__":
    print("Starting Brain Tumor Detection Diagnostic Server on http://127.0.0.1:5001 ...")
    app.run(host="127.0.0.1", port=5001, debug=True)
