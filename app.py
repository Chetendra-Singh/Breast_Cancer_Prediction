from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ── Load model and scaler ──────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── Feature order must match training data ─────────────────────────────────────
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",

    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",

    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with all 30 feature values.
    Returns: { prediction: "Benign" | "Malignant", confidence: 0.0–1.0 }
    """
    try:
        data = request.get_json(force=True)

        # Validate all features are present
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Build feature vector in correct order
        features = np.array([[float(data[f]) for f in FEATURE_NAMES]])

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        pred_class = model.predict(features_scaled)[0]          # 0 = Benign, 1 = Malignant
        pred_proba = model.predict_proba(features_scaled)[0]    # [P(Benign), P(Malignant)]

        label = "Malignant" if pred_class == 1 else "Benign"
        confidence = float(pred_proba[pred_class])              # confidence in the predicted class

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "benign": round(float(pred_proba[0]), 4),
                "malignant": round(float(pred_proba[1]), 4),
            }
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)