import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ---- Load model artifacts at startup ----
model = None
scaler = None
scale_cols = None
feature_order = None
metrics = None

def load_artifacts():
    global model, scaler, scale_cols, feature_order, metrics
    model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    scale_cols_path = os.path.join(MODEL_DIR, "scale_cols.pkl")
    feature_order_path = os.path.join(MODEL_DIR, "feature_order.json")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        scale_cols = joblib.load(scale_cols_path)
        with open(feature_order_path) as f:
            feature_order = json.load(f)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

load_artifacts()


@app.route("/", methods=["GET"])
def home():
    model_ready = model is not None
    return render_template("index.html", model_ready=model_ready, feature_order=feature_order)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            model_ready=False,
            error="Model not trained yet. Run `python train_model.py` first.",
        )

    try:
        # Build a single-row DataFrame from the submitted form, in the exact
        # column order the model was trained on.
        row = {}
        for col in feature_order:
            val = request.form.get(col, "0")
            row[col] = float(val) if val not in (None, "") else 0.0

        input_df = pd.DataFrame([row], columns=feature_order)
        input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = {
            "label": "Fraudulent" if prediction == 1 else "Genuine",
            "is_fraud": bool(prediction == 1),
            "probability": round(float(probability) * 100, 2),
        }
        return render_template(
            "index.html", model_ready=True, feature_order=feature_order,
            result=result, submitted_values=row,
        )
    except Exception as exc:
        return render_template(
            "index.html", model_ready=True, feature_order=feature_order,
            error=f"Could not process input: {exc}",
        )


@app.route("/metrics", methods=["GET"])
def metrics_page():
    return render_template("metrics.html", metrics=metrics)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
