"""Upload a surface image and get a defect decision with latency."""

from __future__ import annotations

import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import features, load_model  # noqa: E402

app = Flask(__name__)
MODEL = load_model(ROOT / "models" / "defect_rf.joblib")


def classify(image: np.ndarray) -> dict:
    start = time.perf_counter()
    label = int(MODEL.predict(features(image).reshape(1, -1))[0])
    probability = MODEL.predict_proba(features(image).reshape(1, -1))[0, label]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "label": "defect" if label else "ok",
        "confidence": float(probability),
        "latency_ms": elapsed_ms,
    }


@app.get("/")
def home():
    return render_template("index.html", result=None)


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return render_template("index.html", result={"error": "فایلی انتخاب نشده است."}), 400
    image = np.asarray(Image.open(BytesIO(upload.read())).convert("L").resize((64, 64)))
    return render_template("index.html", result=classify(image))


@app.post("/api/predict")
def api_predict():
    upload = request.files.get("image")
    if upload is None:
        return {"error": "image file is required"}, 400
    image = np.asarray(Image.open(BytesIO(upload.read())).convert("L").resize((64, 64)))
    return classify(image)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5061, debug=False)
