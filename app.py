import json
from functools import lru_cache

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from core import ARTIFACTS, FEATURES, explain, scores

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024


@lru_cache(maxsize=1)
def load_artifacts():
    report = json.loads((ARTIFACTS / "evaluation.json").read_text(encoding="utf-8"))
    models = joblib.load(ARTIFACTS / "models.joblib")
    return report, models


@app.get("/")
def dashboard():
    report, _ = load_artifacts()
    return render_template("index.html", report=report, features=FEATURES)


@app.post("/api/score")
def score_transaction():
    report, models = load_artifacts()
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Provide a JSON object with a model and feature values."), 400
    model_id = body.get("model")
    raw = body.get("values")
    if not isinstance(model_id, str) or model_id not in models or not isinstance(raw, dict):
        return jsonify(error="Choose a supported model and supply the transaction fields."), 400
    if set(raw) != set(FEATURES):
        return jsonify(error="Enter all 30 transaction features."), 400
    values = {}
    for name in FEATURES:
        value = raw[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not -1e30 <= value <= 1e30:
            return jsonify(error=f"{name} must be a finite number with magnitude at most 1e30."), 400
        values[name] = float(value)
    if values["Amount"] < 0 or values["Time"] < 0:
        return jsonify(error="Amount and Time must be nonnegative."), 400
    row = pd.DataFrame([values], columns=FEATURES)
    model = models[model_id]
    risk = float(scores(model, row)[0])
    threshold = report["models"][model_id]["threshold"]
    return jsonify(score=risk, threshold=threshold, flagged=risk >= threshold,
                   explanation=explain(model_id, model, row))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

