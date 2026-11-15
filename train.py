import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core import ARTIFACTS, FEATURES, NAMES, TARGET, find_data, load_data, prepare, scores, split_data


def build_models(y):
    ratio = (y == 0).sum() / (y == 1).sum()
    workers = min(os.cpu_count() or 1, 6)
    return {
        "cart": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced",
                                       random_state=42),
        "forest": RandomForestClassifier(n_estimators=64, max_depth=15, min_samples_leaf=2,
                                         max_samples=0.5, class_weight="balanced_subsample",
                                         n_jobs=workers, random_state=42),
        "xgb": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.85,
                             colsample_bytree=0.85, scale_pos_weight=float(ratio), tree_method="hist",
                             eval_metric="logloss", n_jobs=workers, random_state=42),
    }


def validation_threshold(y, risk):
    precision, recall, thresholds = precision_recall_curve(y, risk)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[np.argmax(f1)])


def evaluate(y, risk, threshold):
    predictions = risk >= threshold
    precision, recall, _ = precision_recall_curve(y, risk, drop_intermediate=True)
    # Keep the compact curve for display. AP is computed from every test score.
    points = np.unique(np.linspace(0, len(precision) - 1, min(240, len(precision)), dtype=int))
    return {
        "accuracy": accuracy_score(y, predictions), "ap": average_precision_score(y, risk),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "pr": [[float(recall[i]), float(precision[i])] for i in points],
    }


def examples(test, risk, threshold):
    fraud = test[TARGET].to_numpy() == 1
    flagged = risk >= threshold
    cases = []
    groups = [
        ("Flagged fraud", fraud & flagged, True), ("Cleared legitimate", ~fraud & ~flagged, False),
        ("False alert", ~fraud & flagged, True), ("Missed fraud", fraud & ~flagged, False),
    ]
    for label, mask, high in groups:
        positions = np.flatnonzero(mask)
        if len(positions):
            ranked = positions[np.argsort(risk[positions], kind="stable")]
            index = int(ranked[-1] if high else ranked[0])
            row = test.iloc[index]
            cases.append({"label": label, "truth": int(row[TARGET]),
                          "values": {key: float(row[key]) for key in FEATURES}})
    return cases


def describe_partition(frame):
    return {"rows": len(frame), "fraud": int(frame[TARGET].sum()),
            "start": float(frame.Time.min()), "end": float(frame.Time.max())}


def run(path):
    data = load_data(path)
    clean, audit = prepare(data)
    parts = split_data(clean)
    train, validation, test = (parts[name] for name in ("train", "validation", "test"))
    models = build_models(train[TARGET])
    results = {}
    test_scores = {}
    for key, model in models.items():
        print(f"Training {NAMES[key]}...", flush=True)
        model.fit(train[FEATURES], train[TARGET])
        val_scores = scores(model, validation)
        threshold = validation_threshold(validation[TARGET], val_scores)
        test_scores[key] = scores(model, test)
        results[key] = {
            "name": NAMES[key], "threshold": threshold,
            "validation_ap": average_precision_score(validation[TARGET], val_scores),
            **evaluate(test[TARGET], test_scores[key], threshold),
            "importance": sorted(
                [{"name": name, "value": float(value)} for name, value
                 in zip(FEATURES, model.feature_importances_)],
                key=lambda item: -item["value"],
            )[:8],
        }
        print(f"  Validation AP {results[key]['validation_ap']:.3f} | Test AP {results[key]['ap']:.3f}", flush=True)
    recommended = max(results, key=lambda key: results[key]["validation_ap"])
    metadata = {
        "audit": audit, "partitions": {key: describe_partition(value) for key, value in parts.items()},
        "recommended": recommended, "models": results,
        "examples": examples(test, test_scores[recommended], results[recommended]["threshold"]),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(models, ARTIFACTS / "models.joblib", compress=3)
    (ARTIFACTS / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame({TARGET: test[TARGET].to_numpy(), **test_scores}).to_csv(
        ARTIFACTS / "test_scores.csv.gz", index=False
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ScamShield's three classifiers.")
    parser.add_argument("--data", help="Path to creditcard.csv or archive.zip")
    args = parser.parse_args()
    run(find_data(args.data))

