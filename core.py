from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FEATURES = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
TARGET = "Class"
NAMES = {"cart": "CART", "forest": "Random Forest", "xgb": "XGBoost"}


def find_data(path=None):
    if path:
        return Path(path).expanduser()
    candidates = [
        ROOT / "data" / "creditcard.csv", ROOT / "creditcard.csv",
        ROOT / "data" / "archive.zip", ROOT / "archive.zip",
        Path.home() / "Downloads" / "archive.zip",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Put creditcard.csv or archive.zip in data/, or pass --data.")


def load_data(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == "creditcard.csv"]
            if len(matches) != 1:
                raise ValueError("The ZIP must contain exactly one creditcard.csv.")
            with archive.open(matches[0]) as source:
                data = pd.read_csv(source, usecols=FEATURES + [TARGET])
    else:
        data = pd.read_csv(path, usecols=FEATURES + [TARGET])
    if not np.isfinite(data.to_numpy()).all() or set(data[TARGET]) != {0, 1}:
        raise ValueError("Expected finite numerical features and both binary classes (0/1).")
    return data


def prepare(data):
    # Identical feature vectors stay in a single partition. Conflicting labels
    # cannot be resolved from the available fields, so remove their whole group.
    conflicts = data.groupby(FEATURES, sort=False)[TARGET].transform("nunique") > 1
    unambiguous = data.loc[~conflicts]
    clean = unambiguous.drop_duplicates(subset=FEATURES).sort_values("Time").reset_index(drop=True)
    audit = {
        "raw_rows": len(data), "raw_fraud": int(data[TARGET].sum()),
        "conflicting_rows": int(conflicts.sum()),
        "duplicate_rows": len(unambiguous) - len(clean),
        "clean_rows": len(clean), "clean_fraud": int(clean[TARGET].sum()),
    }
    return clean, audit


def split_data(clean):
    # Use observed timestamps as cutoffs, keeping all equal timestamps together.
    first, second = (clean["Time"].iloc[int(len(clean) * p)] for p in (0.6, 0.8))
    train = clean[clean["Time"] < first]
    validation = clean[(clean["Time"] >= first) & (clean["Time"] < second)]
    test = clean[clean["Time"] >= second]
    parts = {"train": train, "validation": validation, "test": test}
    if any(part[TARGET].nunique() != 2 for part in parts.values()):
        raise ValueError("Each chronological partition needs both classes.")
    return parts


def scores(model, frame):
    return model.predict_proba(frame[FEATURES])[:, 1]


def path_contributions(estimator, row):
    tree = estimator.tree_
    values = tree.value[:, 0, :]
    probabilities = values[:, 1] / values.sum(axis=1)
    base = probabilities[0]
    contributions = np.zeros(len(FEATURES))
    node = 0
    while tree.children_left[node] != tree.children_right[node]:
        feature = tree.feature[node]
        child = tree.children_left[node] if row[feature] <= tree.threshold[node] else tree.children_right[node]
        contributions[feature] += probabilities[child] - probabilities[node]
        node = child
    return base, contributions


def explain(model_id, model, frame):
    if model_id == "xgb":
        from xgboost import DMatrix

        values = model.get_booster().predict(DMatrix(frame[FEATURES]), pred_contribs=True)[0]
        base, contributions = values[-1], values[:-1]
        unit = "log-odds"
        method = "TreeSHAP"
    else:
        row = frame[FEATURES].to_numpy(dtype=np.float32)[0]
        estimators = model.estimators_ if model_id == "forest" else [model]
        paths = [path_contributions(tree, row) for tree in estimators]
        base = np.mean([path[0] for path in paths])
        contributions = np.mean([path[1] for path in paths], axis=0)
        unit = "score"
        method = "Tree path contributions"
    ranked = sorted(zip(FEATURES, contributions), key=lambda item: abs(item[1]), reverse=True)
    return {
        "base": float(base), "total": float(base + sum(contributions)), "unit": unit, "method": method,
        "features": [{"name": name, "value": float(value)} for name, value in ranked if value != 0],
    }

