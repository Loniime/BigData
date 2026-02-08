#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)

import mlflow
import mlflow.sklearn

from src.core.azure_storage import AzureStorage
from src.core.data_build import snapshots_to_dataframe, build_supervised_dataset
from src.core.settings import Settings
from src.core.mlflow_utils import mlflow_setup


def time_split_last_days(meta: pd.DataFrame, last_days: int):
    max_date = meta["date"].max()
    cutoff = max_date - pd.Timedelta(days=last_days)
    is_test = meta["date"] > cutoff
    train_idx = np.where(~is_test.to_numpy())[0]
    test_idx = np.where(is_test.to_numpy())[0]
    return train_idx, test_idx


def build_classes_from_relative_gain(y_delta: pd.Series, stars_t: pd.Series, q_low=0.60, q_high=0.90):
    """
    y_rel = y_delta / (stars_t + 1)
    classes:
      0 = stable     (<= q_low)
      1 = faible     (q_low, q_high]
      2 = explosion  (> q_high)
    """
    stars_safe = stars_t.fillna(0).astype(float)
    y_rel = y_delta.astype(float) / (stars_safe + 1.0)

    th_low = float(np.nanquantile(y_rel, q_low))
    th_high = float(np.nanquantile(y_rel, q_high))

    y_class = np.zeros(len(y_rel), dtype=int)
    y_class[(y_rel > th_low) & (y_rel <= th_high)] = 1
    y_class[y_rel > th_high] = 2

    thresholds = {
        "type": "relative_gain_quantiles",
        "q_low": q_low,
        "q_high": q_high,
        "th_low": th_low,
        "th_high": th_high,
    }
    return y_rel, y_class, thresholds


def train_and_evaluate_classifier(X: pd.DataFrame, y_class: np.ndarray, meta: pd.DataFrame) -> Dict:
    train_idx, test_idx = time_split_last_days(meta, Settings.ML_TEST_LAST_DAYS)
    X_train, y_train = X.iloc[train_idx], y_class[train_idx]
    X_test, y_test = X.iloc[test_idx], y_class[test_idx]

    n_estimators = int(os.getenv("CLF_N_ESTIMATORS", "300"))
    max_depth = int(os.getenv("CLF_MAX_DEPTH", "20"))
    random_state = int(os.getenv("ML_RANDOM_STATE", "42"))

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",  
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "recall_explosion": float(recall_score(y_test, y_pred, labels=[2], average=None)[0]),
        "precision_explosion": float(precision_score(y_test, y_pred, labels=[2], average=None, zero_division=0)[0]),
    }

    return {"model": clf, "metrics": metrics, "split": {"train_rows": int(len(train_idx)), "test_rows": int(len(test_idx))}}


def main() -> None:
    storage = AzureStorage()

    blob_names = storage.list_blobs(Settings.SNAPSHOT_PREFIX)
    if not blob_names:
        raise RuntimeError("Aucun snapshot trouvé sur Azure (daily_snapshots/...)")

    snapshots = []
    for name in blob_names:
        snap = storage.download_json(name)
        if snap and "repos" in snap:
            snapshots.append(snap)

    df = snapshots_to_dataframe(snapshots)

    # Dataset regression (y = delta stars H days)
    X, y_delta, feature_cols, meta = build_supervised_dataset(
        df, window_days=Settings.ML_WINDOW_DAYS, horizon_days=Settings.ML_HORIZON_DAYS
    )
    if "stars_t" not in meta.columns:
        raise RuntimeError("meta ne contient pas 'stars_t'. Ajoute stars_t dans build_supervised_dataset(meta).")

    # Build classes
    y_rel, y_class, thresholds = build_classes_from_relative_gain(
        y_delta=y_delta,
        stars_t=meta["stars_t"],
        q_low=float(os.getenv("CLF_Q_LOW", "0.60")),
        q_high=float(os.getenv("CLF_Q_HIGH", "0.90")),
    )

    mlflow_setup()

    with mlflow.start_run(run_name="train_classifier"):
        mlflow.log_param("task", "classification")
        mlflow.log_param("target", "stable/faible/explosion")
        mlflow.log_param("window_days", int(Settings.ML_WINDOW_DAYS))
        mlflow.log_param("horizon_days", int(Settings.ML_HORIZON_DAYS))
        mlflow.log_param("test_last_days", int(Settings.ML_TEST_LAST_DAYS))
        mlflow.log_param("n_features", int(len(feature_cols)))

        mlflow.log_param("threshold_q_low", thresholds["q_low"])
        mlflow.log_param("threshold_q_high", thresholds["q_high"])
        mlflow.log_param("threshold_low", thresholds["th_low"])
        mlflow.log_param("threshold_high", thresholds["th_high"])

        out = train_and_evaluate_classifier(X, y_class, meta)
        clf = out["model"]

        for k, v in out["metrics"].items():
            mlflow.log_metric(k, float(v))

        mlflow.set_tag("dataset", "github_snapshots")
        mlflow.set_tag("pipeline_step", "step4b_train_classifier")

        # Save model
        local_path = "stars_growth_classifier.joblib"
        dump(clf, local_path)

        # 🔧 à définir dans Settings (ou utiliser env)
        clf_blob = getattr(Settings, "CLASSIFIER_MODEL_BLOB", "models/stars_growth_classifier.joblib")
        thresholds_blob = getattr(Settings, "CLASSIFIER_THRESHOLDS_BLOB", "models/classification_thresholds.json")
        meta_blob = getattr(Settings, "CLASSIFIER_META_BLOB", "models/classifier_meta.json")

        with open(local_path, "rb") as f:
            storage.upload_bytes(clf_blob, f.read(), "application/octet-stream")

        # Save thresholds + meta
        classifier_meta = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "task": "classification",
            "classes": {0: "stable", 1: "faible", 2: "explosion"},
            "window_days": Settings.ML_WINDOW_DAYS,
            "horizon_days": Settings.ML_HORIZON_DAYS,
            "features": feature_cols,
            "thresholds": thresholds,
            "metrics": out["metrics"],
            "split": out["split"],
        }

        storage.upload_json(thresholds_blob, thresholds)
        storage.upload_json(meta_blob, classifier_meta)

        with open("classifier_meta.json", "w", encoding="utf-8") as f:
            json.dump(classifier_meta, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact("classifier_meta.json", artifact_path="metadata")

        mlflow.sklearn.log_model(clf, artifact_path="model")

    print("Classifier trained & uploaded")
    print(json.dumps(classifier_meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
