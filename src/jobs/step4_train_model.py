#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

from src.core.azure_storage import AzureStorage
from src.core.data_build import snapshots_to_dataframe, build_supervised_dataset
from src.core.settings import Settings
from src.core.mlflow_utils import mlflow_setup


# 
# Split temps
# 
def time_split_last_days(meta: pd.DataFrame, last_days: int):
    """Time-based split: last `last_days` days are test, earlier are train."""
    max_date = meta["date"].max()
    cutoff = max_date - pd.Timedelta(days=last_days)
    is_test = meta["date"] > cutoff
    train_idx = np.where(~is_test.to_numpy())[0]
    test_idx = np.where(is_test.to_numpy())[0]
    return train_idx, test_idx


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> Dict:
    train_idx, test_idx = time_split_last_days(meta, Settings.ML_TEST_LAST_DAYS)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Baseline: prédire la moyenne du train
    baseline_pred = np.full_like(y_test.to_numpy(), fill_value=float(y_train.mean()), dtype=float)
    baseline_metrics = {
        "r2": float(r2_score(y_test, baseline_pred)),
        "mse": float(mean_squared_error(y_test, baseline_pred)),
        "mae": float(mean_absolute_error(y_test, baseline_pred)),
    }

    # Modèle
    n_estimators = int(os.getenv("RF_N_ESTIMATORS", getattr(Settings, "N_ESTIMATORS", 200)))
    max_depth = int(os.getenv("RF_MAX_DEPTH", getattr(Settings, "MAX_DEPTH", 20)))



    random_state = getattr(Settings, "RANDOM_STATE", None) or int(os.getenv("ML_RANDOM_STATE", "42"))

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=int(random_state),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mse": float(mean_squared_error(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }

    return {
        "model": model,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "split": {
            "test_last_days": int(Settings.ML_TEST_LAST_DAYS),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "max_date": meta["date"].max().strftime("%Y-%m-%d"),
        },
    }


# 
# MAIN
#
def main() -> None:
    storage = AzureStorage()

    # --- Load snapshots
    blob_names = storage.list_blobs(Settings.SNAPSHOT_PREFIX)
    if not blob_names:
        raise RuntimeError("Aucun snapshot trouvé sur Azure (daily_snapshots/...)")

    snapshots = []
    for name in blob_names:
        snap = storage.download_json(name)
        if snap and "repos" in snap:
            snapshots.append(snap)

    min_needed = Settings.ML_WINDOW_DAYS + Settings.ML_HORIZON_DAYS + 1
    if len(snapshots) < min_needed:
        raise RuntimeError(
            f"Pas assez de snapshots ({len(snapshots)}) pour window={Settings.ML_WINDOW_DAYS} "
            f"et horizon={Settings.ML_HORIZON_DAYS} (min={min_needed})"
        )

    df = snapshots_to_dataframe(snapshots)

    # --- Build dataset
    X, y, feature_cols, meta = build_supervised_dataset(
        df, window_days=Settings.ML_WINDOW_DAYS, horizon_days=Settings.ML_HORIZON_DAYS
    )

    if len(X) < 1000:
        print(f"Dataset ML assez petit: {len(X)} lignes.")

    mlflow_setup()

    with mlflow.start_run(run_name="train_model"):
        # Params
        mlflow.log_param("task", "regression")
        mlflow.log_param("target", f"delta_stars_{Settings.ML_HORIZON_DAYS}d")
        mlflow.log_param("window_days", int(Settings.ML_WINDOW_DAYS))
        mlflow.log_param("horizon_days", int(Settings.ML_HORIZON_DAYS))
        mlflow.log_param("test_last_days", int(Settings.ML_TEST_LAST_DAYS))
        mlflow.log_param("n_features", int(len(feature_cols)))

        n_estimators = int(os.getenv("RF_N_ESTIMATORS", getattr(Settings, "N_ESTIMATORS", 200)))
        max_depth = int(os.getenv("RF_MAX_DEPTH", getattr(Settings, "MAX_DEPTH", 20)))

        random_state = getattr(Settings, "RANDOM_STATE", None) or int(os.getenv("ML_RANDOM_STATE", "42"))

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", int(n_estimators))
        mlflow.log_param("max_depth", int(max_depth))
        mlflow.log_param("random_state", int(random_state))

        # Train + Eval
        out = train_and_evaluate(X, y, meta)
        model = out["model"]

        # Metrics
        mlflow.log_metric("r2", float(out["metrics"]["r2"]))
        mlflow.log_metric("mse", float(out["metrics"]["mse"]))
        mlflow.log_metric("mae", float(out["metrics"]["mae"]))

        mlflow.log_metric("baseline_r2", float(out["baseline_metrics"]["r2"]))
        mlflow.log_metric("baseline_mse", float(out["baseline_metrics"]["mse"]))
        mlflow.log_metric("baseline_mae", float(out["baseline_metrics"]["mae"]))

        # Tags utiles
        mlflow.set_tag("dataset", "github_snapshots")
        mlflow.set_tag("pipeline_step", "step4_train_model")

        # Save local model
        local_model_path = "stars_delta_7d_model.joblib"
        dump(model, local_model_path)

        # Upload model to Azure
        with open(local_model_path, "rb") as f:
            storage.upload_bytes(Settings.MODEL_BLOB, f.read(), "application/octet-stream")

        # Metadata (same as you print)
        metadata = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "task": "regression",
            "target": f"delta_stars_{Settings.ML_HORIZON_DAYS}d",
            "window_days": Settings.ML_WINDOW_DAYS,
            "horizon_days": Settings.ML_HORIZON_DAYS,
            "features": feature_cols,
            "metrics": out["metrics"],
            "baseline_metrics": out["baseline_metrics"],
            "split": out["split"],
        }

        # Upload metadata to Azure
        storage.upload_json(Settings.META_BLOB, metadata)

        # Log metadata as MLflow artifact
        meta_path = "train_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(meta_path, artifact_path="metadata")

        # Log model in MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("MLflow run id:", mlflow.active_run().info.run_id)

    print("Model trained & uploaded")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
