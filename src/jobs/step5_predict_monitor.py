#!/usr/bin/env python3
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import mlflow
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from src.core.azure_storage import AzureStorage
from src.core.settings import Settings
from src.core.mlflow_utils import mlflow_setup
from src.core.data_build import (
    snapshots_to_dataframe,
    build_features_for_prediction,
    build_supervised_dataset,
)
def get_truth_delta_stars(df: pd.DataFrame, start_date: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    """Compute y_true = stars(t+h) - stars(t) for repos with both dates available."""
    end_date = start_date + pd.Timedelta(days=horizon_days)
    d0 = df[df["date"] == start_date][["repo_id", "stars"]].rename(columns={"stars": "stars0"})
    d1 = df[df["date"] == end_date][["repo_id", "stars"]].rename(columns={"stars": "stars1"})
    merged = d0.merge(d1, on="repo_id", how="inner")
    merged["y_true"] = merged["stars1"] - merged["stars0"]
    return merged[["repo_id", "y_true"]]


def retrain_model_and_upload(storage: AzureStorage, df_all: pd.DataFrame) -> Dict:
    """
    Optional retrain using src.core.data_build.build_supervised_dataset.
    Keeps the logic aligned with Step4.
    """

    X, y, feature_cols, meta = build_supervised_dataset(
        df_all,
        window_days=Settings.ML_WINDOW_DAYS,
        horizon_days=Settings.ML_HORIZON_DAYS,
    )

    # time split identical to step4 style (last days)
    max_date = meta["date"].max()
    cutoff = max_date - pd.Timedelta(days=Settings.ML_TEST_LAST_DAYS)
    is_test = meta["date"] > cutoff
    train_idx = np.where(~is_test.to_numpy())[0]
    test_idx = np.where(is_test.to_numpy())[0]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    n_estimators = int(os.getenv("RF_N_ESTIMATORS", getattr(Settings, "N_ESTIMATORS", 200)))
    max_depth = int(os.getenv("RF_MAX_DEPTH", getattr(Settings, "MAX_DEPTH", 20)))
    random_state = int(os.getenv("ML_RANDOM_STATE", getattr(Settings, "RANDOM_STATE", 42)))

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, pred))
    rmse = float(np.sqrt(mse))

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mse": mse,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_test, pred)),
    }


    # upload model
    local_path = "stars_delta_7d_model.joblib"
    dump(model, local_path)
    with open(local_path, "rb") as f:
        storage.upload_bytes(Settings.MODEL_BLOB, f.read(), "application/octet-stream")

    # update meta
    metadata = storage.download_json(Settings.META_BLOB) or {}
    metadata.update({
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "task": "regression",
        "target": f"delta_stars_{Settings.ML_HORIZON_DAYS}d",
        "window_days": Settings.ML_WINDOW_DAYS,
        "horizon_days": Settings.ML_HORIZON_DAYS,
        "features": feature_cols,
        "metrics": metrics,
    })
    storage.upload_json(Settings.META_BLOB, metadata)

    return metrics



# =============================================================================
# DATA HELPERS (same logic as step4)
# =============================================================================

def main():
    mlflow_setup()
    storage = AzureStorage()

    # -------------------------
    # Load snapshots
    # -------------------------
    snapshot_blobs = storage.list_blobs(Settings.SNAPSHOT_PREFIX)
    if not snapshot_blobs:
        raise RuntimeError("Aucun snapshot trouvé sur Azure")

    snapshots: List[Dict] = []
    for b in snapshot_blobs:
        s = storage.download_json(b)
        if s and "repos" in s:
            snapshots.append(s)

    df_all = snapshots_to_dataframe(snapshots)
    df_all["date"] = pd.to_datetime(df_all["date"])
    last_date = df_all["date"].max()
    last_date_str = last_date.strftime("%Y-%m-%d")

    # -------------------------
    # Load model from Azure
    # -------------------------
    model_bytes = storage.download_bytes(Settings.MODEL_BLOB)
    if not model_bytes:
        raise RuntimeError(f"Modèle introuvable sur Azure: {Settings.MODEL_BLOB}")

    local_model_path = "stars_delta_7d_model.joblib"
    with open(local_model_path, "wb") as f:
        f.write(model_bytes)

    model = load(local_model_path)

    # -------------------------
    # Build prediction features (from src.core.data_build)
    # -------------------------
    X_pred, meta_pred, feature_cols = build_features_for_prediction(
        df_all,
        window_days=Settings.ML_WINDOW_DAYS,
        as_of_date=last_date,
    )

    if len(X_pred) == 0:
        raise RuntimeError("Aucune feature disponible pour la date courante (pas assez d'historique window_days ?)")

    # -------------------------
    # Predict
    # -------------------------
    # -------------------------
    # Load classifier (optional)
    # -------------------------
    clf_blob = getattr(Settings, "CLASSIFIER_MODEL_BLOB", "models/stars_growth_classifier.joblib")
    clf_bytes = storage.download_bytes(clf_blob)

    clf = None
    if clf_bytes:
        local_clf_path = "stars_growth_classifier.joblib"
        with open(local_clf_path, "wb") as f:
            f.write(clf_bytes)
        clf = load(local_clf_path)

    y_pred = model.predict(X_pred)

    pred_class = None
    pred_proba_explosion = None

    if clf is not None:
        pred_class = clf.predict(X_pred)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_pred)
            # classe 2 = explosion -> colonne index 2 si classes=[0,1,2]
            if proba.shape[1] >= 3:
                pred_proba_explosion = proba[:, 2]


    pred_blob = f"{Settings.PRED_PREFIX}{last_date_str}.json"
    pred_payload = {
        "as_of_date": last_date_str,
        "horizon_days": Settings.ML_HORIZON_DAYS,
        "window_days": Settings.ML_WINDOW_DAYS,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_predictions": int(len(meta_pred)),
        "feature_cols": feature_cols,
        "predictions": [
            {
                "repo_id": int(rid),
                "full_name": str(fname),
                "stars_t": float(stars_t),
                "pred_delta_stars_h": float(pred),
                "pred_stars_t_plus_h": float(stars_t + pred),
                "pred_class": int(c) if pred_class is not None else None,
                "pred_proba_explosion": float(p) if pred_proba_explosion is not None else None,

            }
            
            for rid, fname, stars_t, pred, c, p in zip(
                meta_pred["repo_id"].to_numpy(),
                meta_pred.get("full_name", pd.Series([""] * len(meta_pred))).fillna("").to_numpy(),
                meta_pred["stars_t"].to_numpy(),
                y_pred,
                pred_class if pred_class is not None else [None] * len(meta_pred),
                pred_proba_explosion if pred_proba_explosion is not None else [None] * len(meta_pred),
            )

        ],
    }

    storage.upload_json(pred_blob, pred_payload)
    print(f"Predictions uploaded: {pred_blob} ({len(meta_pred)} repos)")

    # -------------------------
    # MLflow logging
    # -------------------------
    with mlflow.start_run(run_name="predict"):
        mlflow.log_param("window_days", int(Settings.ML_WINDOW_DAYS))
        mlflow.log_param("horizon_days", int(Settings.ML_HORIZON_DAYS))
        mlflow.log_param("n_repos_predicted", int(len(meta_pred)))
        mlflow.log_param("prediction_date", last_date_str)
        mlflow.log_param("n_features", int(len(feature_cols)))
        mlflow.set_tag("pipeline_step", "step5_predict_monitor")
        mlflow.set_tag("prediction_blob", pred_blob)

    # -------------------------
    # Monitoring: evaluate older predictions that now have truth
    # -------------------------
    pred_blobs = storage.list_blobs("predictions/pred_")

    history = storage.download_json(Settings.METRICS_HISTORY_BLOB) or {"history": []}
    already_evaluated = {item.get("pred_as_of_date") for item in history.get("history", []) if item.get("pred_as_of_date")}

    for pb in pred_blobs:
        fname = pb.split("/")[-1]
        date_str = fname.replace("pred_", "").replace(".json", "")

        try:
            as_of = pd.to_datetime(date_str)
        except Exception:
            continue

        if last_date < as_of + pd.Timedelta(days=Settings.ML_HORIZON_DAYS):
            continue

        if date_str in already_evaluated:
            continue

        pred_data = storage.download_json(pb)
        if not pred_data:
            continue

        pred_df = pd.DataFrame(pred_data.get("predictions", []))
        if pred_df.empty:
            continue

        truth_df = get_truth_delta_stars(df_all, start_date=as_of, horizon_days=Settings.ML_HORIZON_DAYS)
        if truth_df.empty:
            continue

        merged = pred_df.merge(truth_df, on="repo_id", how="inner")
        if merged.empty:
            continue

        y_true = merged["y_true"].astype(float).to_numpy()
        y_hat = merged["pred_delta_stars_h"].astype(float).to_numpy()

        mse = float(mean_squared_error(y_true, y_hat))
        rmse = float(np.sqrt(mse))

        metrics = {
            "pred_as_of_date": date_str,
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "n_eval": int(len(merged)),
            "r2": float(r2_score(y_true, y_hat)),
            "mse": mse,
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_true, y_hat)),
        }


        history["history"].append(metrics)
        print(f"Evaluated {date_str}: R2={metrics['r2']:.3f} n={metrics['n_eval']}")

        r2_threshold = float(getattr(Settings, "R2_THRESHOLD", 0.10))
        retrain_on_trigger = bool(getattr(Settings, "RETRAIN_ON_TRIGGER", False))

        if metrics["r2"] < r2_threshold:
            request = {
                "triggered_at": datetime.utcnow().isoformat() + "Z",
                "reason": "r2_below_threshold",
                "threshold": r2_threshold,
                "observed_r2": metrics["r2"],
                "pred_as_of_date": date_str,
                "action": "retrain_suggested" if not retrain_on_trigger else "retrain_started",
            }
            storage.upload_json(Settings.RETRAIN_REQUEST_BLOB, request)
            print(f"Retrain trigger: R2<{r2_threshold}. Wrote {Settings.RETRAIN_REQUEST_BLOB}")

            if retrain_on_trigger:
                new_metrics = retrain_model_and_upload(storage, df_all)
                request["action"] = "retrain_done"
                request["new_model_metrics"] = new_metrics
                storage.upload_json(Settings.RETRAIN_REQUEST_BLOB, request)
                print(f"Retrain done. New model R2={new_metrics['r2']:.3f}")

    storage.upload_json(Settings.METRICS_HISTORY_BLOB, history)
    print(f"Monitoring history updated: {Settings.METRICS_HISTORY_BLOB}")



if __name__ == "__main__":
    main()