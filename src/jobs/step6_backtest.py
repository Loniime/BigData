# src/jobs/step6_backtest.py
from __future__ import annotations

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.core.azure_storage import AzureStorage
from src.core.settings import Settings
from src.core.data_build import (
    snapshots_to_dataframe,
    build_supervised_dataset,
    build_features_for_prediction,
)

from src.core.mlflow_utils import mlflow_setup
import mlflow


WINDOW_DAYS = int(os.getenv("ML_WINDOW_DAYS", "7"))
HORIZON_DAYS = int(os.getenv("ML_HORIZON_DAYS", "7"))

RETRAIN_EVERY_DAYS = int(os.getenv("BACKTEST_RETRAIN_EVERY_DAYS", "7"))
R2_THRESHOLD = float(os.getenv("BACKTEST_R2_THRESHOLD", "0.10"))
ROLLING_EVAL_WINDOW = int(os.getenv("BACKTEST_ROLLING_EVAL_WINDOW", "5"))

BACKTEST_HISTORY_BLOB = os.getenv(
    "BACKTEST_HISTORY_BLOB", "monitoring/metrics_history_backtest.json"
)

RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "200"))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "20"))
ML_RANDOM_STATE = int(os.getenv("ML_RANDOM_STATE", "42"))


def _parse_date_from_blobname(name: str) -> str | None:
    base = name.split("/")[-1]
    if "snapshot_" not in base:
        return None
    part = base.replace("snapshot_", "").replace(".json", "")
    try:
        _ = pd.to_datetime(part).date()
        return part
    except Exception:
        return None


def _load_all_snapshots(storage: AzureStorage) -> List[Dict]:
    blob_names = storage.list_blobs(Settings.SNAPSHOT_PREFIX)
    snapshots: List[Dict] = []
    for name in blob_names:
        snap = storage.download_json(name)
        if snap and isinstance(snap, dict) and "repos" in snap:
            if "date" not in snap:
                d = _parse_date_from_blobname(name)
                if d:
                    snap["date"] = d
            snapshots.append(snap)
    return snapshots


def _train_model(df_upto_t: pd.DataFrame) -> Dict:
    X, y, feature_cols, _meta = build_supervised_dataset(
        df_upto_t, window_days=WINDOW_DAYS, horizon_days=HORIZON_DAYS
    )
    if len(X) < 5000:
        return {"model": None, "feature_cols": feature_cols, "train_rows": int(len(X))}

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=ML_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)
    return {"model": model, "feature_cols": feature_cols, "train_rows": int(len(X))}


def _get_y_true(df_full: pd.DataFrame, t: pd.Timestamp, eval_date: pd.Timestamp) -> pd.DataFrame:
    a = df_full[df_full["date"] == t][["repo_id", "stars"]].rename(columns={"stars": "stars_t"})
    b = df_full[df_full["date"] == eval_date][["repo_id", "stars"]].rename(columns={"stars": "stars_eval"})
    m = a.merge(b, on="repo_id", how="inner")
    m["y_true"] = m["stars_eval"] - m["stars_t"]
    return m[["repo_id", "y_true"]]


def main() -> None:
    storage = AzureStorage()

    print("Loading snapshots from Azure ...")
    snapshots = _load_all_snapshots(storage)
    if not snapshots:
        raise RuntimeError("Aucun snapshot trouvé sur Azure")

    df = snapshots_to_dataframe(snapshots)
    df = df.dropna(subset=["stars"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["repo_id", "date"])

    all_dates = sorted(df["date"].unique())
    min_date = all_dates[0]
    max_date = all_dates[-1]

    last_t = max_date - pd.Timedelta(days=HORIZON_DAYS)
    candidate_t_dates = [d for d in all_dates if (d <= last_t)]
    if not candidate_t_dates:
        raise RuntimeError("Pas assez de dates pour backtest (horizon trop grand ?)")

    history: List[Dict] = []
    retrain_events: List[Dict] = []

    model = None
    last_train_date = None
    last_train_rows = 0
    pending_trigger_reason: str | None = None

    print(
        f"Backtest: window={WINDOW_DAYS}d horizon={HORIZON_DAYS}d "
        f"dates={len(candidate_t_dates)} (from {min_date.date()} to {last_t.date()})"
    )
    print(f"Retrain every {RETRAIN_EVERY_DAYS} day(s) OR when rolling R² < {R2_THRESHOLD}")

    for i, t in enumerate(candidate_t_dates):
        retrained_flag = False
        retrain_reason = None

        eval_date = t + pd.Timedelta(days=HORIZON_DAYS)
        df_upto_t = df[df["date"] <= t].copy()

        # Decide retrain
        need_retrain = False
        if model is None:
            need_retrain = True
        elif last_train_date is not None:
            days_since = int((t - last_train_date).days)
            if days_since >= RETRAIN_EVERY_DAYS:
                need_retrain = True

        # Trigger on rolling r2
        if len(history) >= ROLLING_EVAL_WINDOW:
            rolling = [h["metrics"]["r2"] for h in history[-ROLLING_EVAL_WINDOW:]]
            rolling_mean = float(np.nanmean(rolling))
            if rolling_mean < R2_THRESHOLD:
                need_retrain = True
                pending_trigger_reason = f"rolling_r2_{ROLLING_EVAL_WINDOW}={rolling_mean:.3f} < {R2_THRESHOLD}"
                retrain_events.append({"t": t.strftime("%Y-%m-%d"), "reason": pending_trigger_reason})

        if need_retrain:
            out = _train_model(df_upto_t)
            model = out["model"]
            last_train_date = t
            last_train_rows = out["train_rows"]

            print(f"Train @ {t.date()} -> train_rows={last_train_rows}")
            if model is None:
                print("Model not trained (not enough rows yet). Skipping predictions until model exists.")
                continue

            retrained_flag = True
            retrain_reason = pending_trigger_reason or f"periodic_every_{RETRAIN_EVERY_DAYS}d"
            pending_trigger_reason = None

        # Predict at t
        X_pred, meta_pred, _feature_cols = build_features_for_prediction(
            df_upto_t, window_days=WINDOW_DAYS, as_of_date=t
        )
        if len(X_pred) == 0:
            continue

        y_pred = model.predict(X_pred)

        # Truth at t+h
        y_true_df = _get_y_true(df, t, eval_date)
        pred_df = meta_pred[["repo_id"]].copy()
        pred_df["y_pred"] = y_pred

        merged = pred_df.merge(y_true_df, on="repo_id", how="inner")
        if len(merged) < 200:
            continue

        r2 = float(r2_score(merged["y_true"], merged["y_pred"]))
        mse = float(mean_squared_error(merged["y_true"], merged["y_pred"]))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(merged["y_true"], merged["y_pred"]))


        recent_r2 = (
            [h["metrics"]["r2"] for h in history[-(ROLLING_EVAL_WINDOW - 1):]]
            if ROLLING_EVAL_WINDOW > 1 else []
        )
        rolling_r2_mean_before = float(np.nanmean(recent_r2)) if recent_r2 else None

        history.append(
            {
                "t": t.strftime("%Y-%m-%d"),
                "eval_date": eval_date.strftime("%Y-%m-%d"),
                "n_repos": int(len(merged)),
                "train_rows": int(last_train_rows),
                "model_last_train_date": last_train_date.strftime("%Y-%m-%d") if last_train_date else None,
                "metrics": {"r2": r2, "rmse": rmse, "mae": mae},
                "rolling_r2_mean_before": rolling_r2_mean_before,
                "retrained_today": retrained_flag,
                "triggered_retrain": bool(retrain_reason and retrain_reason.startswith("rolling_r2_")),
                "trigger_reason": retrain_reason,
            }
        )

        if i % 5 == 0:
            print(f"Eval t={t.date()} -> eval={eval_date.date()} n={len(merged)} r2={r2:.3f} mae={mae:.2f}")

    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "window_days": WINDOW_DAYS,
        "horizon_days": HORIZON_DAYS,
        "retrain_every_days": RETRAIN_EVERY_DAYS,
        "r2_threshold": R2_THRESHOLD,
        "rolling_eval_window": ROLLING_EVAL_WINDOW,
        "n_points": len(history),
        "history": history,
        "retrain_events": retrain_events,
        "notes": "Backtest/replay: train only on data <= t; evaluate at t+horizon when available in history.",
    }

    storage.upload_json(BACKTEST_HISTORY_BLOB, payload)
    print(f"Backtest metrics uploaded: {BACKTEST_HISTORY_BLOB} (points={len(history)})")

    # ============= MLflow (PROPRE) =============
    mlflow_setup()
    with mlflow.start_run(run_name="solution") as run:
        mlflow.log_param("window_days", WINDOW_DAYS)
        mlflow.log_param("horizon_days", HORIZON_DAYS)
        mlflow.log_param("retrain_every_days", RETRAIN_EVERY_DAYS)
        mlflow.log_param("r2_threshold", R2_THRESHOLD)
        mlflow.log_param("rolling_eval_window", ROLLING_EVAL_WINDOW)
        mlflow.log_param("rf_n_estimators", RF_N_ESTIMATORS)
        mlflow.log_param("rf_max_depth", RF_MAX_DEPTH)

        r2_vals = [h["metrics"]["r2"] for h in history]
        mae_vals = [h["metrics"]["mae"] for h in history]
        rmse_vals = [h["metrics"].get("rmse") for h in history if h["metrics"].get("rmse") is not None]

        if r2_vals:
            mlflow.log_metric("r2_mean", float(np.mean(r2_vals)))
            mlflow.log_metric("r2_min", float(np.min(r2_vals)))
            mlflow.log_metric("r2_last", float(r2_vals[-1]))

        if mae_vals:
            mlflow.log_metric("mae_mean", float(np.mean(mae_vals)))
            mlflow.log_metric("mae_max", float(np.max(mae_vals)))
        if rmse_vals:
            mlflow.log_metric("rmse_mean", float(np.mean(rmse_vals)))
            mlflow.log_metric("rmse_max", float(np.max(rmse_vals)))
            mlflow.log_metric("rmse_last", float(rmse_vals[-1]))


        n_trigger = sum(1 for h in history if h.get("triggered_retrain"))
        mlflow.log_metric("n_points", len(history))
        mlflow.log_metric("n_triggered_retrain", int(n_trigger))

        mlflow.set_tag("pipeline_step", "backtest")

        # artefact (json)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "metrics_history_backtest.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(p, artifact_path="backtest")


if __name__ == "__main__":
    main()
