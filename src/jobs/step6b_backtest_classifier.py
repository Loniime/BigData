# src/jobs/step6b_backtest_classifier.py
from __future__ import annotations

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

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

RETRAIN_EVERY_DAYS = int(os.getenv("CLF_BACKTEST_RETRAIN_EVERY_DAYS", "14"))
F1_THRESHOLD = float(os.getenv("CLF_BACKTEST_F1_THRESHOLD", "0.40"))
RECALL_THRESHOLD = float(os.getenv("CLF_BACKTEST_RECALL_THRESHOLD", "0.30"))
ROLLING_EVAL_WINDOW = int(os.getenv("CLF_BACKTEST_ROLLING_EVAL_WINDOW", "5"))

BACKTEST_HISTORY_BLOB = os.getenv(
    "CLF_BACKTEST_HISTORY_BLOB", "monitoring/classifier_metrics_history_backtest.json"
)

CLF_N_ESTIMATORS = int(os.getenv("CLF_N_ESTIMATORS", "300"))
CLF_MAX_DEPTH = int(os.getenv("CLF_MAX_DEPTH", "20"))
ML_RANDOM_STATE = int(os.getenv("ML_RANDOM_STATE", "42"))

Q_LOW = float(os.getenv("CLF_Q_LOW", "0.60"))
Q_HIGH = float(os.getenv("CLF_Q_HIGH", "0.90"))


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


def _build_classes_from_relative_gain(y_delta: pd.Series, stars_t: pd.Series, q_low=Q_LOW, q_high=Q_HIGH):
    """
    Construit les classes : 0=stable, 1=faible, 2=explosion
    Basé sur le gain relatif : y_rel = y_delta / (stars_t + 1)
    """
    stars_safe = stars_t.fillna(0).astype(float)
    y_rel = y_delta.astype(float) / (stars_safe + 1.0)

    th_low = float(np.nanquantile(y_rel, q_low))
    th_high = float(np.nanquantile(y_rel, q_high))

    y_class = np.zeros(len(y_rel), dtype=int)
    y_class[(y_rel > th_low) & (y_rel <= th_high)] = 1
    y_class[y_rel > th_high] = 2

    thresholds = {
        "q_low": q_low,
        "q_high": q_high,
        "th_low": th_low,
        "th_high": th_high,
    }
    return y_rel, y_class, thresholds


def _train_classifier(df_upto_t: pd.DataFrame) -> Dict:
    """
    Entraîne un classifier sur les données jusqu'à t
    """
    X, y_delta, feature_cols, meta = build_supervised_dataset(
        df_upto_t, window_days=WINDOW_DAYS, horizon_days=HORIZON_DAYS
    )
    
    if len(X) < 5000:
        return {"model": None, "feature_cols": feature_cols, "train_rows": int(len(X)), "thresholds": None}
    
    if "stars_t" not in meta.columns:
        return {"model": None, "feature_cols": feature_cols, "train_rows": int(len(X)), "thresholds": None}
    
    # Build classes
    y_rel, y_class, thresholds = _build_classes_from_relative_gain(
        y_delta=y_delta,
        stars_t=meta["stars_t"],
        q_low=Q_LOW,
        q_high=Q_HIGH,
    )
    
    clf = RandomForestClassifier(
        n_estimators=CLF_N_ESTIMATORS,
        max_depth=CLF_MAX_DEPTH,
        random_state=ML_RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X, y_class)
    
    return {
        "model": clf,
        "feature_cols": feature_cols,
        "train_rows": int(len(X)),
        "thresholds": thresholds
    }


def _get_y_true_delta_and_stars(df_full: pd.DataFrame, t: pd.Timestamp, eval_date: pd.Timestamp) -> pd.DataFrame:
    """
    Récupère les vraies valeurs : y_true (delta stars) et stars_t pour construire les classes
    """
    a = df_full[df_full["date"] == t][["repo_id", "stars"]].rename(columns={"stars": "stars_t"})
    b = df_full[df_full["date"] == eval_date][["repo_id", "stars"]].rename(columns={"stars": "stars_eval"})
    m = a.merge(b, on="repo_id", how="inner")
    m["y_true_delta"] = m["stars_eval"] - m["stars_t"]
    return m[["repo_id", "y_true_delta", "stars_t"]]


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

    clf = None
    last_train_date = None
    last_train_rows = 0
    last_thresholds = None
    pending_trigger_reason: str | None = None

    print(
        f"Classifier Backtest: window={WINDOW_DAYS}d horizon={HORIZON_DAYS}d "
        f"dates={len(candidate_t_dates)} (from {min_date.date()} to {last_t.date()})"
    )
    print(f"Retrain every {RETRAIN_EVERY_DAYS} day(s) OR when rolling F1 < {F1_THRESHOLD}")

    for i, t in enumerate(candidate_t_dates):
        retrained_flag = False
        retrain_reason = None

        eval_date = t + pd.Timedelta(days=HORIZON_DAYS)
        df_upto_t = df[df["date"] <= t].copy()

        # Decide retrain
        need_retrain = False
        if clf is None:
            need_retrain = True
        elif last_train_date is not None:
            days_since = int((t - last_train_date).days)
            if days_since >= RETRAIN_EVERY_DAYS:
                need_retrain = True

        # Trigger on rolling F1
        if len(history) >= ROLLING_EVAL_WINDOW:
            rolling_f1 = [h["metrics"]["f1_macro"] for h in history[-ROLLING_EVAL_WINDOW:]]
            rolling_mean = float(np.nanmean(rolling_f1))
            if rolling_mean < F1_THRESHOLD:
                need_retrain = True
                pending_trigger_reason = f"rolling_f1_{ROLLING_EVAL_WINDOW}={rolling_mean:.3f} < {F1_THRESHOLD}"
                retrain_events.append({"t": t.strftime("%Y-%m-%d"), "reason": pending_trigger_reason})

        # Trigger on rolling recall explosion
        if len(history) >= ROLLING_EVAL_WINDOW:
            rolling_recall = [h["metrics"]["recall_explosion"] for h in history[-ROLLING_EVAL_WINDOW:] if h["metrics"]["recall_explosion"] is not None]
            if rolling_recall:
                rolling_mean_recall = float(np.nanmean(rolling_recall))
                if rolling_mean_recall < RECALL_THRESHOLD:
                    need_retrain = True
                    if pending_trigger_reason is None:
                        pending_trigger_reason = f"rolling_recall_explosion_{ROLLING_EVAL_WINDOW}={rolling_mean_recall:.3f} < {RECALL_THRESHOLD}"
                        retrain_events.append({"t": t.strftime("%Y-%m-%d"), "reason": pending_trigger_reason})

        if need_retrain:
            out = _train_classifier(df_upto_t)
            clf = out["model"]
            last_train_date = t
            last_train_rows = out["train_rows"]
            last_thresholds = out["thresholds"]

            print(f"Train classifier @ {t.date()} -> train_rows={last_train_rows}")
            if clf is None:
                print("Classifier not trained (not enough rows yet). Skipping predictions.")
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

        y_pred_class = clf.predict(X_pred)

        # Truth at t+h
        truth_df = _get_y_true_delta_and_stars(df, t, eval_date)
        pred_df = meta_pred[["repo_id"]].copy()
        pred_df["y_pred_class"] = y_pred_class

        # Merge with truth
        merged = pred_df.merge(truth_df, on="repo_id", how="inner")
        if len(merged) < 200:
            continue

        # Build true classes using the same thresholds
        if last_thresholds is None:
            continue
        
        y_rel_true = merged["y_true_delta"].astype(float) / (merged["stars_t"].astype(float) + 1.0)
        y_true_class = np.zeros(len(y_rel_true), dtype=int)
        y_true_class[(y_rel_true > last_thresholds["th_low"]) & (y_rel_true <= last_thresholds["th_high"])] = 1
        y_true_class[y_rel_true > last_thresholds["th_high"]] = 2

        # Metrics
        accuracy = float(accuracy_score(y_true_class, merged["y_pred_class"]))
        f1_macro = float(f1_score(y_true_class, merged["y_pred_class"], average="macro"))
        
        # Recall/Precision explosion (class 2)
        recall_explosion = None
        precision_explosion = None
        if 2 in y_true_class:
            recall_vals = recall_score(y_true_class, merged["y_pred_class"], labels=[2], average=None, zero_division=0)
            if len(recall_vals) > 0:
                recall_explosion = float(recall_vals[0])
            
            precision_vals = precision_score(y_true_class, merged["y_pred_class"], labels=[2], average=None, zero_division=0)
            if len(precision_vals) > 0:
                precision_explosion = float(precision_vals[0])

        history.append(
            {
                "t": t.strftime("%Y-%m-%d"),
                "eval_date": eval_date.strftime("%Y-%m-%d"),
                "n_repos": int(len(merged)),
                "train_rows": int(last_train_rows),
                "model_last_train_date": last_train_date.strftime("%Y-%m-%d") if last_train_date else None,
                "metrics": {
                    "accuracy": accuracy,
                    "f1_macro": f1_macro,
                    "recall_explosion": recall_explosion,
                    "precision_explosion": precision_explosion,
                },
                "thresholds": last_thresholds,
                "retrained_today": retrained_flag,
                "triggered_retrain": bool(retrain_reason and ("rolling_f1_" in retrain_reason or "rolling_recall_" in retrain_reason)),
                "trigger_reason": retrain_reason,
            }
        )

        if i % 5 == 0:
            print(f"Eval t={t.date()} -> eval={eval_date.date()} n={len(merged)} f1={f1_macro:.3f} recall_exp={recall_explosion}")

    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "window_days": WINDOW_DAYS,
        "horizon_days": HORIZON_DAYS,
        "retrain_every_days": RETRAIN_EVERY_DAYS,
        "f1_threshold": F1_THRESHOLD,
        "recall_threshold": RECALL_THRESHOLD,
        "rolling_eval_window": ROLLING_EVAL_WINDOW,
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "n_points": len(history),
        "history": history,
        "retrain_events": retrain_events,
        "notes": "Classifier backtest: train on data <= t; evaluate at t+horizon.",
    }

    storage.upload_json(BACKTEST_HISTORY_BLOB, payload)
    print(f"Classifier backtest metrics uploaded: {BACKTEST_HISTORY_BLOB} (points={len(history)})")

    # MLflow logging
    mlflow_setup()
    with mlflow.start_run(run_name="classifier_backtest") as run:
        mlflow.log_param("window_days", WINDOW_DAYS)
        mlflow.log_param("horizon_days", HORIZON_DAYS)
        mlflow.log_param("retrain_every_days", RETRAIN_EVERY_DAYS)
        mlflow.log_param("f1_threshold", F1_THRESHOLD)
        mlflow.log_param("recall_threshold", RECALL_THRESHOLD)
        mlflow.log_param("rolling_eval_window", ROLLING_EVAL_WINDOW)
        mlflow.log_param("q_low", Q_LOW)
        mlflow.log_param("q_high", Q_HIGH)
        mlflow.log_param("clf_n_estimators", CLF_N_ESTIMATORS)
        mlflow.log_param("clf_max_depth", CLF_MAX_DEPTH)

        f1_vals = [h["metrics"]["f1_macro"] for h in history]
        accuracy_vals = [h["metrics"]["accuracy"] for h in history]
        recall_vals = [h["metrics"]["recall_explosion"] for h in history if h["metrics"]["recall_explosion"] is not None]
        precision_vals = [h["metrics"]["precision_explosion"] for h in history if h["metrics"]["precision_explosion"] is not None]

        if f1_vals:
            mlflow.log_metric("f1_macro_mean", float(np.mean(f1_vals)))
            mlflow.log_metric("f1_macro_min", float(np.min(f1_vals)))
            mlflow.log_metric("f1_macro_last", float(f1_vals[-1]))

        if accuracy_vals:
            mlflow.log_metric("accuracy_mean", float(np.mean(accuracy_vals)))
            mlflow.log_metric("accuracy_min", float(np.min(accuracy_vals)))

        if recall_vals:
            mlflow.log_metric("recall_explosion_mean", float(np.mean(recall_vals)))
            mlflow.log_metric("recall_explosion_min", float(np.min(recall_vals)))

        if precision_vals:
            mlflow.log_metric("precision_explosion_mean", float(np.mean(precision_vals)))

        n_trigger = sum(1 for h in history if h.get("triggered_retrain"))
        mlflow.log_metric("n_points", len(history))
        mlflow.log_metric("n_triggered_retrain", int(n_trigger))

        mlflow.set_tag("pipeline_step", "classifier_backtest")

        # Artifact (json)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "classifier_metrics_history_backtest.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(p, artifact_path="classifier_backtest")


if __name__ == "__main__":
    main()