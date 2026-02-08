# GitHub Stars MLOps Platform (Big Data / MLOps)

End-to-end MLOps pipeline to **predict GitHub repository star growth** over the next **7 days**.
The project includes: daily data ingestion, feature engineering, model training, backtesting/monitoring, prediction generation, and a Streamlit dashboard.
Deployment is running on **Azure Container Apps**, using **Azure Blob Storage** for persistence and **MLflow** for experiment tracking.

## Live Demo (Azure)
- **Dashboard (Streamlit):** https://dashboard.orangeocean-5ade08aa.germanywestcentral.azurecontainerapps.io/
- **MLflow Tracking UI:** https://mlflow.orangeocean-5ade08aa.germanywestcentral.azurecontainerapps.io/#/experiments/2/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

> Secrets are not stored in this repository. Configuration is done via environment variables.

---

## Architecture (high-level)

**Data Layer**
- **Azure Blob Storage**
  - `daily_snapshots/` : daily GitHub metrics snapshots
  - `models/` : exported model + metadata
  - `predictions/` : daily prediction files
  - `monitoring/` : metrics history (backtest + monitoring)

**Experiment Tracking**
- **MLflow (Azure)** stores:
  - params / metrics
  - model artifact (`mlflow.sklearn.log_model`)
  - training metadata JSON

**Apps / Jobs**
- **Streamlit Dashboard**:
  - shows dataset stats & plots
  - displays backtest metrics over time
  - provides **Train** and **Predict** buttons (calls the jobs)
  - shows latest predictions and allows CSV export
- **Jobs (Python)**:
  - `step4_train_model` : train + evaluate + upload model + log run in MLflow
  - `step5_predict_monitor` : load model from Azure + predict + upload predictions + update monitoring
  - `step6_backtest` : replay training/prediction over time and produce backtest metrics

---

## Repository structure

flowchart TB
  %% =========================
  %% SOURCES & STOCKAGE
  %% =========================
  subgraph GitHub["GitHub"]
    GH["GitHub API / repos metrics"]
  end

  subgraph Blob["Azure Blob Storage (persistant)"]
    S["daily_snapshots/\nsnapshot_YYYY-MM-DD.json"]
    MOD["models/\nstars_delta_7d_model.joblib\nstars_delta_7d_metadata.json"]
    PRED["predictions/\npred_YYYY-MM-DD.json"]
    MON["monitoring/\nmetrics_history.json\nmetrics_history_backtest.json\nretrain_requested.json (option)"]
  end

  subgraph Track["MLflow (Azure)"]
    EXP["Experiment: github-stars\nRuns + Params + Metrics + Artifacts"]
  end

  %% =========================
  %% CORE / FEATURE
  %% =========================
  subgraph Core["Core (src/core)"]
    FB["data_build.py\nsnapshots_to_dataframe()\nbuild_supervised_dataset()"]
    AZ["azure_storage.py\nlist/download/upload"]
    SET["settings.py\n(env vars)"]
    MLF["mlflow_utils.py\nmlflow_setup()"]
  end

  %% =========================
  %% JOBS
  %% =========================
  subgraph Jobs["Pipeline Jobs (src/jobs)"]
    J4["Step4: Train\nstep4_train_model"]
    J6["Step6: Backtest\nstep6_backtest"]
    J5["Step5: Predict + Monitor\nstep5_predict_monitor"]
  end

  %% =========================
  %% DASHBOARD
  %% =========================
  subgraph UI["Streamlit Dashboard (main.py)"]
    UI1["Home\n- dataset stats\n- top repos\n- time series"]
    UI2["Monitoring\n- backtest curves (R²/MAE)\n- retrain threshold"]
    UI3["Actions\n- START TRAINING\n- GENERATE PREDICTIONS"]
    UI4["MLflow Explorer\n- list runs\n- metrics/params"]
    UI5["Predictions\n- latest topN\n- CSV export\n- history list"]
  end

  %% =========================
  %% FLOWS
  %% =========================

  %% Data collection (existing in your project even if not shown here)
  GH -->|daily collection| S

  %% Training
  S -->|download snapshots| J4
  J4 -->|feature engineering| FB
  J4 -->|log params/metrics/model| EXP
  J4 -->|upload model + metadata| MOD

  %% Predict
  MOD -->|load model| J5
  S -->|latest snapshots| J5
  J5 -->|upload predictions| PRED
  J5 -->|append monitoring metrics| MON
  J5 -->|log run| EXP

  %% Backtest
  S -->|historical snapshots| J6
  J6 -->|rolling train/eval\nretrain logic| MON
  J6 -->|log run| EXP

  %% Dashboard reads
  UI1 -->|read| S
  UI2 -->|read| MON
  UI5 -->|read| PRED
  UI4 -->|read| EXP
  UI3 -->|trigger| J4
  UI3 -->|trigger| J5

  %% Core dependencies
  J4 -.-> AZ
  J5 -.-> AZ
  J6 -.-> AZ
  UI1 -.-> AZ
  J4 -.-> MLF
  J5 -.-> MLF
  J6 -.-> MLF
  SET -.-> J4
  SET -.-> J5
  SET -.-> J6
  SET -.-> UI1


