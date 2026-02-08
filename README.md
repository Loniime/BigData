# GitHub Stars MLOps Platform (Big Data / MLOps)

End-to-end MLOps pipeline to **predict GitHub repository star growth** over the next **7 days**.
The project includes: daily data ingestion, feature engineering, model training, backtesting/monitoring, prediction generation, and a Streamlit dashboard.
Deployment is running on **Azure Container Apps**, using **Azure Blob Storage** for persistence and **MLflow** for experiment tracking.

## Live Demo (Azure)
- **Dashboard (Streamlit):** <PUT_DASHBOARD_URL_HERE>
- **MLflow Tracking UI:** <PUT_MLFLOW_URL_HERE>

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

