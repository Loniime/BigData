import os

class Settings:
    # Azure
    AZURE_CONNECTION_STRING: str = os.getenv("AZURE_CONNECTION_STRING", "")
    AZURE_CONTAINER: str = os.getenv("AZURE_CONTAINER", "github-data")

    # GitHub
    GITHUB_TOKENS: str = os.getenv("GITHUB_TOKENS", "")

    # ML
    ML_WINDOW_DAYS: int = int(os.getenv("ML_WINDOW_DAYS", "7"))
    ML_HORIZON_DAYS: int = int(os.getenv("ML_HORIZON_DAYS", "7"))
    ML_TEST_LAST_DAYS: int = int(os.getenv("ML_TEST_LAST_DAYS", "14"))

    # Paths Azure
    SNAPSHOT_PREFIX: str = "daily_snapshots/snapshot_"
    MODEL_BLOB: str = "models/stars_delta_7d_model.joblib"
    META_BLOB: str = "models/stars_delta_7d_metadata.json"
    PRED_PREFIX: str = "predictions/pred_"
    METRICS_HISTORY_BLOB: str = "monitoring/metrics_history.json"
    RETRAIN_REQUEST_BLOB: str = "monitoring/retrain_requested.json"

    # Model hyperparams
    N_ESTIMATORS: int = int(os.getenv("RF_N_ESTIMATORS", "300"))
    MAX_DEPTH: int = int(os.getenv("RF_MAX_DEPTH", "20"))
    RANDOM_STATE: int = int(os.getenv("ML_RANDOM_STATE", "42"))

    CLASSIFIER_MODEL_BLOB = "models/stars_growth_classifier.joblib"
    CLASSIFIER_THRESHOLDS_BLOB = "models/classification_thresholds.json"
    CLASSIFIER_META_BLOB = "models/classifier_meta.json"

