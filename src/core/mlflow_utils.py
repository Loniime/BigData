import os
import mlflow

def mlflow_setup():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()

    if tracking_uri and not tracking_uri.startswith(("http://", "https://", "file:", "sqlite:", "postgresql:")):
        tracking_uri = "https://" + tracking_uri

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "github-stars")
    mlflow.set_experiment(exp_name)

    print("MLFLOW_TRACKING_URI (resolved) =", tracking_uri)
    print("mlflow.get_tracking_uri() =", mlflow.get_tracking_uri())
