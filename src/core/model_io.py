from joblib import dump, load
from typing import Any

def save_model(path: str, model: Any) -> None:
    dump(model, path)

def load_model(path: str) -> Any:
    return load(path)


