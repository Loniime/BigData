from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def snapshots_to_dataframe(snapshots: List[Dict]) -> pd.DataFrame:
    rows = []
    for snap in snapshots:
        date = snap.get("date")
        for r in snap.get("repos", []):
            rows.append({
                "date": date,
                "repo_id": r.get("id"),
                "full_name": r.get("full_name"),
                "stars": r.get("stars") if "stars" in r else r.get("stargazers_count"),
                "forks": r.get("forks"),
                "watchers": r.get("watchers"),
                "open_issues": r.get("open_issues"),
                "language": r.get("language"),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["stars", "forks", "watchers", "open_issues"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["repo_id", "date", "stars"])
    df["repo_id"] = df["repo_id"].astype(int)
    return df


def build_supervised_dataset(
    df: pd.DataFrame,
    window_days: int,
    horizon_days: int,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    df = df.sort_values(["repo_id", "date"]).copy()

    df[["forks", "watchers", "open_issues"]] = (
        df.groupby("repo_id")[["forks", "watchers", "open_issues"]].ffill()
    )
    df[["forks", "watchers", "open_issues"]] = df[["forks", "watchers", "open_issues"]].fillna(0)

    for col in ["stars", "forks", "watchers", "open_issues"]:
        df[f"{col}_diff1"] = df.groupby("repo_id")[col].diff()

    g = df.groupby("repo_id", group_keys=False)
    def roll_mean(s): return s.rolling(window_days, min_periods=window_days).mean()
    def roll_std(s): return s.rolling(window_days, min_periods=window_days).std()
    def roll_sum(s): return s.rolling(window_days, min_periods=window_days).sum()

    df["stars_gain_w"] = g["stars_diff1"].apply(roll_sum)
    df["stars_mean_gain_w"] = g["stars_diff1"].apply(roll_mean)
    df["stars_std_gain_w"] = g["stars_diff1"].apply(roll_std)

    df["forks_gain_w"] = g["forks_diff1"].apply(roll_sum)
    df["watchers_gain_w"] = g["watchers_diff1"].apply(roll_sum)
    df["issues_gain_w"] = g["open_issues_diff1"].apply(roll_sum)

    df["stars_t_plus_h"] = g["stars"].shift(-horizon_days)
    df["y_delta_stars_h"] = df["stars_t_plus_h"] - df["stars"]

    feature_cols = [
        "stars", "forks", "watchers", "open_issues",
        "stars_gain_w", "stars_mean_gain_w", "stars_std_gain_w",
        "forks_gain_w", "watchers_gain_w", "issues_gain_w",
    ]

    valid = df[feature_cols + ["y_delta_stars_h", "stars_t_plus_h"]].notna().all(axis=1)
    df2 = df.loc[valid].copy()

    X = df2[feature_cols].astype(float)
    y = df2["y_delta_stars_h"].astype(float)
    meta = df2[["repo_id", "date", "full_name", "stars", "stars_t_plus_h"]].copy()
    meta = meta.rename(columns={"stars": "stars_t"})

    return X, y, feature_cols, meta


def build_features_for_prediction(df: pd.DataFrame, window_days: int, as_of_date: pd.Timestamp):
    df = df.sort_values(["repo_id", "date"]).copy()

    df[["forks", "watchers", "open_issues"]] = (
        df.groupby("repo_id")[["forks", "watchers", "open_issues"]].ffill()
    )
    df[["forks", "watchers", "open_issues"]] = df[["forks", "watchers", "open_issues"]].fillna(0)

    for col in ["stars", "forks", "watchers", "open_issues"]:
        df[f"{col}_diff1"] = df.groupby("repo_id")[col].diff()

    g = df.groupby("repo_id", group_keys=False)
    def roll_mean(s): return s.rolling(window_days, min_periods=window_days).mean()
    def roll_std(s): return s.rolling(window_days, min_periods=window_days).std()
    def roll_sum(s): return s.rolling(window_days, min_periods=window_days).sum()

    df["stars_gain_w"] = g["stars_diff1"].apply(roll_sum)
    df["stars_mean_gain_w"] = g["stars_diff1"].apply(roll_mean)
    df["stars_std_gain_w"] = g["stars_diff1"].apply(roll_std)

    df["forks_gain_w"] = g["forks_diff1"].apply(roll_sum)
    df["watchers_gain_w"] = g["watchers_diff1"].apply(roll_sum)
    df["issues_gain_w"] = g["open_issues_diff1"].apply(roll_sum)

    feature_cols = [
        "stars", "forks", "watchers", "open_issues",
        "stars_gain_w", "stars_mean_gain_w", "stars_std_gain_w",
        "forks_gain_w", "watchers_gain_w", "issues_gain_w",
    ]

    df_today = df[df["date"] == as_of_date].copy()
    valid = df_today[feature_cols].notna().all(axis=1)
    df_today = df_today.loc[valid].copy()

    X_pred = df_today[feature_cols].astype(float)
    meta = df_today[["repo_id", "full_name", "date", "stars"]].copy()
    meta = meta.rename(columns={"stars": "stars_t"})

    return X_pred, meta, feature_cols
