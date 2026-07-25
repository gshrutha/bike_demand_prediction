"""Train and compare demand-forecasting models on a time-based split.

Why a time-based split instead of a random one?
    Nearby hours are highly correlated (rush hour today looks like rush hour
    yesterday). A random train/test split would let the model train on hours
    immediately before and after a given test hour, leaking future
    information into training and producing scores that look great in
    development but don't hold up once deployed on genuinely unseen future
    data. Splitting by date — train on the past, test on the future — is the
    only split that honestly simulates production use: "given everything up
    to today, predict tomorrow's demand."

Usage:
    python -m src.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.download_data import download_data
from src.features import build_features, get_feature_columns, load_raw

TARGET = "cnt"


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def time_based_split(df: pd.DataFrame, test_start_date: str):
    """Split chronologically: everything before `test_start_date` is train,
    everything on/after it is test. No shuffling, no randomness.
    """
    cutoff = pd.Timestamp(test_start_date)
    train = df[df["dteday"] < cutoff].reset_index(drop=True)
    test = df[df["dteday"] >= cutoff].reset_index(drop=True)
    return train, test


def evaluate_predictions(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2)}


def main():
    config = load_config()
    download_data(config)  # no-op if already downloaded

    raw = load_raw(config["data"]["raw_csv"])
    featured = build_features(raw)
    feature_cols = get_feature_columns(featured)

    train_df, test_df = time_based_split(featured, config["split"]["test_start_date"])
    print(f"Train: {len(train_df)} rows ({train_df['dteday'].min().date()} to "
          f"{train_df['dteday'].max().date()})")
    print(f"Test:  {len(test_df)} rows ({test_df['dteday'].min().date()} to "
          f"{test_df['dteday'].max().date()})")

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    models = {
        "linear_regression": LinearRegression(**config["model"]["linear_regression"]),
        "random_forest": RandomForestRegressor(**config["model"]["random_forest"]),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)
        results[name] = metrics
        fitted[name] = model
        print(f"{name:>18}: MAE={metrics['mae']:>7}  RMSE={metrics['rmse']:>7}")

    best_name = min(results, key=lambda n: results[n]["mae"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (lowest MAE)")

    out_dir = Path(config["output"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, config["output"]["model_path"])

    # Persist the exact feature columns/order used at train time, and which
    # model won, so predict.py/evaluate.py can reproduce it exactly.
    metrics_payload = {
        "results": results,
        "best_model": best_name,
        "test_start_date": config["split"]["test_start_date"],
        "feature_columns": feature_cols,
    }
    with open(config["output"]["metrics_path"], "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved model to {config['output']['model_path']}")
    print(f"Saved metrics to {config['output']['metrics_path']}")

    if best_name == "random_forest":
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        print("\nTop 10 feature importances:")
        print(importances.head(10).to_string())


if __name__ == "__main__":
    main()
