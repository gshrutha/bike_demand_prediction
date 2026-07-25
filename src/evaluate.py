"""Load the saved model and evaluate it on the held-out (future) test set.

Regenerates the same time-based test split used in training and produces
the actual-vs-predicted plot and a feature-importance chart.

Usage:
    python -m src.evaluate
"""

import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

matplotlib.use("Agg")  # headless-safe backend for saving PNGs

from src.features import build_features, get_feature_columns, load_raw
from src.train import TARGET, evaluate_predictions, load_config, time_based_split


def plot_actual_vs_predicted(test_df: pd.DataFrame, y_true, y_pred, out_path: str):
    # Aggregate to daily totals for a readable chart (17k hourly points is
    # too dense to read; daily totals show the trend clearly).
    daily = pd.DataFrame({
        "dteday": test_df["dteday"],
        "actual": y_true.values,
        "predicted": y_pred,
    }).groupby("dteday").sum()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily.index, daily["actual"], label="Actual", linewidth=1.8)
    ax.plot(daily.index, daily["predicted"], label="Predicted", linewidth=1.8, linestyle="--")
    ax.set_title("Daily Bike Demand: Actual vs. Predicted (held-out test period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total rides per day")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved actual-vs-predicted plot to {out_path}")


def plot_feature_importance(model, feature_cols, out_path: str, top_n: int = 12):
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_ (not a tree-based model); skipping chart.")
        return
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot.barh(ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance plot to {out_path}")


def main():
    config = load_config()

    raw = load_raw(config["data"]["raw_csv"])
    featured = build_features(raw)
    _, test_df = time_based_split(featured, config["split"]["test_start_date"])

    with open(config["output"]["metrics_path"]) as f:
        saved = json.load(f)
    feature_cols = saved["feature_columns"]

    model = joblib.load(config["output"]["model_path"])

    X_test, y_test = test_df[feature_cols], test_df[TARGET]
    preds = model.predict(X_test)
    metrics = evaluate_predictions(y_test, preds)

    print(f"Best model ({saved['best_model']}) on held-out test set "
          f"({saved['test_start_date']} onward):")
    print(f"  MAE:  {metrics['mae']}")
    print(f"  RMSE: {metrics['rmse']}")

    plot_actual_vs_predicted(test_df, y_test, preds, config["output"]["plot_path"])
    plot_feature_importance(model, feature_cols, config["output"]["feature_importance_path"])


if __name__ == "__main__":
    main()
