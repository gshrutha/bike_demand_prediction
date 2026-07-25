"""Feature engineering for the bike-share hourly demand dataset.

Key decisions:
  - `hr` and `mnth` are cyclical (hour 23 is adjacent to hour 0), so they're
    encoded as sin/cos pairs rather than left as raw integers or one-hot'd.
  - `casual` and `registered` sum exactly to `cnt` (the target), so they are
    dropped — leaving them in would let the model "cheat" by learning a
    trivial addition instead of a real demand forecast.
  - `temp`, `atemp`, `hum`, `windspeed` are already min-max normalized in the
    source data (0-1), so they're used as-is.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "cnt"

# Columns that leak the target (casual + registered == cnt) and identifiers
# that carry no predictive signal of their own.
DROP_COLS = ["instant", "casual", "registered"]

# Categorical columns encoded as one-hot (small, non-cyclical cardinality).
CATEGORICAL_COLS = ["season", "weathersit", "weekday"]

# Numeric columns already normalized in the source data — used as-is.
NUMERIC_COLS = ["temp", "atemp", "hum", "windspeed", "holiday", "workingday", "yr"]


def load_raw(raw_csv: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv, parse_dates=["dteday"])
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour-of-day (0-23) and month (1-12) as sin/cos pairs."""
    df = df.copy()
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature-engineering pipeline: cyclical encoding, one-hot
    encoding of categoricals, leakage-column removal. Keeps `dteday` and
    `hr` around (needed for the time-based split) — callers should drop
    them right before fitting.
    """
    df = add_cyclical_features(df)
    df = df.drop(columns=["hr", "mnth"])  # replaced by their sin/cos pairs
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """All columns usable as model features (excludes target + date index)."""
    exclude = {TARGET, "dteday"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    raw = load_raw("data/raw/hour.csv")
    featured = build_features(raw)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    featured.to_csv("data/processed/features.csv", index=False)
    print(f"Wrote {len(featured)} rows, {len(get_feature_columns(featured))} feature columns "
          f"to data/processed/features.csv")
