import numpy as np
import pandas as pd
import pytest

from src.features import add_cyclical_features, build_features
from src.train import time_based_split


def _make_raw_df(n_hours=48, start="2012-09-30"):
    """Build a minimal fake raw dataframe shaped like hour.csv."""
    dteday = pd.date_range(start, periods=n_hours // 24 + 1, freq="D")
    rows = []
    instant = 1
    for day in dteday:
        for hr in range(24):
            if len(rows) >= n_hours:
                break
            rows.append({
                "instant": instant,
                "dteday": day,
                "season": 3,
                "yr": 1,
                "mnth": day.month,
                "hr": hr,
                "holiday": 0,
                "weekday": day.dayofweek,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.5,
                "atemp": 0.5,
                "hum": 0.5,
                "windspeed": 0.2,
                "casual": 10,
                "registered": 90,
                "cnt": 100,
            })
            instant += 1
    return pd.DataFrame(rows)


def test_cyclical_encoding_wraps_hour_0_and_23():
    df = pd.DataFrame({"hr": [0, 23], "mnth": [1, 1]})
    out = add_cyclical_features(df)
    # hour 0 and hour 23 are adjacent on a 24-hour clock, so their sin/cos
    # values should be close to each other, not far apart like raw ints (0 vs 23).
    dist = np.hypot(out["hr_sin"][0] - out["hr_sin"][1], out["hr_cos"][0] - out["hr_cos"][1])
    assert dist == pytest.approx(2 * np.sin(np.pi / 24), abs=1e-6)
    # sanity: hour 0 should be sin=0, cos=1
    assert out["hr_sin"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["hr_cos"][0] == pytest.approx(1.0, abs=1e-9)


def test_build_features_drops_leakage_columns():
    raw = _make_raw_df()
    featured = build_features(raw)
    assert "casual" not in featured.columns
    assert "registered" not in featured.columns
    assert "instant" not in featured.columns
    # target itself should remain — it's not a leakage column
    assert "cnt" in featured.columns


def test_build_features_replaces_hr_and_mnth_with_cyclical_pairs():
    raw = _make_raw_df()
    featured = build_features(raw)
    assert "hr" not in featured.columns
    assert "mnth" not in featured.columns
    for col in ["hr_sin", "hr_cos", "mnth_sin", "mnth_cos"]:
        assert col in featured.columns


def test_time_based_split_has_no_date_overlap():
    raw = _make_raw_df(n_hours=72)  # spans 2012-09-30, 10-01, 10-02
    featured = build_features(raw)
    train, test = time_based_split(featured, "2012-10-01")

    assert train["dteday"].max() < pd.Timestamp("2012-10-01")
    assert test["dteday"].min() >= pd.Timestamp("2012-10-01")
    assert len(train) + len(test) == len(featured)
    assert len(train) > 0 and len(test) > 0
