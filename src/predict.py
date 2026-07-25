"""Predict bike-share demand for a single hour from the command line.

Runs the input through the exact same feature-engineering pipeline used at
training time, then reindexes to the saved training feature columns so a
single row lines up correctly with the one-hot columns the model expects
(a row with e.g. season=1 won't naturally produce season=2/3/4 dummy
columns, so any absent training column is filled with 0).

Example:
    python -m src.predict --hour 18 --month 6 --season 3 \\
        --weather 1 --workingday 1 --temp 0.7 --humidity 0.5 --windspeed 0.2
"""

import argparse
import json

import joblib
import pandas as pd
import yaml

from src.features import build_features


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Predict hourly bike-share ride demand.")
    p.add_argument("--hour", type=int, required=True, choices=range(0, 24),
                    metavar="[0-23]", help="Hour of day (0-23)")
    p.add_argument("--month", type=int, required=True, choices=range(1, 13),
                    metavar="[1-12]", help="Month (1-12)")
    p.add_argument("--season", type=int, required=True, choices=[1, 2, 3, 4],
                    help="1=spring, 2=summer, 3=fall, 4=winter")
    p.add_argument("--weather", type=int, required=True, choices=[1, 2, 3, 4],
                    help="1=clear, 2=mist/cloudy, 3=light rain/snow, 4=heavy rain/snow")
    p.add_argument("--workingday", type=int, required=True, choices=[0, 1],
                    help="1 if a working day (not weekend/holiday), else 0")
    p.add_argument("--temp", type=float, required=True,
                    help="Normalized temperature, 0.0-1.0 (raw temp / 41 deg C)")
    p.add_argument("--humidity", type=float, required=True,
                    help="Normalized humidity, 0.0-1.0")
    p.add_argument("--windspeed", type=float, required=True,
                    help="Normalized wind speed, 0.0-1.0")
    p.add_argument("--weekday", type=int, default=3, choices=range(0, 7),
                    metavar="[0-6]", help="Day of week, 0=Sunday (default: 3=Wednesday)")
    p.add_argument("--holiday", type=int, default=0, choices=[0, 1],
                    help="1 if a holiday, else 0 (default: 0)")
    p.add_argument("--year", type=int, default=1, choices=[0, 1],
                    help="0=2011, 1=2012 (default: 1, the most recent year in training data)")
    p.add_argument("--atemp", type=float, default=None,
                    help="Normalized 'feels like' temperature, 0.0-1.0 (default: same as --temp)")
    return p.parse_args()


def build_input_row(args) -> pd.DataFrame:
    """Assemble a single row shaped like the raw hour.csv, then run it
    through the same build_features() pipeline used in training.
    """
    atemp = args.atemp if args.atemp is not None else args.temp
    raw_row = pd.DataFrame([{
        "instant": 0,
        "dteday": pd.Timestamp("2012-01-01"),  # placeholder, dropped before predict
        "season": args.season,
        "yr": args.year,
        "mnth": args.month,
        "hr": args.hour,
        "holiday": args.holiday,
        "weekday": args.weekday,
        "workingday": args.workingday,
        "weathersit": args.weather,
        "temp": args.temp,
        "atemp": atemp,
        "hum": args.humidity,
        "windspeed": args.windspeed,
        "casual": 0,
        "registered": 0,
        "cnt": 0,  # placeholder target, unused
    }])
    return build_features(raw_row)


def main():
    args = parse_args()
    config = load_config()

    with open(config["output"]["metrics_path"]) as f:
        saved = json.load(f)
    feature_cols = saved["feature_columns"]

    model = joblib.load(config["output"]["model_path"])

    row = build_input_row(args)
    # Align to the exact training columns/order; any one-hot category not
    # present in this single row (e.g. other seasons) becomes 0.
    row = row.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(row)[0]
    print(f"Predicted rides for hour {args.hour}: {round(prediction)}")


if __name__ == "__main__":
    main()
