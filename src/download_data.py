"""Download and extract the UCI Bike Sharing Dataset.

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Usage:
    python -m src.download_data
"""

import io
import zipfile
from pathlib import Path

import requests
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_data(config: dict) -> Path:
    """Download the dataset zip and extract hour.csv into data/raw/.

    Idempotent: if the target CSV already exists, does nothing and returns
    its path immediately.
    """
    raw_dir = Path(config["data"]["raw_dir"])
    raw_csv = Path(config["data"]["raw_csv"])

    if raw_csv.exists():
        print(f"Dataset already present at {raw_csv}, skipping download.")
        return raw_csv

    raw_dir.mkdir(parents=True, exist_ok=True)

    url = config["data"]["url"]
    print(f"Downloading dataset from {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # The archive contains Readme.txt, day.csv, hour.csv — we only need hour.csv.
        with zf.open("hour.csv") as src, open(raw_csv, "wb") as dst:
            dst.write(src.read())

    print(f"Saved hourly data to {raw_csv}")
    return raw_csv


if __name__ == "__main__":
    cfg = load_config()
    download_data(cfg)
