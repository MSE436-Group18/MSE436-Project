from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import (  # noqa: E402
    MODEL_FEATURES,
    SOURCE_COLUMN_MAP,
    prepare_training_data,
    save_model,
    train_valuation_model,
)

DATA_PATH = ROOT / "data" / "usa-real-estate-dataset.zip"
MODEL_PATH = ROOT / "artifacts" / "valuation_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "model_metrics.json"
DATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "ahmedshahriarsakib/usa-real-estate-dataset"
)
MAX_MODEL_ROWS = 300_000
CHUNK_SIZE = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Brickwise's listing-price model.")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download the latest Kaggle dataset version before training.",
    )
    return parser.parse_args()


def download_dataset() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = DATA_PATH.with_suffix(".download")
    print(f"Downloading USA Real Estate data from {DATA_URL}")
    urlretrieve(DATA_URL, temporary_path)
    temporary_path.replace(DATA_PATH)


def load_training_sample() -> tuple[pd.DataFrame, int, int]:
    rng = np.random.default_rng(436)
    reservoir = pd.DataFrame()
    source_rows = 0
    usable_source_rows = 0

    chunks = pd.read_csv(
        DATA_PATH,
        usecols=list(SOURCE_COLUMN_MAP),
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    for chunk in chunks:
        source_rows += len(chunk)
        prepared = prepare_training_data(chunk)
        usable_source_rows += len(prepared)
        prepared["_sample_key"] = rng.random(len(prepared))
        reservoir = pd.concat([reservoir, prepared], ignore_index=True)
        reservoir = reservoir.drop_duplicates(
            subset=["property_key", *MODEL_FEATURES, "listing_price"]
        )
        if len(reservoir) > MAX_MODEL_ROWS:
            reservoir = reservoir.nsmallest(MAX_MODEL_ROWS, "_sample_key")

    if reservoir.empty:
        raise ValueError("The downloaded dataset contains no usable residential listings.")
    sample = reservoir.drop(columns="_sample_key").reset_index(drop=True)
    return sample, source_rows, usable_source_rows


def main() -> None:
    args = parse_args()
    if args.refresh_data or not DATA_PATH.exists():
        download_dataset()

    training_sample, source_rows, usable_source_rows = load_training_sample()
    model, metrics = train_valuation_model(
        training_sample,
        prepared=True,
        source_rows=source_rows,
        usable_source_rows=usable_source_rows,
    )
    save_model(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(
        f"Processed {source_rows:,} source rows; "
        f"{usable_source_rows:,} passed cleaning; "
        f"{len(training_sample):,} were sampled for modelling."
    )
    print(
        "Property-grouped holdout performance: "
        f"MAE=${metrics['mae']:,.0f}, "
        f"R2={metrics['r2']:.3f}, "
        f"80% interval coverage={metrics['interval_coverage_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
