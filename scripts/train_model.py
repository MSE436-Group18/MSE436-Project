from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import save_model, train_valuation_model  # noqa: E402

DATA_PATH = ROOT / "data" / "ames_housing.tsv"
MODEL_PATH = ROOT / "artifacts" / "valuation_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "model_metrics.json"
DATA_URL = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading Ames Housing data from {DATA_URL}")
        urlretrieve(DATA_URL, DATA_PATH)

    raw_data = pd.read_csv(DATA_PATH, sep="\t")
    model, metrics = train_valuation_model(raw_data)
    save_model(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(
        "Time-holdout performance: "
        f"MAE=${metrics['mae']:,.0f}, "
        f"R2={metrics['r2']:.3f}, "
        f"80% interval coverage={metrics['interval_coverage_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
