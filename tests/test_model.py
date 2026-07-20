from pathlib import Path

import pandas as pd

from src.model import PropertyFeatures, load_model, prepare_training_data

ROOT = Path(__file__).resolve().parents[1]


def sample_property(living_area_sqft: int) -> PropertyFeatures:
    return PropertyFeatures(
        living_area_sqft=living_area_sqft,
        bedrooms=3,
        bathrooms=2.0,
        lot_size_acres=0.15,
        state="Texas",
        city="Houston",
        zip_code="77002",
    )


def test_model_returns_an_ordered_positive_range() -> None:
    model = load_model(ROOT / "artifacts" / "valuation_model.joblib")
    valuation = model.predict(sample_property(living_area_sqft=1850))

    assert 0 < valuation.lower <= valuation.expected <= valuation.upper


def test_living_area_changes_the_model_output() -> None:
    model = load_model(ROOT / "artifacts" / "valuation_model.joblib")
    smaller_property = model.predict(sample_property(living_area_sqft=1000))
    larger_property = model.predict(sample_property(living_area_sqft=3000))

    assert larger_property.expected > smaller_property.expected


def test_training_data_keeps_only_plausible_for_sale_homes() -> None:
    raw = pd.DataFrame(
        {
            "status": ["for_sale", "sold", "for_sale"],
            "price": [350_000, 330_000, 9_000_000],
            "bed": [3, 3, 3],
            "bath": [2, 2, 2],
            "acre_lot": [0.2, 0.2, 0.2],
            "street": [101, 102, 103],
            "city": ["Houston", "Houston", "Houston"],
            "state": ["Texas", "Texas", "Texas"],
            "zip_code": [77002, 77002, 77002],
            "house_size": [1800, 1800, 1800],
        }
    )

    cleaned = prepare_training_data(raw)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["zip_code"] == "77002"
