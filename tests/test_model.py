from pathlib import Path

from src.model import PropertyFeatures, load_model

ROOT = Path(__file__).resolve().parents[1]


def sample_property(overall_quality: int) -> PropertyFeatures:
    return PropertyFeatures(
        overall_quality=overall_quality,
        overall_condition=6,
        living_area_sqft=1850,
        bedrooms=3,
        full_bathrooms=2,
        year_built=2003,
        garage_capacity=2.0,
        basement_sqft=980.0,
        neighborhood="Somerst",
        building_type="1Fam",
        kitchen_quality="Gd",
    )


def test_model_returns_an_ordered_positive_range() -> None:
    model = load_model(ROOT / "artifacts" / "valuation_model.joblib")
    valuation = model.predict(sample_property(overall_quality=7))

    assert 0 < valuation.lower <= valuation.expected <= valuation.upper


def test_property_quality_changes_the_model_output() -> None:
    model = load_model(ROOT / "artifacts" / "valuation_model.joblib")
    lower_quality = model.predict(sample_property(overall_quality=5))
    higher_quality = model.predict(sample_property(overall_quality=8))

    assert higher_quality.expected > lower_quality.expected
