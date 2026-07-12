from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

ValuationBasis = Literal["Conservative", "Expected", "Optimistic"]

NUMERIC_FEATURES = [
    "overall_quality",
    "overall_condition",
    "living_area_sqft",
    "bedrooms",
    "full_bathrooms",
    "year_built",
    "garage_capacity",
    "basement_sqft",
]

CATEGORICAL_FEATURES = [
    "neighborhood",
    "building_type",
    "kitchen_quality",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class PropertyFeatures:
    overall_quality: int
    overall_condition: int
    living_area_sqft: int
    bedrooms: int
    full_bathrooms: int
    year_built: int
    garage_capacity: float
    basement_sqft: float
    neighborhood: str
    building_type: str
    kitchen_quality: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)], columns=MODEL_FEATURES)


@dataclass(frozen=True)
class ValuationRange:
    lower: float
    expected: float
    upper: float

    def adjusted(self, multiplier: float) -> ValuationRange:
        if multiplier <= 0:
            raise ValueError("Market multiplier must be positive.")
        return ValuationRange(
            lower=self.lower * multiplier,
            expected=self.expected * multiplier,
            upper=self.upper * multiplier,
        )

    def select(self, basis: ValuationBasis) -> float:
        values = {
            "Conservative": self.lower,
            "Expected": self.expected,
            "Optimistic": self.upper,
        }
        return values[basis]
