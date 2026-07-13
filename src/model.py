from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

SOURCE_COLUMN_MAP = {
    "Overall Qual": "overall_quality",
    "Overall Cond": "overall_condition",
    "Gr Liv Area": "living_area_sqft",
    "Bedroom AbvGr": "bedrooms",
    "Full Bath": "full_bathrooms",
    "Year Built": "year_built",
    "Garage Cars": "garage_capacity",
    "Total Bsmt SF": "basement_sqft",
    "Neighborhood": "neighborhood",
    "Bldg Type": "building_type",
    "Kitchen Qual": "kitchen_quality",
    "SalePrice": "sale_price",
    "Yr Sold": "sale_year",
}


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


@dataclass
class ValuationModelBundle:
    preprocessor: Any
    point_model: GradientBoostingRegressor
    lower_model: GradientBoostingRegressor
    upper_model: GradientBoostingRegressor
    neighborhoods: list[str]
    building_types: list[str]
    kitchen_qualities: list[str]
    training_years: tuple[int, int]
    test_year: int

    def predict(self, features: PropertyFeatures) -> ValuationRange:
        transformed = self.preprocessor.transform(features.to_frame())
        point = float(np.expm1(self.point_model.predict(transformed)[0]))
        raw_lower = float(np.expm1(self.lower_model.predict(transformed)[0]))
        raw_upper = float(np.expm1(self.upper_model.predict(transformed)[0]))

        lower = max(0.0, min(raw_lower, point))
        upper = max(point, raw_upper)
        return ValuationRange(lower=lower, expected=point, upper=upper)


def prepare_training_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    required_columns = list(SOURCE_COLUMN_MAP)
    missing = sorted(set(required_columns) - set(raw_data.columns))
    if missing:
        raise ValueError(f"Ames dataset is missing required columns: {missing}")

    data = raw_data[required_columns].rename(columns=SOURCE_COLUMN_MAP).copy()
    data = data.dropna(subset=["sale_price", "sale_year"])
    data["sale_year"] = data["sale_year"].astype(int)
    return data


def _make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=True,
    )
