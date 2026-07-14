from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def _make_regressor(loss: str, *, alpha: float = 0.9) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss=loss,
        alpha=alpha,
        n_estimators=300,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=8,
        random_state=436,
    )


def train_valuation_model(
    raw_data: pd.DataFrame,
) -> tuple[ValuationModelBundle, dict[str, Any]]:
    data = prepare_training_data(raw_data)
    test_year = int(data["sale_year"].max())
    train_data = data[data["sale_year"] < test_year]
    test_data = data[data["sale_year"] == test_year]
    if train_data.empty or test_data.empty:
        raise ValueError("A time-based train/test split could not be created.")

    preprocessor = _make_preprocessor()
    x_train = preprocessor.fit_transform(train_data[MODEL_FEATURES])
    x_test = preprocessor.transform(test_data[MODEL_FEATURES])
    y_train = train_data["sale_price"].to_numpy(dtype=float)
    y_test = test_data["sale_price"].to_numpy(dtype=float)
    log_y_train = np.log1p(y_train)

    point_model = _make_regressor("huber")
    lower_model = _make_regressor("quantile", alpha=0.10)
    upper_model = _make_regressor("quantile", alpha=0.90)
    point_model.fit(x_train, log_y_train)
    lower_model.fit(x_train, log_y_train)
    upper_model.fit(x_train, log_y_train)

    point_predictions = np.expm1(point_model.predict(x_test))
    lower_predictions = np.expm1(lower_model.predict(x_test))
    upper_predictions = np.expm1(upper_model.predict(x_test))
    ordered_lower = np.minimum(lower_predictions, point_predictions)
    ordered_upper = np.maximum(upper_predictions, point_predictions)

    baseline_prediction = np.full_like(y_test, np.median(y_train))
    feature_names = preprocessor.get_feature_names_out()
    ranked_features = sorted(
        zip(feature_names, point_model.feature_importances_, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )

    metrics: dict[str, Any] = {
        "data_source": "Ames Housing, Journal of Statistics Education",
        "source_url": "https://jse.amstat.org/v19n3/decock/AmesHousing.txt",
        "training_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "training_years": [
            int(train_data["sale_year"].min()),
            int(train_data["sale_year"].max()),
        ],
        "test_year": test_year,
        "mae": float(mean_absolute_error(y_test, point_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, point_predictions))),
        "r2": float(r2_score(y_test, point_predictions)),
        "mape_pct": float(np.mean(np.abs((y_test - point_predictions) / y_test)) * 100),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_prediction)),
        "interval_coverage_pct": float(
            np.mean((y_test >= ordered_lower) & (y_test <= ordered_upper)) * 100
        ),
        "top_features": [
            {
                "feature": name.replace("numeric__", "").replace("categorical__", ""),
                "importance": float(importance),
            }
            for name, importance in ranked_features[:8]
        ],
    }

    bundle = ValuationModelBundle(
        preprocessor=preprocessor,
        point_model=point_model,
        lower_model=lower_model,
        upper_model=upper_model,
        neighborhoods=sorted(data["neighborhood"].dropna().astype(str).unique()),
        building_types=sorted(data["building_type"].dropna().astype(str).unique()),
        kitchen_qualities=sorted(data["kitchen_quality"].dropna().astype(str).unique()),
        training_years=(
            int(train_data["sale_year"].min()),
            int(train_data["sale_year"].max()),
        ),
        test_year=test_year,
    )
    return bundle, metrics


def save_model(bundle: ValuationModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_model(path: Path) -> ValuationModelBundle:
    model = joblib.load(path)
    if not isinstance(model, ValuationModelBundle):
        raise TypeError(f"Unexpected model artifact type: {type(model)!r}")
    return model
