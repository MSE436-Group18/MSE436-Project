from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

ValuationBasis = Literal["Conservative", "Expected", "Optimistic"]

NUMERIC_FEATURES = [
    "living_area_sqft",
    "bedrooms",
    "bathrooms",
    "lot_size_acres",
]

CATEGORICAL_FEATURES = [
    "state",
    "city",
    "zip_code",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SOURCE_COLUMN_MAP = {
    "status": "listing_status",
    "price": "listing_price",
    "bed": "bedrooms",
    "bath": "bathrooms",
    "acre_lot": "lot_size_acres",
    "street": "property_id",
    "city": "city",
    "state": "state",
    "zip_code": "zip_code",
    "house_size": "living_area_sqft",
}

DATA_SOURCE_NAME = "USA Real Estate Dataset (Realtor.com listings via Kaggle)"
DATA_SOURCE_URL = (
    "https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset"
)


@dataclass(frozen=True)
class PropertyFeatures:
    living_area_sqft: int
    bedrooms: int
    bathrooms: float
    lot_size_acres: float
    state: str
    city: str
    zip_code: str

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
    point_model: HistGradientBoostingRegressor
    lower_model: HistGradientBoostingRegressor
    upper_model: HistGradientBoostingRegressor
    locations: dict[str, dict[str, list[str]]]
    training_rows: int
    test_rows: int

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
        raise ValueError(f"USA Real Estate dataset is missing required columns: {missing}")

    data = raw_data[required_columns].rename(columns=SOURCE_COLUMN_MAP).copy()
    numeric_columns = [
        "listing_price",
        "bedrooms",
        "bathrooms",
        "lot_size_acres",
        "living_area_sqft",
        "property_id",
        "zip_code",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    valid_rows = (
        data["listing_status"].eq("for_sale")
        & data["listing_price"].between(20_000, 5_000_000)
        & data["bedrooms"].between(1, 10)
        & data["bathrooms"].between(1, 10)
        & data["living_area_sqft"].between(300, 10_000)
        & (data["lot_size_acres"].isna() | data["lot_size_acres"].between(0, 20))
        & data["state"].notna()
        & data["city"].notna()
        & data["zip_code"].notna()
    )
    data = data.loc[valid_rows].copy()
    data["state"] = data["state"].astype(str).str.strip()
    data["city"] = data["city"].astype(str).str.strip()
    data["zip_code"] = data["zip_code"].round().astype("Int64").astype("string").str.zfill(5)
    property_id = data["property_id"].round().astype("Int64").astype("string")
    fallback_id = "row-" + data.index.astype(str)
    property_id = property_id.fillna(pd.Series(fallback_id, index=data.index, dtype="string"))
    data["property_key"] = data["state"] + "|" + data["zip_code"] + "|" + property_id
    data = data.drop_duplicates(subset=["property_key", *MODEL_FEATURES, "listing_price"])
    return data[[*MODEL_FEATURES, "listing_price", "property_key"]]


def _make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                TargetEncoder(target_type="continuous", random_state=436),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def _make_regressor(
    loss: Literal["squared_error", "quantile"],
    *,
    quantile: float | None = None,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=436,
    )


def _build_location_options(data: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    locations: dict[str, dict[str, list[str]]] = {}
    rows = (
        data[["state", "city", "zip_code"]]
        .drop_duplicates()
        .sort_values(CATEGORICAL_FEATURES)
    )
    for row in rows.itertuples(index=False):
        state_locations = locations.setdefault(str(row.state), {})
        city_zip_codes = state_locations.setdefault(str(row.city), [])
        city_zip_codes.append(str(row.zip_code))
    return locations


def train_valuation_model(
    raw_data: pd.DataFrame,
    *,
    prepared: bool = False,
    source_rows: int | None = None,
    usable_source_rows: int | None = None,
) -> tuple[ValuationModelBundle, dict[str, Any]]:
    data = raw_data.copy() if prepared else prepare_training_data(raw_data)
    del raw_data
    if data.empty:
        raise ValueError("No usable for-sale residential listings remained after cleaning.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=436)
    train_indices, test_indices = next(
        splitter.split(data[MODEL_FEATURES], groups=data["property_key"])
    )
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    if train_data.empty or test_data.empty:
        raise ValueError("A property-grouped train/test split could not be created.")

    preprocessor = _make_preprocessor()
    y_train = train_data["listing_price"].to_numpy(dtype=float)
    y_test = test_data["listing_price"].to_numpy(dtype=float)
    log_y_train = np.log1p(y_train)
    x_train = preprocessor.fit_transform(train_data[MODEL_FEATURES], log_y_train)
    x_test = preprocessor.transform(test_data[MODEL_FEATURES])

    point_model = _make_regressor("squared_error")
    lower_model = _make_regressor("quantile", quantile=0.10)
    upper_model = _make_regressor("quantile", quantile=0.90)
    point_model.fit(x_train, log_y_train)
    lower_model.fit(x_train, log_y_train)
    upper_model.fit(x_train, log_y_train)

    point_predictions = np.expm1(point_model.predict(x_test))
    lower_predictions = np.expm1(lower_model.predict(x_test))
    upper_predictions = np.expm1(upper_model.predict(x_test))
    ordered_lower = np.minimum(lower_predictions, point_predictions)
    ordered_upper = np.maximum(upper_predictions, point_predictions)

    baseline_prediction = np.full_like(y_test, np.median(y_train))
    importance_size = min(10_000, len(test_data))
    importance_rng = np.random.default_rng(436)
    importance_indices = importance_rng.choice(len(test_data), size=importance_size, replace=False)
    importance = permutation_importance(
        point_model,
        x_test[importance_indices],
        np.log1p(y_test[importance_indices]),
        n_repeats=3,
        random_state=436,
        scoring="neg_mean_absolute_error",
    )
    ranked_features = sorted(
        zip(MODEL_FEATURES, importance.importances_mean, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )

    metrics: dict[str, Any] = {
        "data_source": DATA_SOURCE_NAME,
        "source_url": DATA_SOURCE_URL,
        "source_rows": int(source_rows if source_rows is not None else len(data)),
        "usable_source_rows": int(
            usable_source_rows if usable_source_rows is not None else len(data)
        ),
        "model_sample_rows": int(len(data)),
        "training_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "split_strategy": "Property-grouped random 80/20 holdout",
        "mae": float(mean_absolute_error(y_test, point_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, point_predictions))),
        "r2": float(r2_score(y_test, point_predictions)),
        "mape_pct": float(np.mean(np.abs((y_test - point_predictions) / y_test)) * 100),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_prediction)),
        "interval_coverage_pct": float(
            np.mean((y_test >= ordered_lower) & (y_test <= ordered_upper)) * 100
        ),
        "top_features": [
            {"feature": name, "importance": float(feature_importance)}
            for name, feature_importance in ranked_features
        ],
    }

    bundle = ValuationModelBundle(
        preprocessor=preprocessor,
        point_model=point_model,
        lower_model=lower_model,
        upper_model=upper_model,
        locations=_build_location_options(data),
        training_rows=len(train_data),
        test_rows=len(test_data),
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
