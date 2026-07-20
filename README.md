# MSE436 Term Project: Brickwise

Brickwise is a human-in-the-loop decision support system for a small residential property investor deciding whether to **BUY**, **WATCH**, or **AVOID** a candidate listing.

The interface is load bearing, property facts change a trained valuation model,
while financing, operating, valuation, and risk controls change the investment recommendation.

Repository: <https://github.com/MSE436-Group18/MSE436-Project>

## Working demo

### Value play: BUY

![Value Play recommendation](screenshots/value-play-buy.png)

### Same property under a rate shock: AVOID

![Rate Shock recommendation](screenshots/rate-shock-avoid.png)

## Run locally

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The checked-in model artifact makes the demo runnable without a data download. To reproduce
the training pipeline from the cached source file—or refresh the public Kaggle snapshot first:

```bash
python scripts/train_model.py
python scripts/train_model.py --refresh-data
```

## What the system does

1. A gradient-boosted regression model predicts expected sale value from property features.
2. Two quantile models produce a lower and upper valuation estimate for an 80% model range.
3. The user selects a conservative, expected, or optimistic valuation basis.
4. A cash-flow engine projects mortgage payments, rent, vacancy, costs, equity, and sale proceeds.
5. The selected risk policy compares base and downside returns against explicit thresholds.
6. The interface explains exactly why the result is BUY, WATCH, or AVOID.

## Model evidence

The prototype uses the Kaggle USA Real Estate Dataset, sourced from Realtor.com: 2,226,382
records across U.S. states and ZIP codes. Cleaning retains plausible, complete for-sale residential
listings; a fixed-seed reservoir sample makes training reproducible and practical on a laptop.
The holdout is grouped by encoded property ID so the same property cannot leak across the split.

| Measure | Result |
| --- | ---: |
| Source rows processed | 2,226,382 |
| Rows passing cleaning rules | 880,541 |
| Reproducible model sample | 300,000 |
| Training / property-grouped holdout | 240,013 / 59,987 |
| Holdout MAE | $140,717 |
| Holdout R-squared | 0.714 |
| 80% interval coverage | 80.0% |
| Median-price baseline MAE | $297,520 |

Histogram gradient boosting fits this task because listing prices contain non-linear size and
location effects, while remaining fast enough for live what-if analysis. Target encoding handles
high-cardinality city and ZIP fields without a huge sparse matrix.

## Decision controls

- Model inputs: state, city, ZIP code, living area, beds, baths, and lot size.
- Model settings: valuation basis and an explicit market-index multiplier for source-snapshot
  versus current/local conditions.
- Scenario inputs: asking price, rent, mortgage rate, down payment, vacancy, rent growth,
  appreciation, maintenance, tax, insurance, and holding period.
- Decision settings: minimum annual return and conservative, balanced, or growth risk policy.

The charts are decision-specific: asking price is shown against the model valuation range, and
downside/base/upside returns are shown against the user's minimum return threshold.

## Verification

```bash
pytest
ruff check .
mypy
```

Tests cover mortgage calculations, scenario sensitivity, recommendation changes, valuation-basis
sensitivity, model output ordering, and property-feature sensitivity.

## Data source

<https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset>
