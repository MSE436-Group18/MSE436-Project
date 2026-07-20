# Data

The valuation pipeline uses the **USA Real Estate Dataset** published on Kaggle and collected
from Realtor.com:

- Dataset page: <https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset>
- Programmatic access: `https://www.kaggle.com/api/v1/datasets/download/ahmedshahriarsakib/usa-real-estate-dataset`
- Raw structure: one CSV containing 2,226,382 listing records and 12 fields
- Fields used by the model: price, listing status, beds, baths, lot size, living area, city,
  state, ZIP code, and privacy-encoded street/property ID

The downloaded ZIP is intentionally gitignored because it is about 38 MB compressed. The
checked-in model artifact lets the demo run without downloading the source data.

## Rebuild or update

```bash
# Rebuild from the locally cached source ZIP
python scripts/train_model.py

# Download the latest Kaggle version, then rebuild
python scripts/train_model.py --refresh-data
```

The training job reads the large CSV in chunks, keeps only `for_sale` homes with plausible price,
bed, bath, living-area, lot-size, and location values, removes exact duplicates, then maintains a
fixed-seed 300,000-row reservoir sample. The validation split is grouped by the encoded property
identifier so duplicate observations of one property cannot appear in both training and holdout.

## Important limitations

- `price` is an advertised price or recently sold price, not a consistently verified closing price.
- The file has no listing timestamp, so a valid chronological backtest is not possible.
- It does not contain rent, property condition, property type, vacancy, or financing variables.
- Missing fields and geographic representation are uneven, and cleaning excludes luxury and
  unusual properties outside the documented ranges.
- Kaggle labels the license as "Other" and the publisher describes the data as educational-use
  material, so redistribution and any non-course use require a separate rights review.
- Kaggle is a refreshable snapshot, not a guaranteed live feed. A deployed system should schedule
  version checks/retraining and join current transaction, rent, vacancy, and interest-rate sources.
