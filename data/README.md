# Data

The training script downloads the original Ames Housing tab-separated dataset from the
Journal of Statistics Education:

- Data: <https://jse.amstat.org/v19n3/decock/AmesHousing.txt>
- Documentation: <https://jse.amstat.org/v19n3/decock/DataDocumentation.txt>
- Paper: <https://doi.org/10.1080/10691898.2011.11889627>

The raw file is intentionally not committed. Run `python scripts/train_model.py` to download it
and rebuild the checked-in model artifact. The dataset contains 2,930 residential sales in Ames,
Iowa from 2006 through 2010. It is suitable for a working valuation prototype, but it is not a
current market feed and should not be treated as investment advice.
