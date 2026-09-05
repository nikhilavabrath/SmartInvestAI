"""
Tests for utils/feature_engineering.py.
"""
import numpy as np
import pandas as pd

from utils.feature_engineering import build_feature_table, build_multiday_feature_table


def test_build_feature_table_has_no_infinities(synthetic_ohlcv_with_zero_volume):
    """Reproduces and guards against the real IOC.NS bug: a zero-volume day
    makes Volume_Change (.pct_change()) produce inf, which crashes XGBoost."""
    features = build_feature_table(synthetic_ohlcv_with_zero_volume, daily_sentiment=None, horizon=1)
    numeric = features.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values).any()


def test_build_feature_table_has_no_nulls(synthetic_features):
    """The final table should be fully clean -- warm-up NaNs, look-ahead NaNs,
    and inf-turned-NaN should all have been dropped."""
    assert not synthetic_features.isnull().values.any()


def test_build_feature_table_target_direction_matches_target_price(synthetic_features):
    """Target_Direction must always agree with whether Target_Price rose vs Close."""
    implied_direction = (synthetic_features["Target_Price"] > synthetic_features["Close"]).astype(int)
    assert (synthetic_features["Target_Direction"] == implied_direction).all()


def test_build_feature_table_target_return_reconstructs_price_exactly(synthetic_features):
    """Target_Return is defined as (Target_Price - Close) / Close -- reconstructing
    price via Close * (1 + Target_Return) must be exact, since the LSTM's price
    prediction depends on this reconstruction being correct."""
    reconstructed = synthetic_features["Close"] * (1 + synthetic_features["Target_Return"])
    assert np.allclose(reconstructed, synthetic_features["Target_Price"])


def test_sentiment_defaults_to_neutral_when_none_provided(synthetic_ohlcv):
    features = build_feature_table(synthetic_ohlcv, daily_sentiment=None, horizon=1)
    assert (features["Sentiment"] == 0.0).all()


def test_sentiment_merges_by_date(synthetic_ohlcv):
    sentiment = pd.Series(
        [0.5, -0.3],
        index=pd.to_datetime(["2024-06-03", "2024-06-04"]),
    )
    features = build_feature_table(synthetic_ohlcv, daily_sentiment=sentiment, horizon=1)
    # Forward-filled from the last known score, not left neutral, for dates after the series starts
    assert features.loc["2024-06-05":, "Sentiment"].iloc[0] != 0.0


def test_multiday_feature_table_has_all_horizon_targets(synthetic_multiday_features):
    for h in (1, 2, 3, 4, 5):
        assert f"Target_Return_h{h}" in synthetic_multiday_features.columns


def test_multiday_targets_grow_with_horizon_variance(synthetic_multiday_features):
    """Sanity check: a 5-day return should generally have higher variance than
    a 1-day return, since more time has passed for price to move."""
    var_1d = synthetic_multiday_features["Target_Return_h1"].var()
    var_5d = synthetic_multiday_features["Target_Return_h5"].var()
    assert var_5d > var_1d