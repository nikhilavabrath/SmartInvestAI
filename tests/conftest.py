"""
Shared fixtures for the test suite. Everything here uses SYNTHETIC data --
no live network calls to yfinance/NewsAPI in tests, so the suite runs fast,
deterministically, and offline (in CI, on a plane, wherever).
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def clear_data_fetcher_caches():
    """
    data_fetcher.py deliberately caches raw fetches and company-name lookups
    at module level (to reduce real Yahoo Finance calls in production -- see
    its docstring). That's correct in production, but it means test cases
    reusing the same ticker string (many use "AAPL" or "TEST") would leak
    cached state between tests without this reset. Autouse means every test
    gets a clean cache with no per-test opt-in needed.
    """
    from utils import data_fetcher
    data_fetcher._RAW_CACHE.clear()
    data_fetcher._INFO_CACHE.clear()
    yield
    data_fetcher._RAW_CACHE.clear()
    data_fetcher._INFO_CACHE.clear()


@pytest.fixture
def synthetic_ohlcv():
    """A clean, realistic-shaped OHLCV DataFrame -- 500 business days, mild trend."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 200 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.3,
        "High": close + abs(np.random.randn(n)) * 0.6,
        "Low": close - abs(np.random.randn(n)) * 0.6,
        "Close": close,
        "Volume": np.random.randint(5_000_000, 20_000_000, n),
    }, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def synthetic_ohlcv_with_zero_volume(synthetic_ohlcv):
    """Same as synthetic_ohlcv, but with a zero-volume day injected -- reproduces
    the real IOC.NS bug (inf from .pct_change() dividing by zero)."""
    df = synthetic_ohlcv.copy()
    df.iloc[100, df.columns.get_loc("Volume")] = 0
    return df


@pytest.fixture
def synthetic_features(synthetic_ohlcv):
    """A fully-built single-horizon feature table, ready for model training."""
    from utils.feature_engineering import build_feature_table
    return build_feature_table(synthetic_ohlcv, daily_sentiment=None, horizon=1)


@pytest.fixture
def synthetic_multiday_features(synthetic_ohlcv):
    """A fully-built multi-horizon feature table, for the 5-day model."""
    from utils.feature_engineering import build_multiday_feature_table
    return build_multiday_feature_table(synthetic_ohlcv, daily_sentiment=None, horizons=(1, 2, 3, 4, 5))