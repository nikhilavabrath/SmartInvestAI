"""
Tests for utils/data_fetcher.py -- the cleaning/validation layer.
No live network calls: fetch_ohlcv's yfinance call is mocked, so these
tests only verify OUR cleaning logic, not Yahoo's API itself.
"""
from unittest import mock
import numpy as np
import pandas as pd
import pytest

from utils.data_fetcher import fetch_ohlcv


def test_fetch_ohlcv_flattens_multiindex_columns(synthetic_ohlcv):
    """Reproduces the real bug: yfinance sometimes returns MultiIndex columns
    even for a single ticker, which breaks pandas_ta if not flattened."""
    multi_df = synthetic_ohlcv.copy()
    multi_df.columns = pd.MultiIndex.from_product([multi_df.columns, ["AAPL"]])

    with mock.patch("utils.data_fetcher._download_with_retry", return_value=multi_df):
        result = fetch_ohlcv("AAPL")

    assert not isinstance(result.columns, pd.MultiIndex)
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_fetch_ohlcv_drops_high_less_than_low_rows(synthetic_ohlcv):
    """A row where High < Low is corrupted data, not a real trading day -- must be dropped."""
    bad_df = synthetic_ohlcv.copy()
    bad_df.iloc[10, bad_df.columns.get_loc("High")] = 1.0
    bad_df.iloc[10, bad_df.columns.get_loc("Low")] = 999.0

    with mock.patch("utils.data_fetcher._download_with_retry", return_value=bad_df):
        result = fetch_ohlcv("AAPL")

    assert (result["High"] >= result["Low"]).all()


def test_fetch_ohlcv_raises_on_empty_response():
    with mock.patch("utils.data_fetcher._download_with_retry", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="No data returned"):
            fetch_ohlcv("BADTICKER")


def test_fetch_ohlcv_raises_on_too_little_history(synthetic_ohlcv):
    tiny_df = synthetic_ohlcv.head(10)
    with mock.patch("utils.data_fetcher._download_with_retry", return_value=tiny_df):
        with pytest.raises(ValueError, match="not enough history"):
            fetch_ohlcv("AAPL")


def test_get_company_name_caches_result():
    """Verifies the fix for the sentiment-search bug: company name is resolved
    once and cached, not re-fetched from yfinance on every call."""
    from utils.data_fetcher import get_company_name, _INFO_CACHE
    _INFO_CACHE.clear()

    with mock.patch("utils.data_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {"longName": "Indian Oil Corporation Limited"}
        name1 = get_company_name("IOC.NS")
        name2 = get_company_name("IOC.NS")

    assert name1 == "Indian Oil Corporation Limited"
    assert name2 == name1
    assert mock_ticker.call_count == 1  # second call must hit the cache, not yfinance again


def test_get_company_name_falls_back_to_ticker_on_failure():
    from utils.data_fetcher import get_company_name, _INFO_CACHE
    _INFO_CACHE.clear()

    with mock.patch("utils.data_fetcher.yf.Ticker", side_effect=Exception("network error")):
        name = get_company_name("UNKNOWNTICKER")

    assert name == "UNKNOWNTICKER"