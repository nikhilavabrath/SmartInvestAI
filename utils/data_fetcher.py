"""
Data acquisition layer.

Pulls OHLCV data from yfinance and returns a clean, validated DataFrame.
This is the "automated data cleaning and transformation pipeline" piece.
"""
import time
import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

_INFO_CACHE = {}
_INFO_TTL_SECONDS = 3600  # 1 hour -- company name/info barely changes


def _get_ticker_info(ticker: str) -> dict:
    """
    Fetches and caches yfinance's full .info dict for a ticker once, so
    get_company_name() and get_ticker_currency() share a single cached
    lookup instead of each hitting yfinance separately.
    """
    now = time.time()
    cached = _INFO_CACHE.get(ticker)
    if cached and (now - cached[0]) < _INFO_TTL_SECONDS:
        return cached[1]
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    _INFO_CACHE[ticker] = (now, info)
    return info


def get_company_name(ticker: str) -> str:
    """
    Resolves a ticker symbol to a company name suitable for a NEWS SEARCH
    query -- e.g. 'IOC.NS' -> 'Indian Oil Corporation Limited'. Exchange-
    suffixed tickers like 'IOC.NS' or 'RELIANCE.NS' are meaningless as a
    NewsAPI query (no article literally contains the string "IOC.NS"), so
    sentiment fetching needs the real company name, not the raw symbol --
    this was silently producing 0.0 sentiment for every non-US ticker.
    Falls back to the raw ticker if lookup fails, so sentiment degrades to
    "probably no results" rather than crashing the whole pipeline.
    """
    info = _get_ticker_info(ticker)
    return info.get("longName") or info.get("shortName") or ticker


CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "\u20b9", "GBP": "\u00a3", "EUR": "\u20ac",
    "JPY": "\u00a5", "CNY": "\u00a5", "AUD": "A$", "CAD": "C$",
}


def get_ticker_currency(ticker: str) -> dict:
    """
    Resolves a ticker's trading currency -- e.g. IOC.NS trades in INR, not
    USD. Was hardcoded to '$' everywhere in the frontend regardless of
    exchange, which is simply wrong for any non-US ticker. Returns both
    the raw ISO code and a display symbol, falling back to showing the
    code itself (e.g. "AED") for currencies without a common symbol.
    """
    info = _get_ticker_info(ticker)
    code = info.get("currency") or "USD"
    return {"code": code, "symbol": CURRENCY_SYMBOLS.get(code, code + " ")}

# Yahoo Finance's session/crumb token occasionally gets invalidated when
# several requests for the same ticker land in quick succession -- exactly
# what happens here, since the frontend fires /predict, /predict-week,
# /explain, and /candlestick-data in parallel, each of which needs raw
# OHLCV for the same ticker. Two mitigations: (1) retry transient failures,
# (2) a short-lived cache so those near-simultaneous calls reuse one fetch
# instead of hitting Yahoo 3-4 times independently within the same second.
_RAW_CACHE = {}
_CACHE_TTL_SECONDS = 600  # 10 minutes


def _download_with_retry(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
            if df is not None and not df.empty:
                return df
            last_error = ValueError(f"Empty data returned for '{ticker}'.")
        except Exception as e:
            last_error = e
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))  # brief backoff before retrying
    raise last_error


def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker and return a clean, single-level-column DataFrame.

    Handles three real yfinance quirks that the original code didn't:
    - yfinance sometimes returns MultiIndex columns (e.g. ('Close', 'AAPL'))
      even when you only ask for a single ticker -> we flatten to plain names.
    - Missing/garbled rows from exchange holidays, halts, or API hiccups.
    - Transient "Invalid Crumb" 401s from Yahoo's session handling under
      concurrent requests -> retried with backoff, and a short cache avoids
      re-triggering the same race in the first place.
    """
    cache_key = (ticker, period, interval)
    now = time.time()
    cached = _RAW_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        df = cached[1].copy()
    else:
        df = _download_with_retry(ticker, period, interval)
        _RAW_CACHE[cache_key] = (now, df.copy())

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    # --- Fix 1: flatten MultiIndex columns if yfinance gave us any ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns {missing} for '{ticker}'.")

    df = df[REQUIRED_COLUMNS].copy()

    # --- Fix 2: cleaning & validation (this is what was missing entirely) ---
    # Rows with no Close price are unusable for anything downstream.
    df = df.dropna(subset=["Close"])

    # Forward-fill small OHLC gaps (e.g. partial holiday sessions) rather than dropping.
    df[["Open", "High", "Low"]] = df[["Open", "High", "Low"]].ffill()
    df["Volume"] = df["Volume"].fillna(0)

    # Sanity check: a row where High < Low is corrupted data, not a real trading day.
    bad_rows = df["High"] < df["Low"]
    if bad_rows.any():
        df = df.loc[~bad_rows]

    df = df.sort_index()
    df.index.name = "Date"

    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} valid rows for '{ticker}' after cleaning — "
            "not enough history for feature engineering (need 50+ for SMA_50 warm-up)."
        )

    return df