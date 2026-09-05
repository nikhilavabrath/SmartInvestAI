"""
Builds a POOLED feature table across multiple tickers, so one model learns
general market patterns instead of one stock's idiosyncrasies from a few
hundred rows.

Run this, then pass the result to models.xgb_direction_model.train_on_split
using utils.evaluation.pooled_chronological_split.
"""
import pandas as pd

from utils.data_fetcher import fetch_ohlcv
from utils.sentiment import fetch_daily_sentiment
from utils.feature_engineering import build_feature_table

# A reasonably diverse starting universe -- large, liquid, well-covered-by-news
# tickers across a few sectors, so the model isn't just learning "tech stock behavior".
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "V", "JNJ", "PG", "XOM",
]


def build_pooled_features(tickers=None, period: str = "5y", horizon: int = 1) -> pd.DataFrame:
    """
    Fetches + engineers features for each ticker independently (so no
    cross-ticker leakage in indicator calculations), tags each row with its
    ticker, and concatenates into one table sorted by date.

    A ticker that fails to fetch (bad symbol, rate limit, etc.) is skipped
    with a printed warning rather than crashing the whole run -- useful
    when pulling 10+ tickers where one flaky request shouldn't ruin the rest.
    """
    tickers = tickers or DEFAULT_TICKERS
    frames = []

    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period=period)
            sentiment = fetch_daily_sentiment(ticker)
            features = build_feature_table(df, daily_sentiment=sentiment, horizon=horizon)
            features["Ticker"] = ticker
            frames.append(features)
            print(f"  {ticker}: {len(features)} rows")
        except Exception as e:
            print(f"  {ticker}: SKIPPED ({e})")

    if not frames:
        raise RuntimeError("No tickers succeeded -- nothing to pool.")

    pooled = pd.concat(frames).sort_index()
    return pooled