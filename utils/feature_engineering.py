"""
Feature engineering.

Turns raw OHLCV (+ optional sentiment) into a model-ready feature table.
This is the "analyzed time-series data to identify trends" piece.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta


def _clean_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some engineered features are ratios (Return_1d, Volume_Change, etc.)
    computed via .pct_change(), which divides by the PREVIOUS row's value.
    If that previous value was exactly 0 (a real zero-volume day, which
    does happen for thinly-traded stocks), the result is +/-inf, not NaN.
    XGBoost hard-rejects inf outright ("Input data contains inf") rather
    than treating it like a missing value, so it has to be caught here,
    upstream, not left for the model to choke on later.
    """
    return df.replace([np.inf, -np.inf], np.nan)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds trend, momentum, volatility and volume-based features.
    Each one captures something an LSTM trained on raw Close price alone
    has no way to see directly:

    - SMA/EMA: is price above or below its recent average (trend direction)
    - RSI: is the stock overbought/oversold (momentum)
    - MACD: is short-term momentum accelerating relative to long-term
    - Bollinger Bands: how far price has stretched from its normal range
    - Return_1d/5d: the actual % move, which is what we ultimately care about
    - Volume features: is unusual volume confirming or contradicting the price move
    """
    df = df.copy()

    # Trend
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.ema(length=20, append=True)

    # Momentum
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)

    # Volatility
    df.ta.bbands(length=20, append=True)
    df["Volatility_10"] = df["Close"].pct_change().rolling(10).std()

    # Returns — arguably the most important features; raw price level doesn't generalize
    # across tickers (a $50 stock and a $2000 stock aren't comparable), but % returns are.
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)

    # Volume
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_SMA_10"] = df["Volume"].rolling(10).mean()

    return df


def add_sentiment(df: pd.DataFrame, daily_sentiment: "pd.Series | None") -> pd.DataFrame:
    """
    Merges a daily sentiment score (indexed by date) into the feature frame.
    If no sentiment data is available (no API key configured yet), fills with
    0 (neutral) rather than dropping the column or erroring — this is what
    lets us build and test the whole pipeline before NewsAPI is wired in.
    """
    df = df.copy()
    if daily_sentiment is None or daily_sentiment.empty:
        df["Sentiment"] = 0.0
    else:
        df["Sentiment"] = daily_sentiment.reindex(df.index).ffill().fillna(0.0)
    return df


def add_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Adds prediction targets:
    - Target_Price: actual close price `horizon` trading days ahead (for reference/reconstruction)
    - Target_Return: (Target_Price - Close) / Close -- the STATIONARY version of the
      price target. A 1% move means the same thing whether a stock is at $150 or
      $325, whereas the raw price level drifts over years and doesn't generalize
      across a train/test split that spans a real trend. Models should predict
      this, not Target_Price directly.
    - Target_Direction: 1 if price rises by horizon, else 0 (classification target)
    """
    df = df.copy()
    df["Target_Price"] = df["Close"].shift(-horizon)
    df["Target_Return"] = (df["Target_Price"] - df["Close"]) / df["Close"]
    df["Target_Direction"] = (df["Target_Price"] > df["Close"]).astype(int)
    return df


def add_multi_horizon_targets(df: pd.DataFrame, horizons=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """
    Adds one Target_Return_h{N} column per horizon in `horizons` -- e.g.
    Target_Return_h1, Target_Return_h3, Target_Return_h5. Used for DIRECT
    multi-horizon forecasting: a model predicts all of these from the same
    input window in one pass, instead of predicting day 1 and feeding that
    prediction back in to get day 2 (autoregressive rollout, which
    compounds error -- a bad day-1 guess poisons every day after it).
    """
    df = df.copy()
    for h in horizons:
        df[f"Target_Return_h{h}"] = (df["Close"].shift(-h) - df["Close"]) / df["Close"]
    return df


def build_multiday_feature_table(df: pd.DataFrame, daily_sentiment=None, horizons=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Full pipeline for multi-horizon forecasting: features + sentiment + all horizon targets, cleaned."""
    out = add_technical_indicators(df)
    out = add_sentiment(out, daily_sentiment)
    out = add_multi_horizon_targets(out, horizons=horizons)
    out = _clean_infinities(out)
    out = out.dropna()
    return out


def build_feature_table(
        df: pd.DataFrame,
        daily_sentiment: "pd.Series | None" = None,
        horizon: int = 1,
) -> pd.DataFrame:
    """
    Full pipeline: raw OHLCV -> engineered features + sentiment + targets, cleaned.

    Two kinds of NaNs get dropped here, for different reasons:
    1. Warm-up NaNs at the START (e.g. SMA_50 needs 50 prior rows to exist at all)
    2. Look-ahead NaNs at the END (the last `horizon` rows have no future price yet)
    A third source -- +/-inf from a zero-volume or zero-price day breaking a
    .pct_change() ratio -- is converted to NaN too (see _clean_infinities),
    so all three get swept up by the same final dropna().
    """
    out = add_technical_indicators(df)
    out = add_sentiment(out, daily_sentiment)
    out = add_targets(out, horizon=horizon)
    out = _clean_infinities(out)
    out = out.dropna()
    return out