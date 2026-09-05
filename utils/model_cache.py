"""
Per-ticker model cache.

Fixes the original app.py's biggest bug: the old code loaded ONE model
(trained once on AAPL) and used it to predict every ticker a user typed
in, regardless of what stock it actually was. Here, each ticker gets its
own model, trained on its own data, cached to disk so we don't retrain
on every API request (retraining the LSTM takes real time).

Cache is time-based: a model older than CACHE_MAX_AGE_SECONDS is retrained
automatically on the next request for that ticker, so predictions stay
reasonably current without manual intervention.
"""
import os
import time
import json
import joblib
import pandas as pd
from tensorflow import keras

from utils.data_fetcher import fetch_ohlcv, get_company_name
from utils.sentiment import fetch_daily_sentiment
from utils.feature_engineering import build_feature_table, build_multiday_feature_table
from models.xgb_direction_model import train_direction_model
from models.lstm_price_model import train_lstm_price_model
from models.lstm_multiday_model import train_multiday_model

CACHE_DIR = "models/cache"
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # retrain at most once a day per ticker


def _fresh(path: str, max_age: int = CACHE_MAX_AGE_SECONDS) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age


def _ready(model_path: str, report_path: str, max_age: int = CACHE_MAX_AGE_SECONDS) -> bool:
    """
    A cached model is only usable as-is if BOTH the model file and its
    report exist and are fresh. Without this, a model cached before report-
    saving existed (or any run where the report write failed) would load
    silently forever without ever producing a report -- exactly the bug
    that made /model-info show "not yet trained" despite /predict and
    /explain having clearly already run successfully.
    """
    return _fresh(model_path, max_age) and os.path.exists(report_path)


def _save_report(path: str, report: dict):
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)


def get_cached_report(ticker: str, kind: str):
    """
    Reads back a previously-saved training report (direction/price/multiday)
    for `ticker`, so the frontend can show real backtested numbers instead
    of nothing. Returns None if that model hasn't been trained yet for this
    ticker -- the frontend should treat that as "not available yet", not
    an error (hitting /predict or /explain for that ticker will train it).
    """
    path = os.path.join(CACHE_DIR, f"{ticker}_{kind}_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_features(ticker: str, period: str = "2y", horizon: int = 1) -> pd.DataFrame:
    """Shared data-fetch + feature-build path used by every endpoint below."""
    df = fetch_ohlcv(ticker, period=period)
    sentiment = fetch_daily_sentiment(get_company_name(ticker))
    return build_feature_table(df, daily_sentiment=sentiment, horizon=horizon)


def get_multiday_features(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Same as get_features, but with all 5 horizon targets for multi-day forecasting."""
    df = fetch_ohlcv(ticker, period=period)
    sentiment = fetch_daily_sentiment(get_company_name(ticker))
    return build_multiday_feature_table(df, daily_sentiment=sentiment, horizons=(1, 2, 3, 4, 5))


def get_or_train_direction_model(ticker: str):
    """Returns a cached XGBoost direction model for `ticker`, training one if needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{ticker}_xgb.joblib")
    report_path = os.path.join(CACHE_DIR, f"{ticker}_direction_report.json")

    if _ready(path, report_path):
        return joblib.load(path)

    features = get_features(ticker, period="2y", horizon=1)
    model, report = train_direction_model(features, save_path=path)
    _save_report(report_path, report)
    print(f"[cache] trained direction model for {ticker} -- beats_baseline_f1={report['beats_baseline_f1']}")
    return model


def get_or_train_price_model(ticker: str):
    """Returns (model, feature_scaler, target_scaler) for `ticker`, training if needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, f"{ticker}_lstm.keras")
    fscaler_path = os.path.join(CACHE_DIR, f"{ticker}_lstm_fscaler.joblib")
    tscaler_path = os.path.join(CACHE_DIR, f"{ticker}_lstm_tscaler.joblib")
    report_path = os.path.join(CACHE_DIR, f"{ticker}_price_report.json")

    if _ready(model_path, report_path):
        model = keras.models.load_model(model_path)
    else:
        features = get_features(ticker, period="5y", horizon=1)
        model, report = train_lstm_price_model(
            features, seq_len=30, epochs=50,
            save_path=model_path, feature_scaler_path=fscaler_path, target_scaler_path=tscaler_path,
        )
        _save_report(report_path, report)
        print(f"[cache] trained price model for {ticker} -- beats_baseline_rmse={report['beats_baseline_rmse']}")

    feature_scaler = joblib.load(fscaler_path)
    target_scaler = joblib.load(tscaler_path)
    return model, feature_scaler, target_scaler


def get_or_train_multiday_model(ticker: str):
    """Returns (model, feature_scaler, target_scaler) for the 5-day direct multi-horizon model."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, f"{ticker}_lstm_multiday.keras")
    fscaler_path = os.path.join(CACHE_DIR, f"{ticker}_lstm_multiday_fscaler.joblib")
    tscaler_path = os.path.join(CACHE_DIR, f"{ticker}_lstm_multiday_tscaler.joblib")
    report_path = os.path.join(CACHE_DIR, f"{ticker}_multiday_report.json")

    if _ready(model_path, report_path):
        model = keras.models.load_model(model_path)
    else:
        features = get_multiday_features(ticker, period="5y")
        model, report = train_multiday_model(
            features, seq_len=30, epochs=50,
            save_path=model_path, feature_scaler_path=fscaler_path, target_scaler_path=tscaler_path,
        )
        _save_report(report_path, report)
        print(f"[cache] trained multiday model for {ticker} -- "
              f"day_1 beats_baseline={report['per_horizon']['day_1']['beats_baseline_rmse']}")

    feature_scaler = joblib.load(fscaler_path)
    target_scaler = joblib.load(tscaler_path)
    return model, feature_scaler, target_scaler