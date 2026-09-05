"""
Evaluation framework: chronological splits, naive baselines, and metrics.

Built BEFORE any model, on purpose. Every model we build later gets judged
against the baselines defined here — if it can't beat them, it isn't earning
its complexity.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Splits a time-indexed feature table into train/val/test IN DATE ORDER.
    Never shuffle time series data — a random split leaks future information
    into training (e.g. training on Thursday's data to predict Wednesday).

    Returns (train_df, val_df, test_df).
    """
    df = df.sort_index()
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def pooled_chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Like chronological_split, but for a POOLED multi-ticker table where the
    same date appears once per ticker. Splitting by row-fraction would be
    wrong here (tickers interleave), so instead we pick cutoff DATES from
    the unique sorted date range, then filter every ticker's rows by date.

    This guarantees no row in val/test has a date that any row in train
    saw for ANY ticker -- the split is leakage-safe across the whole pool.
    """
    unique_dates = pd.Series(df.index.unique()).sort_values().reset_index(drop=True)
    n = len(unique_dates)
    train_end_date = unique_dates.iloc[int(n * train_frac)]
    val_end_date = unique_dates.iloc[int(n * (train_frac + val_frac))]

    train_df = df[df.index <= train_end_date]
    val_df = df[(df.index > train_end_date) & (df.index <= val_end_date)]
    test_df = df[df.index > val_end_date]

    return train_df, val_df, test_df


def naive_price_baseline(df: pd.DataFrame) -> pd.Series:
    """
    The baseline every regression model must beat: "tomorrow's price = today's price".
    If a model can't beat this, it's adding noise, not signal.
    """
    return df["Close"]


def naive_direction_baseline(train_df: pd.DataFrame, n: int) -> np.ndarray:
    """
    The baseline every classifier must beat: always predict the majority class
    observed in training. For stocks this is usually "up" (markets drift upward
    slightly more often than not) — which makes it a sneaky-strong baseline that
    a lot of naive "AI stock predictors" quietly fail to beat.
    """
    majority_class = train_df["Target_Direction"].mode()[0]
    return np.full(n, majority_class)


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, and MAPE — the standard trio for price-prediction error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape}


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, precision, recall, F1 — the standard trio for direction prediction."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_against_baseline(test_df: pd.DataFrame, train_df: pd.DataFrame,
                              model_price_preds=None, model_direction_preds=None) -> dict:
    """
    Runs both naive baselines on test_df and, if provided, compares a real
    model's predictions against them side by side. This is the report you'll
    actually want to screenshot for your resume/portfolio write-up.
    """
    results = {}

    # --- Regression: naive baseline ---
    naive_preds = naive_price_baseline(test_df)
    results["naive_regression"] = regression_metrics(test_df["Target_Price"], naive_preds)

    # --- Classification: naive baseline ---
    naive_dir_preds = naive_direction_baseline(train_df, len(test_df))
    results["naive_classification"] = classification_metrics(test_df["Target_Direction"], naive_dir_preds)

    if model_price_preds is not None:
        results["model_regression"] = regression_metrics(test_df["Target_Price"], model_price_preds)

    if model_direction_preds is not None:
        results["model_classification"] = classification_metrics(test_df["Target_Direction"], model_direction_preds)

    return results