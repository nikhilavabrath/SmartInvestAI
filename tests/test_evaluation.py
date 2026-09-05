"""
Tests for utils/evaluation.py -- the chronological splitting and baseline
logic that everything else in this project is judged against. These tests
matter more than most: a bug here would mean every reported metric in the
whole project is untrustworthy.
"""
import numpy as np
import pandas as pd

from utils.evaluation import (
    chronological_split, pooled_chronological_split,
    regression_metrics, classification_metrics,
    naive_price_baseline, naive_direction_baseline,
)


def test_chronological_split_has_no_leakage(synthetic_features):
    train, val, test = chronological_split(synthetic_features)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_chronological_split_preserves_all_rows(synthetic_features):
    train, val, test = chronological_split(synthetic_features)
    assert len(train) + len(val) + len(test) == len(synthetic_features)


def test_pooled_chronological_split_has_no_leakage_across_tickers(synthetic_features):
    """The pooled split must guarantee no date overlap across the WHOLE pool,
    not just within one ticker -- this is what makes multi-ticker training
    leakage-safe."""
    a = synthetic_features.copy()
    a["Ticker"] = "A"
    b = synthetic_features.copy()
    b["Ticker"] = "B"
    pooled = pd.concat([a, b]).sort_index()

    train, val, test = pooled_chronological_split(pooled)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    # both tickers should be represented in each split, not accidentally dropped
    assert set(train["Ticker"].unique()) == {"A", "B"}


def test_regression_metrics_are_zero_for_perfect_predictions():
    y = np.array([100.0, 101.0, 99.0, 102.0])
    metrics = regression_metrics(y, y)
    assert metrics["RMSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["MAPE_%"] == 0


def test_classification_metrics_are_perfect_for_perfect_predictions():
    y = np.array([1, 0, 1, 1, 0])
    metrics = classification_metrics(y, y)
    assert metrics["Accuracy"] == 1.0
    assert metrics["F1"] == 1.0


def test_naive_price_baseline_is_just_close(synthetic_features):
    baseline = naive_price_baseline(synthetic_features)
    assert (baseline == synthetic_features["Close"]).all()


def test_naive_direction_baseline_is_majority_class(synthetic_features):
    train, val, test = chronological_split(synthetic_features)
    baseline = naive_direction_baseline(train, len(test))
    expected_majority = train["Target_Direction"].mode()[0]
    assert (baseline == expected_majority).all()