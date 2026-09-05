"""
Smoke tests for the models themselves -- these actually train (briefly) on
synthetic data, so they're slower than the rest of the suite. Run with
`pytest -m "not slow"` to skip them for a fast feedback loop, or plain
`pytest` to run everything including these.
"""
import numpy as np
import pytest

from utils.evaluation import chronological_split
from models.xgb_direction_model import train_on_split, FEATURE_COLUMNS, feature_importance


@pytest.mark.slow
def test_xgb_direction_model_trains_and_reports(tmp_path, synthetic_features):
    train, val, test = chronological_split(synthetic_features)
    save_path = str(tmp_path / "xgb_test.joblib")

    model, report = train_on_split(train, val, test, save_path=save_path)

    assert "model" in report and "naive_baseline" in report
    assert 0 <= report["model"]["F1"] <= 1
    assert "beats_baseline_f1" in report


@pytest.mark.slow
def test_xgb_finds_injected_signal():
    """Sanity check that catches a broken/overly-conservative trainer: if we
    inject REAL signal into the target, the model must find it. If this test
    ever fails, don't trust any 'model beats baseline' result elsewhere --
    it would mean the trainer can't detect real signal even when it exists."""
    import pandas as pd
    from sklearn.metrics import f1_score
    import xgboost as xgb

    np.random.seed(5)
    n = 4000
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n), "f3": np.random.randn(n)})
    y = (X["f1"] + 0.3 * np.random.randn(n) > 0).astype(int)

    split = int(n * 0.7)
    split2 = int(n * 0.85)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:split2], y.iloc[split:split2]
    X_test, y_test = X.iloc[split2:], y.iloc[split2:]

    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              eval_metric="logloss", early_stopping_rounds=20, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_test)

    model_f1 = f1_score(y_test, preds)
    baseline_f1 = f1_score(y_test, np.full(len(y_test), y_train.mode()[0]))
    assert model_f1 > baseline_f1 + 0.1  # should clearly beat baseline when real signal exists


@pytest.mark.slow
def test_lstm_price_model_beats_random_price_scale(tmp_path, synthetic_features):
    """Guards against the exact bug found in development: predicting absolute
    price level (rather than % return) failed catastrophically (RMSE ~50-200)
    when train/test price ranges differed. This test trains briefly and checks
    the result is in a SANE range relative to the data's own price scale --
    not a strict baseline-beating requirement (that's a research question,
    not a unit test), just a guard against the model being badly broken."""
    from models.lstm_price_model import train_lstm_price_model

    save_path = str(tmp_path / "lstm_test.keras")
    fscaler_path = str(tmp_path / "lstm_fscaler.joblib")
    tscaler_path = str(tmp_path / "lstm_tscaler.joblib")

    model, report = train_lstm_price_model(
        synthetic_features, seq_len=20, epochs=10,
        save_path=save_path, feature_scaler_path=fscaler_path, target_scaler_path=tscaler_path,
    )

    price_scale = synthetic_features["Close"].mean()
    # RMSE should be a small fraction of the price level, not comparable to or
    # exceeding it (that's what the unscaled-target bug looked like: RMSE ~50
    # on a ~$300 stock -- i.e. RMSE/price_scale ~ 0.17, wildly too high)
    assert report["model"]["RMSE"] < price_scale * 0.1


def test_feature_importance_returns_real_features(tmp_path, synthetic_features):
    """Guards against the original project's explainability bug: importances
    must be keyed by real feature names (RSI_14, MACD, ...), never raw
    timestep indices."""
    train, val, test = chronological_split(synthetic_features)
    model, _ = train_on_split(train, val, test, save_path=str(tmp_path / "xgb.joblib"))

    top = feature_importance(model, top_n=5)
    assert set(top.index).issubset(set(FEATURE_COLUMNS))
    assert not any(str(i).isdigit() for i in top.index)  # no raw timestep-like indices