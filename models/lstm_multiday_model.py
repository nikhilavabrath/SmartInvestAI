"""
Direct multi-horizon LSTM: predicts returns for the next 1-5 days in a
SINGLE forward pass from one input window, instead of feeding each day's
prediction back in as the next day's input (autoregressive rollout --
what the original /predict-week endpoint did). Autoregressive rollout
compounds error: a bad day-1 guess poisons every day that follows it.
Direct multi-horizon output means each day's forecast has independent
error instead of stacked error.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from utils.evaluation import chronological_split, regression_metrics
from models.lstm_price_model import SEQUENCE_FEATURES

HORIZONS = [1, 2, 3, 4, 5]
TARGET_COLUMNS = [f"Target_Return_h{h}" for h in HORIZONS]


def make_multiday_sequences(df: pd.DataFrame, feature_scaler: StandardScaler, seq_len: int = 30, fit_scaler: bool = False):
    """
    Same idea as make_sequences in lstm_price_model.py, but y is now a
    VECTOR per row (one return per horizon) instead of a single value.
    """
    feature_values = df[SEQUENCE_FEATURES].values
    scaled = feature_scaler.fit_transform(feature_values) if fit_scaler else feature_scaler.transform(feature_values)

    X, y, close_at_t = [], [], []
    targets = df[TARGET_COLUMNS].values
    closes = df["Close"].values
    for i in range(seq_len, len(df)):
        X.append(scaled[i - seq_len:i])
        y.append(targets[i])
        close_at_t.append(closes[i])

    return np.array(X), np.array(y), np.array(close_at_t)


def build_multiday_model(input_shape, n_horizons: int):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(n_horizons),  # one output unit per horizon, predicted together
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_multiday_model(
        features: pd.DataFrame,
        seq_len: int = 30,
        epochs: int = 50,
        save_path: str = "models/lstm_multiday.keras",
        feature_scaler_path: str = "models/lstm_multiday_fscaler.joblib",
        target_scaler_path: str = "models/lstm_multiday_tscaler.joblib",
):
    """
    Trains on chronological train/val/test split, evaluates EACH horizon
    separately against its own naive persistence baseline (predict "no
    change" over h days), and saves model + scalers.
    """
    train_df, val_df, test_df = chronological_split(features)

    feature_scaler = StandardScaler()
    X_train, y_train_raw, _ = make_multiday_sequences(train_df, feature_scaler, seq_len=seq_len, fit_scaler=True)
    X_val, y_val_raw, _ = make_multiday_sequences(val_df, feature_scaler, seq_len=seq_len, fit_scaler=False)
    X_test, y_test_raw, close_test = make_multiday_sequences(test_df, feature_scaler, seq_len=seq_len, fit_scaler=False)

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train_raw)
    y_val = target_scaler.transform(y_val_raw)

    model = build_multiday_model((seq_len, len(SEQUENCE_FEATURES)), n_horizons=len(HORIZONS))
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=32, callbacks=[early_stop], verbose=0,
    )

    test_preds_scaled = model.predict(X_test, verbose=0)
    test_preds_return = target_scaler.inverse_transform(test_preds_scaled)

    report = {
        "per_horizon": {},
        "epochs_trained": len(history.history["loss"]),
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    }

    for idx, h in enumerate(HORIZONS):
        pred_price_h = close_test * (1 + test_preds_return[:, idx])
        actual_price_h = close_test * (1 + y_test_raw[:, idx])
        naive_price_h = close_test  # persistence baseline: "no change" over h days

        model_metrics = regression_metrics(actual_price_h, pred_price_h)
        baseline_metrics = regression_metrics(actual_price_h, naive_price_h)

        report["per_horizon"][f"day_{h}"] = {
            "model": model_metrics,
            "naive_baseline": baseline_metrics,
            "beats_baseline_rmse": model_metrics["RMSE"] < baseline_metrics["RMSE"],
        }

    model.save(save_path)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)

    return model, report


def predict_next_week(model, feature_scaler: StandardScaler, target_scaler: StandardScaler,
                      features: pd.DataFrame, seq_len: int = 30):
    """
    Predicts prices for the next len(HORIZONS) trading days in ONE forward
    pass -- no autoregressive feedback, so no compounding error.
    Returns (predicted_prices: list[float], last_close: float, horizons: list[int]).
    """
    recent = features.tail(seq_len)
    if len(recent) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows of feature history; got {len(recent)}.")

    X = feature_scaler.transform(recent[SEQUENCE_FEATURES].values)
    X = X.reshape(1, seq_len, len(SEQUENCE_FEATURES))

    pred_scaled = model.predict(X, verbose=0)
    pred_returns = target_scaler.inverse_transform(pred_scaled)[0]

    last_close = float(recent["Close"].iloc[-1])
    predicted_prices = [float(last_close * (1 + r)) for r in pred_returns]

    return predicted_prices, last_close, HORIZONS