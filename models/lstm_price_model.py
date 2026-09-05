"""
Multivariate LSTM price regression.

Two real upgrades over the original model:
1. Uses ALL 21 engineered features (returns, RSI, MACD, Bollinger, volume,
   sentiment...) per timestep, not just raw Close price.
2. Fits the scaler ONLY on the training split and reuses that exact scaler
   for val/test. The original code refit MinMaxScaler at inference time,
   which is a subtle but real bug -- a model's inputs must be normalized
   the same way at prediction time as they were during training, or the
   numbers it sees no longer mean what it learned them to mean.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from utils.evaluation import chronological_split, regression_metrics, naive_price_baseline

SEQUENCE_FEATURES = [
    "Close", "Volume", "SMA_10", "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    "BBL_20_2.0_2.0", "BBM_20_2.0_2.0", "BBU_20_2.0_2.0", "BBB_20_2.0_2.0", "BBP_20_2.0_2.0",
    "Volatility_10", "Return_1d", "Return_5d", "Volume_Change", "Volume_SMA_10", "Sentiment",
]


def make_sequences(df: pd.DataFrame, feature_scaler: StandardScaler, seq_len: int = 30, fit_scaler: bool = False):
    """
    Converts a feature table into (X, y_return, close_at_prediction_day) sequences.
    X[i] = seq_len consecutive rows of SCALED features ending at day i
    y_return[i] = Target_Return for day i (the STATIONARY target the network trains on)
    close_at_prediction_day[i] = Close on day i -- needed to reconstruct an
        actual price prediction later via close * (1 + predicted_return)

    fit_scaler=True must ONLY be passed for the training split. Fitting the
    scaler on val/test would leak their distribution into "normalization" --
    exactly the kind of subtle leak that inflates reported performance.
    """
    feature_values = df[SEQUENCE_FEATURES].values
    scaled = feature_scaler.fit_transform(feature_values) if fit_scaler else feature_scaler.transform(feature_values)

    X, y_return, close_at_t = [], [], []
    returns = df["Target_Return"].values
    closes = df["Close"].values
    for i in range(seq_len, len(df)):
        X.append(scaled[i - seq_len:i])
        y_return.append(returns[i])
        close_at_t.append(closes[i])

    return np.array(X), np.array(y_return), np.array(close_at_t)


def build_lstm_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def predict_next_price(model, feature_scaler: StandardScaler, target_scaler: StandardScaler,
                       features: pd.DataFrame, seq_len: int = 30):
    """
    Predicts the next-period price using the most recent seq_len rows of a
    feature table. Returns (predicted_price, last_known_close).

    Uses the EXACT same feature_scaler and target_scaler that were fit
    during training -- never refits them here. Refitting a scaler at
    inference time (what the original code did) means the same price
    could get normalized differently than the model was trained to expect,
    silently corrupting predictions.
    """
    recent = features.tail(seq_len)
    if len(recent) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows of feature history to predict; got {len(recent)}.")

    X = feature_scaler.transform(recent[SEQUENCE_FEATURES].values)
    X = X.reshape(1, seq_len, len(SEQUENCE_FEATURES))

    pred_return_scaled = model.predict(X, verbose=0).flatten()
    pred_return = target_scaler.inverse_transform(pred_return_scaled.reshape(-1, 1)).flatten()[0]

    last_close = float(recent["Close"].iloc[-1])
    predicted_price = last_close * (1 + pred_return)

    return float(predicted_price), last_close


def train_lstm_price_model(
        features: pd.DataFrame,
        seq_len: int = 30,
        epochs: int = 50,
        save_path: str = "models/lstm_price.keras",
        feature_scaler_path: str = "models/lstm_feature_scaler.joblib",
        target_scaler_path: str = "models/lstm_target_scaler.joblib",
):
    """
    Trains the LSTM to predict Target_Return (stationary), reconstructs
    actual price predictions via close * (1 + predicted_return), and
    evaluates against the naive persistence baseline in real price units.

    Bug fix history on this function, both found by testing on real data:
    1. Target left unscaled while inputs were scaled -> network couldn't
       learn to output large values from a near-zero initialization.
       Fixed by scaling the target too.
    2. (This version) Even with scaling fixed, predicting absolute PRICE
       failed badly on real AAPL data (RMSE 50, MAPE 14.9%) because price
       level isn't stationary -- a scaler fit on 2021-2023 training prices
       (~$150-280) has no valid way to represent 2025-2026 test prices
       (~$290-325). Fixed by predicting Target_Return (% change) instead,
       which means the same thing regardless of price level, then
       reconstructing price via close * (1 + predicted_return) -- exact
       by construction, not an approximation.
    """
    train_df, val_df, test_df = chronological_split(features)

    feature_scaler = StandardScaler()
    X_train, y_train_ret, _ = make_sequences(train_df, feature_scaler, seq_len=seq_len, fit_scaler=True)
    X_val, y_val_ret, _ = make_sequences(val_df, feature_scaler, seq_len=seq_len, fit_scaler=False)
    X_test, y_test_ret, close_test = make_sequences(test_df, feature_scaler, seq_len=seq_len, fit_scaler=False)

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train_ret.reshape(-1, 1)).flatten()
    y_val = target_scaler.transform(y_val_ret.reshape(-1, 1)).flatten()

    model = build_lstm_model((seq_len, len(SEQUENCE_FEATURES)))

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )

    test_preds_scaled = model.predict(X_test, verbose=0).flatten()
    test_preds_return = target_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

    # Reconstruct actual price predictions: exact by construction since
    # Target_Return was DEFINED as (Target_Price - Close) / Close.
    test_preds_price = close_test * (1 + test_preds_return)
    test_actual_price = close_test * (1 + y_test_ret)  # == original Target_Price, reconstructed the same way

    model_metrics = regression_metrics(test_actual_price, test_preds_price)

    # The naive baseline must be scored on the SAME rows the LSTM was scored
    # on -- test_df loses its first seq_len rows to sequence windowing, so
    # we align the baseline to match or the comparison is apples-to-oranges.
    aligned_test_df = test_df.iloc[seq_len:]
    naive_preds = naive_price_baseline(aligned_test_df)
    baseline_metrics = regression_metrics(aligned_test_df["Target_Price"], naive_preds)

    report = {
        "model": model_metrics,
        "naive_baseline": baseline_metrics,
        "beats_baseline_rmse": model_metrics["RMSE"] < baseline_metrics["RMSE"],
        "epochs_trained": len(history.history["loss"]),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    model.save(save_path)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)

    return model, report