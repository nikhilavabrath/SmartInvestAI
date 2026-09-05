"""
Direction classifier: predicts up/down using engineered features.

XGBoost over the feature table (not raw price sequences) — this is the
standard approach for tabular financial features, trains fast, and gives
real per-feature importances (unlike the LSTM's raw timesteps).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import f1_score

from utils.evaluation import chronological_split, classification_metrics, naive_direction_baseline

FEATURE_COLUMNS = [
    "SMA_10", "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    "BBL_20_2.0_2.0", "BBM_20_2.0_2.0", "BBU_20_2.0_2.0", "BBB_20_2.0_2.0", "BBP_20_2.0_2.0",
    "Volatility_10", "Return_1d", "Return_5d",
    "Volume_Change", "Volume_SMA_10", "Sentiment",
]


def tune_hyperparameters(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                         save_path: str = "models/xgb_direction_tuned.joblib"):
    """
    Small grid search over XGBoost hyperparameters, each candidate trained
    on train_df and scored by F1 on val_df (never on test_df -- test stays
    untouched until the very end, exactly once, with the winning config).

    We can't use standard k-fold CV here (it would shuffle time order and
    leak future data into training), so this uses a single, fixed
    chronological train/val split for every candidate instead -- the
    correct adaptation of hyperparameter search to time-series data.
    """
    from sklearn.model_selection import ParameterGrid

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["Target_Direction"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["Target_Direction"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["Target_Direction"]

    param_grid = {
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [1, 5, 10],
    }

    best_f1 = -1
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        model = xgb.XGBClassifier(
            n_estimators=300,
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_preds = model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_params = params

    test_preds = best_model.predict(X_test)
    model_metrics = classification_metrics(y_test, test_preds)

    naive_preds = naive_direction_baseline(train_df, len(test_df))
    baseline_metrics = classification_metrics(y_test, naive_preds)

    report = {
        "best_params": best_params,
        "best_val_f1": best_f1,
        "model": model_metrics,
        "naive_baseline": baseline_metrics,
        "beats_baseline_f1": model_metrics["F1"] > baseline_metrics["F1"],
        "best_iteration": best_model.best_iteration,
    }

    joblib.dump(best_model, save_path)
    return best_model, report


def train_direction_model(features: pd.DataFrame, save_path: str = "models/xgb_direction.joblib"):
    """
    Single-ticker convenience wrapper: splits one ticker's feature table
    chronologically, then delegates to train_on_split.
    """
    train_df, val_df, test_df = chronological_split(features)
    return train_on_split(train_df, val_df, test_df, save_path=save_path)


def train_on_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   save_path: str = "models/xgb_direction.joblib"):
    """
    Core trainer: takes ALREADY-SPLIT train/val/test frames (so it works
    identically for a single ticker or a pooled multi-ticker table -- the
    caller decides how the split was made) and trains + evaluates.

    Returns (model, report_dict) so the caller can inspect/log the numbers
    (e.g. into MLflow) without re-parsing printed text.
    """
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["Target_Direction"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["Target_Direction"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["Target_Direction"]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    test_preds = model.predict(X_test)
    model_metrics = classification_metrics(y_test, test_preds)

    naive_preds = naive_direction_baseline(train_df, len(test_df))
    baseline_metrics = classification_metrics(y_test, naive_preds)

    report = {
        "model": model_metrics,
        "naive_baseline": baseline_metrics,
        "beats_baseline_f1": model_metrics["F1"] > baseline_metrics["F1"],
        "best_iteration": model.best_iteration,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }

    joblib.dump(model, save_path)
    return model, report


def feature_importance(model, top_n: int = 10) -> pd.Series:
    """
    Returns the top_n most important features by XGBoost's gain-based importance.
    This is what powers real, honest explainability -- 'RSI_14 mattered most'
    instead of 'timestep 43 mattered most'.
    """
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    return importances.sort_values(ascending=False).head(top_n)