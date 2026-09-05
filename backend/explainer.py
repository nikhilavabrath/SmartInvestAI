"""
Explainability for the direction classifier.

Uses SHAP's TreeExplainer, which is fast and EXACT for tree-based models
(unlike KernelExplainer, which is a slow approximation and was the wrong
tool for the original LSTM-over-raw-timesteps setup). Because we're
explaining engineered features (RSI, MACD, etc.) instead of raw price
timesteps, the output is finally something a human -- or an interviewer --
can actually read and trust: "RSI_14 pushed the prediction toward 'up'"
instead of "timestep 43 contributed +0.03".
"""
import numpy as np
import pandas as pd
import shap

from models.xgb_direction_model import FEATURE_COLUMNS


def explain_prediction(model, row: pd.Series, top_n: int = 5) -> dict:
    """
    Explains a single prediction (one row of features -- typically the
    most recent trading day for a ticker).

    Returns a dict with the predicted class/probability plus the top_n
    features that pushed the prediction toward "up" or "down", each with
    its SHAP value (signed -- positive pushes toward "up", negative
    toward "down") and its actual value that day, so the explanation is
    self-contained (e.g. "RSI_14 = 78.3, pushed toward UP, contribution +0.41").
    """
    X = row[FEATURE_COLUMNS].to_frame().T.astype(float)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # TreeExplainer returns a 2D array (1 row, n_features) for binary XGBoost classifiers
    values = shap_values[0] if shap_values.ndim == 2 else shap_values

    contributions = pd.Series(values, index=FEATURE_COLUMNS)
    top_features = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(top_n)

    proba_up = float(model.predict_proba(X)[0][1])

    explanation = {
        "predicted_direction": "up" if proba_up > 0.5 else "down",
        "probability_up": round(proba_up, 4),
        "top_factors": [
            {
                "feature": feat,
                "value": round(float(row[feat]), 4),
                "shap_contribution": round(float(contrib), 4),
                "pushed_toward": "up" if contrib > 0 else "down",
            }
            for feat, contrib in top_features.items()
        ],
    }
    return explanation