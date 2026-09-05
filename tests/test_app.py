"""
Tests for backend/app.py's routes, using Flask's test client. Data fetching
is mocked throughout -- these test ROUTING, request/response shape, and
error handling, not live yfinance/NewsAPI behavior (that's what manual
testing against real tickers is for -- see TESTING.md).
"""
from unittest import mock
import pandas as pd
import pytest

from backend.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200


def test_favicon_returns_204_not_404(client):
    """A silent, deliberate fix -- browsers auto-request this and it
    shouldn't show up as an error in server logs or dev tools."""
    r = client.get("/favicon.ico")
    assert r.status_code == 204


def test_predict_returns_expected_shape(client, synthetic_features):
    with mock.patch("backend.app.get_features", return_value=synthetic_features), \
            mock.patch("utils.model_cache.get_features", return_value=synthetic_features):
        r = client.get("/predict?ticker=TEST")

    assert r.status_code == 200
    data = r.get_json()
    assert set(data.keys()) == {"ticker", "predicted_price", "last_close"}
    assert isinstance(data["predicted_price"], (int, float))


def test_explain_returns_top_factors_with_real_feature_names(client, synthetic_features):
    with mock.patch("backend.app.get_features", return_value=synthetic_features), \
            mock.patch("utils.model_cache.get_features", return_value=synthetic_features):
        r = client.get("/explain?ticker=TEST")

    assert r.status_code == 200
    data = r.get_json()
    assert "top_factors" in data
    assert len(data["top_factors"]) > 0
    for factor in data["top_factors"]:
        assert "feature" in factor and "shap_contribution" in factor
        assert not factor["feature"].isdigit()  # never a raw timestep index


def test_sentiment_uses_company_name_not_raw_ticker(client):
    """Regression test for the IOC.NS bug: sentiment must search using the
    resolved company name, not the raw (possibly exchange-suffixed) ticker."""
    with mock.patch("backend.app.get_company_name", return_value="Indian Oil Corporation Limited") as mock_name, \
            mock.patch("backend.app.fetch_daily_sentiment", return_value=pd.Series(dtype=float)) as mock_sentiment:
        client.get("/sentiment?ticker=IOC.NS")

    mock_name.assert_called_once_with("IOC.NS")
    mock_sentiment.assert_called_once_with("Indian Oil Corporation Limited")


def test_sentiment_degrades_gracefully_with_no_news_data(client):
    with mock.patch("backend.app.get_company_name", return_value="Test Co"), \
            mock.patch("backend.app.fetch_daily_sentiment", return_value=pd.Series(dtype=float)):
        r = client.get("/sentiment?ticker=TEST")

    assert r.status_code == 200
    data = r.get_json()
    assert data["latest_sentiment"] == 0.0
    assert "note" in data


def test_model_info_returns_nulls_for_untrained_ticker(client):
    with mock.patch("backend.app.get_cached_report", return_value=None):
        r = client.get("/model-info?ticker=NEVERTRAINED")

    assert r.status_code == 200
    data = r.get_json()
    assert data["direction_model"] is None
    assert data["price_model"] is None


def test_predict_handles_upstream_failure_gracefully(client):
    """When fetching/training fails (bad ticker, network issue), the route
    should return a clean 500 with an error message -- never a raw traceback
    or an unhandled exception that crashes the server."""
    with mock.patch("backend.app.get_or_train_price_model", side_effect=ValueError("No data returned for ticker 'BADTICKER'.")):
        r = client.get("/predict?ticker=BADTICKER")

    assert r.status_code == 500
    assert "error" in r.get_json()