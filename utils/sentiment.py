"""
News + sentiment scoring.

Fetches recent headlines for a ticker/company from NewsAPI and scores them
with VADER, producing a daily-averaged sentiment series that feature_engineering.py
merges into the model's feature table.

Requires NEWSAPI_KEY in the environment (see .env.example). If it's missing or
the request fails, functions here degrade gracefully to "no sentiment data"
rather than crashing the whole pipeline — callers treat that as neutral (0.0).
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()  # reads .env into environment variables

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """VADER compound sentiment score: -1 (very negative) to +1 (very positive)."""
    return _analyzer.polarity_scores(text)["compound"]


def fetch_daily_sentiment(query: str, days: int = 30) -> pd.Series:
    """
    Fetch recent headlines for `query` (e.g. a company name or ticker) from
    NewsAPI, score each with VADER, and return a daily-averaged sentiment
    score indexed by date.

    Returns an empty Series (not an error) if no API key is configured or
    the request fails — the caller treats an empty series as neutral.
    NewsAPI's free tier only returns articles from the last month, so
    `days` beyond ~30 won't get you more history.
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return pd.Series(dtype=float)

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={query}&from={from_date}&language=en&sortBy=publishedAt"
        f"&pageSize=100&apiKey={api_key}"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except Exception:
        return pd.Series(dtype=float)

    rows = []
    for a in articles:
        title = a.get("title") or ""
        published = a.get("publishedAt")
        if not title or not published:
            continue
        date = pd.to_datetime(published).date()
        rows.append({"date": date, "score": score_text(title)})

    if not rows:
        return pd.Series(dtype=float)

    daily = pd.DataFrame(rows).groupby("date")["score"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily.name = "Sentiment"
    return daily
