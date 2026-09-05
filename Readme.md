# SmartInvestAI - Honest AI-Driven Stock Trend Prediction

A stock trend prediction system that treats "the model is close to a naive
baseline" as a real, reportable finding rather than something to hide. Every
prediction is shown next to what a trivial "assume no change" guess would
have gotten, because that comparison is the whole point of the project.

## Why this project is built the way it is

Short-horizon stock price movement is close to a random walk — a well-known
finding in finance, and one this project independently reproduces rather than
ignores. Four separate experiments (single-ticker, more data, pooled across
10 tickers, and a 576-config hyperparameter search) all failed to
meaningfully beat a naive "predict no change" baseline for 1-day direction.
That's treated as a real result, not a bug to keep chasing — see
**Backtested results** below.

## Architecture

```
frontend/          Vanilla HTML/CSS/JS — ticker input, prediction, forecast
                    chart, SHAP explainability panel, sentiment, backtested
                    performance panel, candlestick chart, agent chat
backend/app.py      Flask API — thin routing layer over utils/ and models/
utils/
  data_fetcher.py       yfinance OHLCV fetch + cleaning/validation
  feature_engineering.py  Technical indicators (RSI, MACD, Bollinger, etc.)
                          + sentiment + targets (price, direction, return)
  sentiment.py          NewsAPI headline fetch + VADER scoring
  evaluation.py         Chronological train/val/test split, naive baselines,
                        regression/classification metrics — no leakage
  model_cache.py        Per-ticker model training + disk caching
  agent.py / agent_tools.py   Hand-rolled tool-calling agent (see below)
models/
  xgb_direction_model.py   XGBoost direction (up/down) classifier
  lstm_price_model.py      Multivariate LSTM, predicts % return not price
                          level (see "Bugs found" below for why)
  lstm_multiday_model.py   Direct 5-day multi-horizon LSTM (not autoregressive)
tests/              pytest suite, 30+ tests, all data fetching mocked
```

## Tech stack

Python, Flask, yfinance, pandas-ta, scikit-learn, XGBoost, TensorFlow/Keras,
SHAP, VADER sentiment, NewsAPI, Chart.js, Plotly. Agent: hand-rolled
tool-calling loop against Groq's free-tier API (OpenAI-compatible), no
LangChain/LangGraph — built manually to actually understand the mechanics.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in NEWSAPI_KEY and GROQ_API_KEY (both free, see below)
python backend/app.py
```
Open `http://127.0.0.1:5000/`. First request for a new ticker trains and
caches that ticker's models (a minute or two); subsequent requests are fast.

**Free API keys needed:**
- [NewsAPI](https://newsapi.org) - free tier, 500 req/day, powers sentiment
- [Groq](https://console.groq.com) - free tier, no card required, powers the
  chat agent (see `utils/agent.py` for why Groq over paid options)

## Testing

```bash
pip install pytest
pytest                  # everything (~60s, includes real model training)
pytest -m "not slow"    # fast subset (~10s), skips model training
```
See `TESTING.md` for what's covered, what's deliberately not covered, and a
manual-testing checklist for things that depend on live external APIs.

## Backtested results

Numbers from a real run (AAPL, chronological 70/15/15 split, no leakage —
see `utils/evaluation.py`):

| Model | Metric | Model | Naive baseline | Beats baseline? |
|---|---|---|---|---|
| Direction (1-day) | F1 | 0.40–0.55 (ticker-dependent) | 0.00–0.70 | Inconsistent |
| Price (1-day) | RMSE | 5.08 | 5.11 | Essentially tied |
| Direction (pooled, 10 tickers, 576-config tuned) | F1 | 0.69 | 0.70 | No |

These are checked live per-ticker in the app's "Model performance,
backtested" panel and the agent's `get_backtested_performance` tool - not
just numbers frozen in this README.

## Bugs found and fixed along the way

Kept here because they're as much a part of the engineering story as the
features themselves:
- **yfinance MultiIndex columns** silently broke `pandas_ta` — fixed by
  flattening columns in `data_fetcher.py`.
- **Unscaled LSTM target**: predicting raw price with only inputs scaled
  meant the network couldn't learn to output the right magnitude at all
  (RMSE ~228). Fixed by scaling the target too.
- **Non-stationary price level**: even with scaling fixed, predicting
  absolute price failed badly (RMSE ~50, MAPE 14.9%) across a multi-year
  trend, because a scaler fit on 2021-2023 prices doesn't represent
  2025-2026 prices. Fixed by predicting % return instead and reconstructing
  price exactly via `close * (1 + return)`.
- **`inf` from zero-volume days** crashed XGBoost outright on thinly-traded
  stocks (e.g. IOC.NS) via `.pct_change()` divide-by-zero. Fixed with an
  explicit inf→NaN cleanup pass.
- **Sentiment searched the literal ticker string** ("IOC.NS"), which matches
  no news articles. Fixed by resolving the real company name first.
- **Hardcoded `$` currency** on non-US tickers. Fixed by resolving the
  actual trading currency from yfinance.
- **Agent tool-call looping**: on broad multi-part questions, the free Groq
  model sometimes re-called the same tool repeatedly instead of progressing,
  exhausting its iteration budget. Fixed with same-turn result caching plus
  a corrective nudge back to the model.

## Limitations (stated plainly, on purpose)

- 1-day and 5-day direction prediction show no consistent edge over a naive
  baseline across the tickers tested. Price regression is close to tied with
  a naive "no change" guess.
- Sentiment coverage depends on NewsAPI's free-tier ~30-day history window
  and general-purpose (non-finance-tuned) VADER scoring.
- The agent runs on a free open-weight model; it's noticeably better at
  focused single-topic questions than broad multi-part ones (a real,
  observed tradeoff of not using a paid frontier model).
- Nothing here is investment advice, and the tool/agent are both built to
  say so explicitly rather than imply otherwise.

## Possible future work

- Cross-sectional pooled models across a larger ticker universe
- Direct multi-day *classification* (not just regression) with calibrated
  probabilities
- Docker packaging
- A paid frontier model for the agent, if more reliable multi-part tool
  planning becomes a priority