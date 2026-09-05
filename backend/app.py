import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import yfinance as yf

from utils.model_cache import (
    get_or_train_direction_model, get_or_train_price_model, get_or_train_multiday_model,
    get_features, get_multiday_features, get_cached_report,
)
from utils.sentiment import fetch_daily_sentiment
from utils.data_fetcher import _download_with_retry, get_company_name, get_ticker_currency
from models.lstm_price_model import predict_next_price
from models.lstm_multiday_model import predict_next_week
from backend.explainer import explain_prediction
from utils.agent import run_agent

app = Flask(__name__, static_folder="../frontend")


# --- Static frontend ---
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/style.css")
def serve_css():
    return send_from_directory(app.static_folder, "style.css")

@app.route("/script.js")
def serve_js():
    return send_from_directory(app.static_folder, "script.js")

# Silences the browser's automatic favicon.ico request (harmless 404 noise otherwise)
@app.route("/favicon.ico")
def favicon():
    return "", 204


# --- Predict: next-day price ---
@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        model, feature_scaler, target_scaler = get_or_train_price_model(ticker)
        features = get_features(ticker, period='2y', horizon=1)
        predicted_price, last_close = predict_next_price(model, feature_scaler, target_scaler, features, seq_len=30)

        return jsonify({
            'ticker': ticker,
            'predicted_price': round(predicted_price, 2),
            'last_close': round(last_close, 2),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Predict-week: 5-day direct multi-horizon forecast ---
@app.route('/predict-week', methods=['GET'])
def predict_week():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        model, feature_scaler, target_scaler = get_or_train_multiday_model(ticker)
        features = get_multiday_features(ticker, period='2y')
        prices, last_close, horizons = predict_next_week(model, feature_scaler, target_scaler, features, seq_len=30)

        labels = pd.bdate_range(start=datetime.today() + timedelta(days=1), periods=len(horizons)).strftime('%Y-%m-%d').tolist()

        return jsonify({
            'ticker': ticker,
            'last_close': round(last_close, 2),
            'labels': labels,
            'prices': [round(p, 2) for p in prices],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Explain: SHAP feature-level reasoning behind the direction prediction ---
@app.route('/explain', methods=['GET'])
def explain():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        model = get_or_train_direction_model(ticker)
        features = get_features(ticker, period='2y', horizon=1)
        last_row = features.iloc[-1]

        explanation = explain_prediction(model, last_row, top_n=5)
        explanation['ticker'] = ticker
        return jsonify(explanation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Sentiment: recent news sentiment for the ticker ---
@app.route('/sentiment', methods=['GET'])
def sentiment_endpoint():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        daily = fetch_daily_sentiment(get_company_name(ticker))
        if daily.empty:
            return jsonify({
                'ticker': ticker,
                'latest_sentiment': 0.0,
                'avg_7day_sentiment': 0.0,
                'note': 'No recent news data available (NewsAPI free tier covers ~30 days).',
            })

        latest = float(daily.iloc[-1])
        avg_recent = float(daily.tail(7).mean())
        return jsonify({
            'ticker': ticker,
            'latest_sentiment': round(latest, 4),
            'avg_7day_sentiment': round(avg_recent, 4),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Model-info: real backtested numbers for this ticker, for transparency ---
# Returns whatever's already been trained+cached; doesn't trigger training
# itself, since /predict, /predict-week, /explain already do that. If a
# model hasn't been trained for this ticker yet, that section is null and
# the frontend shows "not yet available" rather than a fake number.
@app.route('/model-info', methods=['GET'])
def model_info():
    ticker = request.args.get('ticker', 'AAPL')
    return jsonify({
        'ticker': ticker,
        'direction_model': get_cached_report(ticker, 'direction'),
        'price_model': get_cached_report(ticker, 'price'),
        'multiday_model': get_cached_report(ticker, 'multiday'),
    })


# --- Ticker name: resolves a symbol to its full company name + trading currency ---
@app.route('/ticker-name', methods=['GET'])
def ticker_name():
    ticker = request.args.get('ticker', 'AAPL')
    name = get_company_name(ticker)
    currency = get_ticker_currency(ticker)
    return jsonify({
        'ticker': ticker,
        'name': name if name != ticker else None,
        'currency_code': currency['code'],
        'currency_symbol': currency['symbol'],
    })


# --- Candlestick: daily OHLC history for chart context ---
# Upgraded from the original (2 days of hourly data -- too short to show
# any real trend) to a proper multi-month daily view.
@app.route('/candlestick-data', methods=['GET'])
def candlestick_data():
    ticker = request.args.get('ticker', 'AAPL')
    period = request.args.get('period', '6mo')

    try:
        df = _download_with_retry(ticker, period=period, interval='1d')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            return jsonify({'error': 'No candlestick data found'}), 404

        candles = []
        for ts, row in df.iterrows():
            if row[['Open', 'High', 'Low', 'Close']].isna().any():
                continue
            candles.append({
                'time': ts.strftime('%Y-%m-%d'),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
            })

        return jsonify({'ticker': ticker, 'candles': candles})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Agent chat: conversation history lives server-side, keyed by a
# session id the frontend generates and holds onto. This keeps the raw
# Anthropic SDK message objects (needed for multi-turn tool-use context)
# out of the JSON sent to the browser -- the frontend only ever sees
# plain {response, tool_calls} JSON, never the SDK internals.
_AGENT_SESSIONS = {}

@app.route('/agent/chat', methods=['POST'])
def agent_chat():
    data = request.get_json(force=True) or {}
    session_id = data.get('session_id')
    user_message = data.get('message', '').strip()

    if not session_id or not user_message:
        return jsonify({'error': 'session_id and message are required'}), 400

    try:
        history = _AGENT_SESSIONS.get(session_id, [])
        result = run_agent(user_message, conversation_history=history)
        _AGENT_SESSIONS[session_id] = result['messages']

        return jsonify({
            'response': result['response'],
            'tool_calls': result['tool_calls'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/agent/reset', methods=['POST'])
def agent_reset():
    data = request.get_json(force=True) or {}
    session_id = data.get('session_id')
    _AGENT_SESSIONS.pop(session_id, None)
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(debug=True)