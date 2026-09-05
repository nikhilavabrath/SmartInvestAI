"""
Tools the agent can call. Each one wraps an EXISTING, already-tested piece
of the pipeline -- the agent doesn't do new ML, it orchestrates what's
already built. Every tool returns a plain dict (JSON-serializable, small)
so it's cheap to send back to the LLM as a tool result, and every tool
catches its own exceptions and returns {"error": ...} instead of raising --
an agent mid-conversation should never crash on a bad ticker, it should be
able to see the error and explain it to the user.
"""
from utils.model_cache import (
    get_or_train_direction_model, get_or_train_price_model, get_or_train_multiday_model,
    get_features, get_multiday_features, get_cached_report,
)
from utils.data_fetcher import get_company_name
from utils.sentiment import fetch_daily_sentiment
from models.lstm_price_model import predict_next_price
from models.lstm_multiday_model import predict_next_week
from backend.explainer import explain_prediction


def get_next_day_prediction(ticker: str) -> dict:
    try:
        model, feature_scaler, target_scaler = get_or_train_price_model(ticker)
        features = get_features(ticker, period="2y", horizon=1)
        predicted_price, last_close = predict_next_price(model, feature_scaler, target_scaler, features, seq_len=30)
        report = get_cached_report(ticker, "price")
        return {
            "ticker": ticker,
            "predicted_price": round(predicted_price, 2),
            "last_close": round(last_close, 2),
            "backtested_rmse": report["model"]["RMSE"] if report else None,
            "naive_baseline_rmse": report["naive_baseline"]["RMSE"] if report else None,
        }
    except Exception as e:
        return {"error": str(e)}


def get_five_day_forecast(ticker: str) -> dict:
    try:
        model, feature_scaler, target_scaler = get_or_train_multiday_model(ticker)
        features = get_multiday_features(ticker, period="2y")
        prices, last_close, horizons = predict_next_week(model, feature_scaler, target_scaler, features, seq_len=30)
        return {
            "ticker": ticker,
            "last_close": round(last_close, 2),
            "forecast_by_day": {f"day_{h}": round(p, 2) for h, p in zip(horizons, prices)},
        }
    except Exception as e:
        return {"error": str(e)}


def get_direction_explanation(ticker: str) -> dict:
    try:
        model = get_or_train_direction_model(ticker)
        features = get_features(ticker, period="2y", horizon=1)
        explanation = explain_prediction(model, features.iloc[-1], top_n=5)
        report = get_cached_report(ticker, "direction")
        explanation["backtested_f1"] = report["model"]["F1"] if report else None
        explanation["naive_baseline_f1"] = report["naive_baseline"]["F1"] if report else None
        return explanation
    except Exception as e:
        return {"error": str(e)}


def get_news_sentiment(ticker: str) -> dict:
    try:
        company_name = get_company_name(ticker)
        daily = fetch_daily_sentiment(company_name)
        if daily.empty:
            return {"ticker": ticker, "sentiment": 0.0, "note": "No recent news coverage found."}
        return {
            "ticker": ticker,
            "latest_sentiment": round(float(daily.iloc[-1]), 4),
            "avg_7day_sentiment": round(float(daily.tail(7).mean()), 4),
        }
    except Exception as e:
        return {"error": str(e)}


def get_backtested_performance(ticker: str) -> dict:
    """Returns whatever's already been trained+cached for this ticker -- does
    NOT trigger training itself, so this is a fast, cheap tool to call."""
    return {
        "ticker": ticker,
        "direction_model": get_cached_report(ticker, "direction"),
        "price_model": get_cached_report(ticker, "price"),
        "multiday_model": get_cached_report(ticker, "multiday"),
    }


# Dispatch table: tool name (as the LLM will call it) -> Python function
TOOL_FUNCTIONS = {
    "get_next_day_prediction": get_next_day_prediction,
    "get_five_day_forecast": get_five_day_forecast,
    "get_direction_explanation": get_direction_explanation,
    "get_news_sentiment": get_news_sentiment,
    "get_backtested_performance": get_backtested_performance,
}

# Tool schemas in Anthropic's tool-use format -- kept as the source of truth
# since it's the simpler/flatter shape. Converted to OpenAI's nested format
# (needed for Groq, which is OpenAI-compatible) by to_openai_tool_schemas().
TOOL_SCHEMAS = [
    {
        "name": "get_next_day_prediction",
        "description": "Get tomorrow's predicted closing price for a ticker, along with the model's backtested RMSE vs a naive baseline.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL or TCS.NS"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_five_day_forecast",
        "description": "Get a direct (non-autoregressive) 5-trading-day price forecast for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL or TCS.NS"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_direction_explanation",
        "description": "Get the model's up/down direction prediction with SHAP-based feature explanations (which real indicators, like RSI or MACD, drove the prediction), plus backtested F1 vs baseline.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL or TCS.NS"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news_sentiment",
        "description": "Get recent news sentiment (VADER score, -1 to +1) for a ticker's company, based on recent headlines.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL or TCS.NS"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_backtested_performance",
        "description": "Get this ticker's already-trained models' backtested performance vs naive baselines, if available. Does not trigger new training.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL or TCS.NS"}},
            "required": ["ticker"],
        },
    },
]


def to_openai_tool_schemas() -> list:
    """
    Converts TOOL_SCHEMAS from Anthropic's flat {name, description,
    input_schema} shape into OpenAI/Groq's nested
    {type: "function", function: {name, description, parameters}} shape.
    One source of truth for the tool definitions either way.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        }
        for schema in TOOL_SCHEMAS
    ]