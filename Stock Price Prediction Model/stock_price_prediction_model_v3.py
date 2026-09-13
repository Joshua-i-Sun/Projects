"""
Stock Price Prediction Model (Return-Prediction Version)

Predicts next-day RETURN instead of next-day price level. Returns are
approximately stationary, so tree-based models can actually learn from
them; price levels are non-stationary, which is why Random Forest failed
in the previous version.

Baseline: predict return = 0, which is mathematically equivalent to
"tomorrow's price = today's price" (the same baseline as before).

Author: Joshua Sun
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sqlalchemy import create_engine


# ---------- 1. DATA LOADING ----------

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


# ---------- 2. FEATURE ENGINEERING ----------

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create STATIONARY technical features.

    Key change from the price-level version: we no longer include SMA_10
    or SMA_50 as raw dollar values, because those are non-stationary and
    a tree-based model cannot extrapolate them. Instead we use SMA_ratio
    (dimensionless), MACD (already a difference of EMAs, roughly stationary),
    volatility (a rolling standard deviation of returns), and lagged returns.
    """
    df = data.copy()

    # Daily return
    df["Return"] = df["Close"].pct_change()

    # Moving averages (kept for plotting only, NOT as features)
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Stationary feature: ratio of short vs. long moving average
    df["SMA_ratio"] = df["SMA_10"] / df["SMA_50"]

    # MACD (12-day EMA - 26-day EMA)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Historical volatility (30-day rolling std of returns)
    df["Volatility"] = df["Return"].rolling(window=30).std()

    # 7-day price change (dimensionless)
    df["Price_change_7d"] = df["Close"].pct_change(periods=7)

    # NEW TARGET: next-day return (stationary)
    df["Target"] = df["Return"].shift(-1)

    df = df.dropna()
    return df


# ---------- 3. BASELINE ----------

def naive_baseline_returns(y_test: pd.Series) -> dict:
    """
    Baseline for return prediction: predict return = 0.
    This is mathematically equivalent to predicting tomorrow's price =
    today's price, but evaluated in return space so it is comparable to
    the model outputs.
    """
    baseline_pred = np.zeros(len(y_test))
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, baseline_pred)),
        "MAE": mean_absolute_error(y_test, baseline_pred),
        "R2": r2_score(y_test, baseline_pred),
    }


# ---------- 4. MODEL TRAINING & EVALUATION ----------

def train_and_evaluate(data: pd.DataFrame):
    """Train Linear Regression and Random Forest on stationary features."""

    # NOTE: no raw SMA_10 / SMA_50 here. Only stationary features.
    feature_cols = [
        "SMA_ratio", "MACD", "MACD_signal",
        "Volatility", "Return", "Price_change_7d",
    ]

    X = data[feature_cols]
    y = data["Target"]

    # Chronological split (80/20), no shuffle
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Baseline
    baseline = naive_baseline_returns(y_test)

    def metrics(y_true, y_pred):
        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    results = pd.DataFrame({
        "Baseline (return=0)": baseline,
        "Linear Regression": metrics(y_test, lr_pred),
        "Random Forest": metrics(y_test, rf_pred),
    }).T

    print("\n===== Model Comparison (returns) =====")
    print(results.round(4))
    print("======================================\n")

    # Reconstruct prices for the plot
    current_prices = data["Close"].iloc[split_idx:]
    actual_prices = current_prices.values * (1 + y_test.values)
    lr_prices = current_prices.values * (1 + lr_pred)
    rf_prices = current_prices.values * (1 + rf_pred)

    return (
        results,
        y_test, lr_pred, rf_pred,
        actual_prices, lr_prices, rf_prices,
        current_prices,
        lr, rf, scaler, feature_cols,
    )


# ---------- 5. VISUALIZATION ----------

def plot_results(data, current_prices, actual_prices, lr_prices, rf_prices, ticker):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: reconstructed prices
    axes[0].plot(current_prices.index, actual_prices, label="Actual", linewidth=2)
    axes[0].plot(current_prices.index, lr_prices, label="Linear Regression", alpha=0.8)
    axes[0].plot(current_prices.index, rf_prices, label="Random Forest", alpha=0.8)
    axes[0].plot(current_prices.index, current_prices.values,
                 label="Baseline (= today's price)", alpha=0.6, linestyle="--")
    axes[0].set_title(f"{ticker}: Reconstructed Next-Day Price from Return Predictions")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: feature trends
    axes[1].plot(data.index, data["Close"], label="Close", linewidth=1.5)
    axes[1].plot(data.index, data["SMA_10"], label="SMA 10", alpha=0.7)
    axes[1].plot(data.index, data["SMA_50"], label="SMA 50", alpha=0.7)
    axes[1].set_title(f"{ticker}: Price & Moving Averages")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Price (USD)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("prediction_results.png", dpi=150)
    print("Saved plot to prediction_results.png")
    plt.show()


# ---------- 6. DATABASE ----------

def save_to_db(data: pd.DataFrame, table_name: str = "stock_data"):
    engine = create_engine("sqlite:///stocks.db")
    data.to_sql(table_name, engine, if_exists="replace", index=True)
    print(f"Data saved to table '{table_name}'")


# ---------- 7. MAIN ----------

def main():
    ticker = "NVDA"
    start_date = "2020-01-01"
    end_date = "2025-01-01"

    print(f"Fetching {ticker} data...")
    raw = fetch_data(ticker, start_date, end_date)

    print("Engineering stationary features...")
    data = engineer_features(raw)
    print(f"Feature set shape: {data.shape}")

    print("Training models...")
    (results, y_test, lr_pred, rf_pred,
     actual_prices, lr_prices, rf_prices,
     current_prices, lr, rf, scaler, features) = train_and_evaluate(data)

    print("Saving to database...")
    save_to_db(data, f"{ticker}_stock")

    print("Plotting...")
    plot_results(data, current_prices, actual_prices, lr_prices, rf_prices, ticker)

    print("Done.")


if __name__ == "__main__":
    main()