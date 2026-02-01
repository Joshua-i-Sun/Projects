import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering and target creation."""
    data["Return"] = data["Close"].pct_change()
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["Target"] = (data["Return"].shift(-1) > 0).astype(int)
    data = data.dropna()
    return data


def train_model(data: pd.DataFrame):
    """Train logistic regression model on stock data."""
    X = data[["SMA_10", "SMA_50", "Return"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Prediction Accuracy: {acc:.2f}")

    return model, X_test, y_test, y_pred


def save_to_db(data: pd.DataFrame, table_name: str = "stock_data"):
    """Save processed data to SQLite database (can adapt for Databricks)."""
    engine = create_engine("sqlite:///stocks.db")
    data.to_sql(table_name, engine, if_exists="replace", index=True)
    print(f"Data saved to database table '{table_name}'")


def plot_data(data: pd.DataFrame, ticker: str):
    """Visualize stock closing price and moving averages."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close Price")
    plt.plot(data.index, data["SMA_10"], label="10-day SMA")
    plt.plot(data.index, data["SMA_50"], label="50-day SMA")
    plt.legend()
    plt.title(f"{ticker} Stock Price & Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()


def main():
    ticker = "AAPL" 
    start_date = "2020-01-01"
    end_date = "2025-01-01"

    print(f"Fetching {ticker} stock data...")
    raw_data = fetch_data(ticker, start_date, end_date)

    print("Preprocessing data...")
    processed_data = preprocess_data(raw_data)

    print("Training model...")
    model, X_test, y_test, y_pred = train_model(processed_data)

    print("Saving data to database...")
    save_to_db(processed_data, f"{ticker}_stock")

    print("Plotting results...")
    plot_data(processed_data, ticker)

    print("Done!")


if __name__ == "__main__":
    main()
