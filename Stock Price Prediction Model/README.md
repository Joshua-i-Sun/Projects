# Stock Price Prediction Model

A next-day return prediction model built with Python, Pandas, and
Scikit-learn. Uses engineered technical features and benchmarks two
regression models against a naive zero-return baseline.

## Overview

This project predicts next-day stock returns using technical indicators
derived from historical price data. The goal is not to "beat the market" —
it is to build a clean, methodologically correct modeling pipeline and to
report the results honestly, including a negative result.

The project includes:

- Automated data loading from Yahoo Finance via `yfinance`
- Feature engineering of stationary technical indicators
- Two regression models: Linear Regression and Random Forest
- A naive baseline (predict return = 0) for comparison
- Chronological train/test split (no shuffle) to avoid data leakage
- Evaluation with RMSE, MAE, and R²
- Optional SQLite persistence and Matplotlib visualization

## Key Finding

**Neither model meaningfully outperforms the naive zero-return baseline.**

This is the honest result and the most important takeaway. Daily stock
returns are dominated by noise. Short-horizon prediction from technical
indicators alone is extremely difficult, and a naive baseline that always
predicts "tomorrow's price = today's price" is a very strong benchmark.

Reporting this honestly is more useful than tuning the model to produce
a misleadingly optimistic result.

## Project Structure

```
stock_prediction/
├── stock_prediction.py     # Main script: fetch, engineer, train, evaluate
├── prediction_results.png  # Generated plot (actual vs. predicted)
├── stocks.db               # SQLite database (created on first run)
└── README.md
```

## Methodology

### Why the target is next-day return, not next-day price

An earlier version of this project predicted the next-day **price**
directly. Random Forest failed catastrophically on that version
(RMSE ~$33, R² = -0.67) because:

- Stock prices are **non-stationary** — their distribution shifts over
  time.
- Tree-based models cannot **extrapolate** beyond the target range seen
  during training. When test-period prices fell outside that range, every
  prediction was systematically wrong.

Switching the target to next-day **return** (which is approximately
stationary) removed this failure mode entirely. Random Forest went from
R² = -0.67 to R² = -0.0056 — back into a reasonable range.

This is a known issue in time-series forecasting, and it is one of the
most valuable lessons from this project.

### Features engineered

All features are **stationary** — that is, dimensionless or roughly
constant in distribution over time. Raw price levels are intentionally
excluded.

| Feature | Description |
|---|---|
| `Return` | Daily percentage change in close price |
| `SMA_ratio` | Ratio of 10-day to 50-day simple moving average |
| `MACD` | 12-day EMA minus 26-day EMA (momentum) |
| `MACD_signal` | 9-day EMA of MACD (signal line) |
| `Volatility` | 30-day rolling standard deviation of daily returns |
| `Price_change_7d` | 7-day percentage change in close price |

Moving averages (SMA_10, SMA_50) are computed for **plotting only** and
are not used as model features, because their raw dollar values are
non-stationary.

### Baseline

The baseline predicts that **tomorrow's return will be zero**, which is
mathematically equivalent to predicting "tomorrow's price equals today's
price." It is the simplest possible prediction and a standard benchmark
for time-series forecasting.

If a model cannot beat this baseline, it is not adding value.

### Train/test split

The data is split **chronologically** (80% train, 20% test) with no
shuffling. Random shuffling on time-series data causes **data leakage** —
the model would train on future data and "predict" the past, producing
artificially good results.

## Results

Trained on NVDA daily data (2020–2025), evaluated on a chronological 20%
holdout.

| Model                | RMSE   | MAE    | R²      |
|----------------------|--------|--------|---------|
| Baseline (return=0)  | 0.0335 | 0.0250 | -0.0155 |
| Linear Regression    | 0.0333 | 0.0250 |  0.0016 |
| Random Forest        | 0.0334 | 0.0245 | -0.0056 |

**Interpretation:**

- **Linear Regression** marginally beat the baseline on RMSE
  (0.0333 vs. 0.0335 — a 0.6% improvement).
- **Random Forest** marginally beat the baseline on MAE
  (0.0245 vs. 0.0250 — a 2% improvement).
- Both improvements are small and consistent with the low
  signal-to-noise ratio of daily returns.
- The models are not meaningfully better than the baseline. This is the
  honest finding.

### Why the baseline R² is negative

The baseline predicts 0 every day. R² compares a model's errors against
the errors of predicting the **mean** of the test set — not against 0.
Because NVDA had a positive average return during the test period,
predicting 0 is worse than predicting the mean, so the baseline's R² is
slightly negative (-0.0155). This is expected and does not indicate a bug.

### Why Random Forest's R² is slightly negative but its MAE is best

Random Forest produces lower average absolute error but occasionally
makes larger errors, which RMSE and R² penalize more heavily than MAE.
This is a common trade-off and illustrates why looking at a single metric
can be misleading.

## How to Run

### Requirements

```bash
pip install yfinance pandas numpy scikit-learn matplotlib sqlalchemy pyarrow
```

Or, if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Run

```bash
python stock_prediction.py
```

By default, the script:

1. Downloads NVDA daily data from 2020-01-01 to 2025-01-01.
2. Engineers stationary technical features.
3. Trains Linear Regression and Random Forest on an 80/20 chronological
   split.
4. Evaluates both models against the naive zero-return baseline.
5. Prints a results table to the console.
6. Saves the processed data to `stocks.db`.
7. Saves a plot to `prediction_results.png`.

To change the ticker or date range, edit the `main()` function:

```python
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2025-01-01"
```

## Code Overview

| Function | Purpose |
|---|---|
| `fetch_data(ticker, start, end)` | Download historical data via `yfinance` |
| `engineer_features(data)` | Compute stationary technical features and next-day return target |
| `naive_baseline_returns(y_test)` | Baseline: predict return = 0 |
| `train_and_evaluate(data)` | Chronological split, train both models, print comparison table |
| `plot_results(...)` | Save actual vs. predicted price plot |
| `save_to_db(data, table_name)` | Persist processed data to SQLite |

## Limitations

- **Single ticker.** Results are reported for NVDA only. Performance
  varies significantly across tickers and time periods.
- **No hyperparameter tuning.** Random Forest uses default-adjacent
  settings (`n_estimators=100`, `max_depth=5`). A proper grid search or
  time-series cross-validation would be the next step.
- **Technical indicators only.** No fundamental data, sentiment, or
  macro features.
- **No transaction costs modeled.** Even if returns were predictable, a
  real trading strategy would need to account for spreads, commissions,
  and slippage.
- **No statistical significance testing.** The improvement over the
  baseline is not tested for significance.

## Future Improvements

- **Cross-validation across multiple tickers** (e.g., SPY, MSFT, TSLA)
  to test whether the marginal improvement generalizes.
- **Walk-forward validation** instead of a single train/test split.
- **Hyperparameter tuning** with `TimeSeriesSplit` and `GridSearchCV`.
- **Additional features:** volume-based signals, sector-relative returns,
  volatility regime indicators.
- **Model comparison:** gradient boosting (XGBoost, LightGBM), which
  often outperform vanilla Random Forest on tabular data.
- **Statistical testing:** compare RMSE distributions across tickers and
  test whether the improvement over baseline is significant.

## What This Project Demonstrates

- **Feature engineering:** constructing stationary indicators from raw
  price data, and knowing when to exclude non-stationary features.
- **Time-series methodology:** chronological splits, avoiding data
  leakage, choosing a target that a tree-based model can actually learn.
- **Model evaluation:** using multiple metrics (RMSE, MAE, R²), comparing
  against a naive baseline, and interpreting the results honestly.
- **Diagnosing failure modes:** recognizing why Random Forest failed on
  price prediction and how to fix it via target transformation.

## License

This project is for educational and portfolio purposes.