# Stock Return Prediction Model (v4)

A next-day **return** prediction pipeline with walk-forward validation,
bootstrap significance testing, and a long/flat backtest, exposed through
a Streamlit UI.

## Key finding

Neither model meaningfully outperforms the naive zero-return baseline,
and bootstrap confidence intervals confirm the difference is **not
statistically significant**. Daily returns are dominated by noise; this
is the honest, useful result.

## What's new over v3

- **Walk-forward validation** (5 expanding folds) replaces the single
  80/20 split, giving mean ± std per metric.
- **More models**: Ridge, Random Forest, Gradient Boosting, optional XGBoost.
- **Directional accuracy** — the metric that matters for a trading signal.
- **Bootstrap CI** on the RMSE difference vs baseline.
- **Long/flat backtest** with configurable transaction costs, benchmarked
  against buy-and-hold.
- **Richer stationary features**: RSI(14), Bollinger Band position,
  volume ratio, lagged returns, 30-day price change.
- **Interactive Streamlit UI** with per-fold metrics, feature importance,
  equity curves, and CSV downloads.

## Project structure
stock_prediction/
├── stock_prediction_v4.py # Pipeline (importable + CLI)
├── app.py # Streamlit UI
├── requirements.txt
├── README.md
├── stocks.db # Created on first run
└── prediction_results.png # Created by the CLI

## Overview
**Methodology
Why the target is next-day return, not price
An earlier version predicted price directly. Random Forest failed
catastrophically (R² ≈ -0.67) because prices are non-stationary and
tree-based models cannot extrapolate outside the range seen during
training. Switching to next-day return removed that failure mode
entirely (R² back into the -0.006 range).

**Why walk-forward instead of a single split
A single 80/20 split gives one number with no uncertainty. Walk-forward
gives five numbers — one per fold — so you can see whether a model's
edge is consistent or an artifact of a particular test window.

**Why bootstrap the RMSE difference
A small RMSE improvement can be pure luck. Bootstrapping resamples the
test residuals 2,000 times and reports the 95% interval of the
difference. If it contains 0, the "improvement" is noise.

**Features
All features are stationary: returns, MA ratios, MACD family,
multi-window volatility, RSI(14), Bollinger Band position, volume ratio,
multi-horizon price changes, and cyclical day-of-week encoding. Raw
price levels and raw SMA dollar values are excluded.

**Results
Trained on NVDA daily data (2020–2025), 5 walk-forward folds.
The exact numbers depend on the date range and folds; the UI prints
them per run. The qualitative result is stable: all models cluster
within a few basis points of the baseline RMSE, and the bootstrap CI
contains zero.

**Limitations
Single ticker by default. Extend to a ticker list to test
generalization.

No hyperparameter tuning. TimeSeriesSplit + GridSearchCV is
the natural next step.

No fundamental, sentiment, or macro features.

No live trading integration. The backtest is illustrative.

**What this project demonstrates
Time-series methodology: stationary targets, chronological splits,
walk-forward validation, avoiding leakage.

Model evaluation: multiple metrics, directional accuracy, naive
baselines, bootstrap significance testing.

Diagnosing failure modes: recognizing why tree models fail on
non-stationary targets and fixing it via target transformation.

Product thinking: shipping a UI that a non-technical reviewer can
run in one click.

## How to run

```bash
pip install -r requirements.txt

# CLI
python stock_prediction_v4.py --ticker NVDA --start 2020-01-01 --end 2025-01-01

# UI
streamlit run app.py

## Overview


## License

This project is for educational and portfolio purposes.