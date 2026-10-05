"""
Stock Return Prediction Model v4

Upgrades over v3:
- Walk-forward validation (5 expanding folds) instead of a single split
- Ridge, Random Forest, Gradient Boosting, optional XGBoost
- Richer stationary feature set (RSI, Bollinger, volume, lagged returns)
- Directional accuracy metric
- Bootstrap CI on the RMSE difference vs baseline
- Long/flat backtest with transaction costs
- Modular functions so app.py can import them

Author: Joshua Sun
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import create_engine

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------- 1. DATA ----------

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end,
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        raise ValueError(f"No data for {ticker} between {start} and {end}")
    return data


# ---------- 2. FEATURES ----------

# These are the candidate features. Any that can't be computed from the
# available columns (e.g. Volume if the source omits it) are dropped at
# runtime by `available_features`.
CANDIDATE_FEATURES = [
    "Return", "Return_lag1", "Return_lag2", "Return_lag5",
    "SMA_ratio_10_50", "SMA_ratio_5_20",
    "MACD", "MACD_signal", "MACD_hist",
    "Volatility_10", "Volatility_30",
    "RSI_14", "BB_position",
    "Volume_ratio",
    "Price_change_7d", "Price_change_30d",
    "DayOfWeek_sin", "DayOfWeek_cos",
]


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build STATIONARY features plus a next-day-return target.

    Every feature is dimensionless or roughly constant in distribution
    over time. Raw price levels and raw SMA dollar values are excluded
    as model inputs (SMA_10 and SMA_50 are kept only for plotting), because
    a tree-based model cannot extrapolate a non-stationary target.
    """
    df = data.copy()
    close = df["Close"]

    # --- Returns & lags ---
    df["Return"] = close.pct_change()
    for lag in (1, 2, 5):
        df[f"Return_lag{lag}"] = df["Return"].shift(lag)

    # --- Moving averages (plotting only) ---
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_50"] = close.rolling(50).mean()

    # --- Stationary MA ratios ---
    df["SMA_ratio_10_50"] = df["SMA_10"] / df["SMA_50"]
    df["SMA_ratio_5_20"] = close.rolling(5).mean() / close.rolling(20).mean()

    # --- MACD family ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # --- Volatility at two horizons ---
    df["Volatility_10"] = df["Return"].rolling(10).std()
    df["Volatility_30"] = df["Return"].rolling(30).std()

    # --- RSI(14) ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # --- Bollinger Band position (0 = lower band, 1 = upper band) ---
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    df["BB_position"] = (close - lower) / (upper - lower).replace(0, np.nan)

    # --- Volume ratio vs 20-day average ---
    if "Volume" in df.columns and df["Volume"].notna().any():
        vol_ma20 = df["Volume"].rolling(20).mean()
        df["Volume_ratio"] = df["Volume"] / vol_ma20.replace(0, np.nan)

    # --- Multi-horizon price changes ---
    df["Price_change_7d"] = close.pct_change(7)
    df["Price_change_30d"] = close.pct_change(30)

    # --- Cyclical day-of-week encoding (Mon..Fri) ---
    dow = pd.Series(df.index.dayofweek, index=df.index)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * dow / 5)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * dow / 5)

    # --- Target: next-day return ---
    df["Target"] = df["Return"].shift(-1)

    return df.dropna()


def available_features(df: pd.DataFrame) -> list:
    """Candidate features that actually exist and are fully numeric."""
    cols = []
    for c in CANDIDATE_FEATURES:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# ---------- 3. MODELS ----------

def build_models(random_state: int = 42) -> dict:
    """Return {name: (estimator, needs_scaling)}."""
    models = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge (alpha=1.0)": (Ridge(alpha=1.0), True),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=300, max_depth=5, min_samples_leaf=5,
                random_state=random_state, n_jobs=-1,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                random_state=random_state,
            ),
            False,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                random_state=random_state, n_jobs=-1, verbosity=0,
            ),
            False,
        )
    return models


# ---------- 4. WALK-FORWARD EVALUATION ----------

def walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    models: Optional[dict] = None,
) -> dict:
    """
    Expanding-window walk-forward validation.

    Fold 0 trains on the first chunk and tests on the next.
    Fold 1 trains on the first + second chunk and tests on the next, etc.
    This is the honest alternative to a single train/test split on
    time-series data: every fold only ever trains on the past and
    evaluates on the future.

    Returns a dict with:
        summary        -- per-model mean/std across folds
        per_fold       -- tidy per-fold metrics
        y_true         -- concatenated test targets
        preds          -- {model_name: concatenated predictions}
        fold_sizes     -- list of (n_train, n_test)
    """
    models = models or build_models(random_state)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rows = []
    y_test_all = []
    preds_by_model = {name: [] for name in models}
    preds_by_model["Baseline (return=0)"] = []
    fold_sizes = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        fold_sizes.append((len(train_idx), len(test_idx)))

        # Baseline: predict return = 0
        base_pred = np.zeros(len(y_te))
        rows.append({
            "fold": fold, "model": "Baseline (return=0)",
            "RMSE": float(np.sqrt(mean_squared_error(y_te, base_pred))),
            "MAE":  float(mean_absolute_error(y_te, base_pred)),
            "R2":   float(r2_score(y_te, base_pred)),
            "DirAcc": np.nan,
            "n_train": len(train_idx), "n_test": len(test_idx),
        })
        preds_by_model["Baseline (return=0)"].append(
            pd.Series(base_pred, index=y_te.index))

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        for name, (est, needs_scale) in models.items():
            if needs_scale:
                est.fit(X_tr_sc, y_tr)
                pred = est.predict(X_te_sc)
            else:
                est.fit(X_tr, y_tr)
                pred = est.predict(X_te)

            rows.append({
                "fold": fold, "model": name,
                "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                "MAE":  float(mean_absolute_error(y_te, pred)),
                "R2":   float(r2_score(y_te, pred)),
                "DirAcc": float((np.sign(pred) == np.sign(y_te)).mean()),
                "n_train": len(train_idx), "n_test": len(test_idx),
            })
            preds_by_model[name].append(pd.Series(pred, index=y_te.index))

        y_test_all.append(y_te)

    per_fold = pd.DataFrame(rows)
    summary = (
        per_fold.groupby("model")[["RMSE", "MAE", "R2", "DirAcc"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.round(4).reset_index()

    y_true = pd.concat(y_test_all)
    preds = {k: pd.concat(v) for k, v in preds_by_model.items()}

    return {
        "summary": summary,
        "per_fold": per_fold,
        "y_true": y_true,
        "preds": preds,
        "fold_sizes": fold_sizes,
    }


# ---------- 5. STATISTICAL COMPARISON ----------

def bootstrap_rmse_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 2000,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap the distribution of (RMSE_a - RMSE_b).
    Positive mean means pred_a is worse (higher RMSE).
    The 95% interval tells you whether the difference is statistically
    distinguishable from zero: if the interval contains 0, it isn't.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rmse_a = np.sqrt(np.mean((y_true[idx] - pred_a[idx]) ** 2))
        rmse_b = np.sqrt(np.mean((y_true[idx] - pred_b[idx]) ** 2))
        diffs[i] = rmse_a - rmse_b

    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower":  float(np.percentile(diffs, 2.5)),
        "ci_upper":  float(np.percentile(diffs, 97.5)),
        "significant": bool(
            np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0
        ),
    }


# ---------- 6. BACKTEST ----------

def backtest_long_flat(
    y_true: np.ndarray,
    preds: np.ndarray,
    cost_bps: float = 5.0,
) -> dict:
    """
    Long/flat strategy: go long when predicted return > 0, else stay flat.
    `cost_bps` is one-way transaction cost in basis points, charged
    whenever the position changes.

    y_true[i] is the return realized on day i+1 (the target of the model
    trained on day i), so position[i] * y_true[i] is the return earned by
    holding the position implied by prediction[i].
    """
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)

    positions = (preds > 0).astype(float)
    prev = np.concatenate([[0.0], positions[:-1]])
    cost = np.abs(positions - prev) * (cost_bps / 1e4)

    strat_ret = positions * y_true - cost
    bh_ret = y_true.copy()
    bh_ret[0] -= cost_bps / 1e4  # one entry cost for buy-and-hold

    cum_strat = np.cumprod(1.0 + strat_ret)
    cum_bh = np.cumprod(1.0 + bh_ret)

    def _stats(r):
        ann = 252
        mean = r.mean() * ann
        vol = r.std() * np.sqrt(ann)
        sharpe = mean / vol if vol > 0 else 0.0
        return {"ann_return": float(mean), "ann_vol": float(vol),
                "sharpe": float(sharpe)}

    return {
        "positions": positions,
        "strategy_returns": strat_ret,
        "bh_returns": bh_ret,
        "cum_strategy": cum_strat,
        "cum_bh": cum_bh,
        "strategy_stats": _stats(strat_ret),
        "bh_stats": _stats(bh_ret),
        "n_trades": int(np.abs(np.diff(np.concatenate([[0], positions]))).sum()),
    }


# ---------- 7. FEATURE IMPORTANCE ----------

def compute_feature_importance(
    data: pd.DataFrame,
    feature_cols: list,
    models: Optional[dict] = None,
    train_frac: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit each model on the chronological training portion and return
    normalized feature importances. Linear models use |coef|;
    tree models use feature_importances_.
    """
    models = models or build_models(random_state)
    X = data[feature_cols]
    y = data["Target"]
    split = int(len(X) * train_frac)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)

    rows = []
    for name, (est, needs_scale) in models.items():
        if needs_scale:
            est.fit(X_tr_sc, y_tr)
            vals = np.abs(getattr(est, "coef_", None))
            if vals is None:
                continue
        elif hasattr(est, "feature_importances_"):
            est.fit(X_tr, y_tr)
            vals = est.feature_importances_
        else:
            continue

        total = float(np.sum(vals)) or 1.0
        for f, v in zip(feature_cols, vals):
            rows.append({"model": name, "feature": f,
                         "importance": float(v) / total})

    return pd.DataFrame(rows)


# ---------- 8. PLOTTING HELPERS ----------

def plot_price_and_predictions(
    data: pd.DataFrame,
    y_true: pd.Series,
    preds: dict,
    ticker: str,
    save_path: str = "prediction_results.png",
):
    """Two-panel matplotlib figure: reconstructed prices and raw returns."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Panel 1: reconstructed next-day prices ---
    split_idx = len(data) - len(y_true)
    current_prices = data["Close"].iloc[split_idx:].values
    actual_prices = current_prices * (1 + y_true.values)
    axes[0].plot(y_true.index, actual_prices, label="Actual", linewidth=2)
    for name, pred in preds.items():
        if name.startswith("Baseline"):
            axes[0].plot(y_true.index, current_prices,
                         label="Baseline (= today's price)",
                         linestyle="--", alpha=0.6)
        else:
            axes[0].plot(y_true.index, current_prices * (1 + pred.values),
                         label=name, alpha=0.8)
    axes[0].set_title(f"{ticker}: Reconstructed Next-Day Price")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # --- Panel 2: raw returns (where the differences actually live) ---
    axes[1].plot(y_true.index, y_true.values, label="Actual return",
                 linewidth=1.5, alpha=0.7)
    for name, pred in preds.items():
        if name.startswith("Baseline"):
            continue
        axes[1].plot(pred.index, pred.values, label=f"{name} (pred)",
                     alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title(f"{ticker}: Predicted vs Actual Next-Day Return")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Return")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------- 9. DATABASE ----------

def save_to_db(data: pd.DataFrame, table_name: str = "stock_data",
               db_path: str = "stocks.db"):
    engine = create_engine(f"sqlite:///{db_path}")
    data.to_sql(table_name, engine, if_exists="replace", index=True)
    return db_path, table_name


# ---------- 10. MAIN ----------

def run_pipeline(
    ticker: str = "NVDA",
    start: str = "2020-01-01",
    end: str = "2025-01-01",
    n_splits: int = 5,
    cost_bps: float = 5.0,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """End-to-end run. Returns a result dict for app.py to consume."""
    raw = fetch_data(ticker, start, end)
    data = engineer_features(raw)
    feature_cols = available_features(data)

    if not feature_cols:
        raise ValueError("No usable features after engineering.")

    X = data[feature_cols]
    y = data["Target"]

    wf = walk_forward_evaluate(X, y, n_splits=n_splits,
                               random_state=random_state)

    y_true = wf["y_true"]
    preds = wf["preds"]

    # Bootstrap CI for each non-baseline model vs the baseline
    base = preds["Baseline (return=0)"].values
    bootstrap = {}
    for name, pred in preds.items():
        if name.startswith("Baseline"):
            continue
        bootstrap[name] = bootstrap_rmse_difference(
            y_true.values, pred.values, base, random_state=random_state)

    # Backtest each non-baseline model
    backtests = {}
    for name, pred in preds.items():
        if name.startswith("Baseline"):
            continue
        backtests[name] = backtest_long_flat(y_true.values, pred.values,
                                             cost_bps=cost_bps)

    importances = compute_feature_importance(data, feature_cols,
                                             random_state=random_state)

    if verbose:
        print(f"\n=== {ticker} walk-forward summary ({n_splits} folds) ===")
        print(wf["summary"].to_string(index=False))
        print("\n=== Bootstrap RMSE diff vs baseline (positive = worse) ===")
        for name, b in bootstrap.items():
            sig = "significant" if b["significant"] else "NOT significant"
            print(f"  {name:24s}  diff={b['mean_diff']:+.5f}  "
                  f"95% CI [{b['ci_lower']:+.5f}, {b['ci_upper']:+.5f}]  ({sig})")
        print("\n=== Backtest (long/flat, cost = {} bps) ===".format(cost_bps))
        for name, bt in backtests.items():
            s, b = bt["strategy_stats"], bt["bh_stats"]
            print(f"  {name:24s}  "
                  f"ann_ret={s['ann_return']:+.4f}  sharpe={s['sharpe']:+.2f}  "
                  f"trades={bt['n_trades']}  |  "
                  f"buy&hold ann_ret={b['ann_return']:+.4f}  sharpe={b['sharpe']:+.2f}")

    plot_path = plot_price_and_predictions(data, y_true, preds, ticker)
    save_to_db(data, f"{ticker}_stock")

    return {
        "ticker": ticker,
        "data": data,
        "feature_cols": feature_cols,
        "walk_forward": wf,
        "bootstrap": bootstrap,
        "backtests": backtests,
        "importances": importances,
        "plot_path": plot_path,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stock return prediction v4")
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--cost-bps", type=float, default=5.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        ticker=args.ticker, start=args.start, end=args.end,
        n_splits=args.splits, cost_bps=args.cost_bps,
    )