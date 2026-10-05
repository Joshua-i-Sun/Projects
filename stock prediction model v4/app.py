"""Streamlit UI for the Stock Return Prediction Model v4.

Run with:  streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import stock_prediction_v4 as sp

st.set_page_config(
    page_title="Stock Return Prediction v4",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Ticker", "NVDA").upper().strip()
    col_a, col_b = st.columns(2)
    with col_a:
        start = st.date_input("Start", pd.Timestamp("2020-01-01"))
    with col_b:
        end = st.date_input("End", pd.Timestamp("2025-01-01"))

    n_splits = st.slider("Walk-forward folds", 3, 8, 5)
    cost_bps = st.slider("Transaction cost (bps)", 0, 50, 5, 1)
    n_boot = st.slider("Bootstrap iterations", 500, 5000, 2000, 500)

    run_btn = st.button("▶️  Run pipeline", type="primary",
                        use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Stock Return Prediction — v4")
st.caption(
    "Walk-forward validated next-day return prediction with "
    "bootstrap significance testing and a long/flat backtest."
)


@st.cache_data(show_spinner=False)
def _cached_fetch(ticker, start, end):
    return sp.fetch_data(ticker, str(start), str(end))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

if run_btn:
    st.session_state.error = None
    st.session_state.result = None
    try:
        with st.spinner(f"Fetching {ticker}…"):
            raw = _cached_fetch(ticker, start, end)
        with st.spinner("Engineering features…"):
            data = sp.engineer_features(raw)
            feature_cols = sp.available_features(data)
        with st.spinner(f"Walk-forward validation ({n_splits} folds)…"):
            X, y = data[feature_cols], data["Target"]
            wf = sp.walk_forward_evaluate(X, y, n_splits=n_splits)
        with st.spinner("Bootstrapping RMSE differences…"):
            base = wf["preds"]["Baseline (return=0)"].values
            bootstrap = {}
            for name, pred in wf["preds"].items():
                if name.startswith("Baseline"):
                    continue
                bootstrap[name] = sp.bootstrap_rmse_difference(
                    wf["y_true"].values, pred.values, base, n_boot=n_boot)
        with st.spinner("Backtesting…"):
            backtests = {}
            for name, pred in wf["preds"].items():
                if name.startswith("Baseline"):
                    continue
                backtests[name] = sp.backtest_long_flat(
                    wf["y_true"].values, pred.values, cost_bps=cost_bps)
        with st.spinner("Computing feature importances…"):
            importances = sp.compute_feature_importance(data, feature_cols)

        st.session_state.result = {
            "ticker": ticker, "data": data, "feature_cols": feature_cols,
            "walk_forward": wf, "bootstrap": bootstrap,
            "backtests": backtests, "importances": importances,
        }
    except Exception as exc:
        st.session_state.error = f"{type(exc).__name__}: {exc}"

if st.session_state.error:
    st.error(st.session_state.error)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.result:
    r = st.session_state.result
    wf = r["walk_forward"]
    data = r["data"]
    summary = wf["summary"]

    # --- Top metrics (best model by RMSE) ---
    non_base = summary[summary["model"] != "Baseline (return=0)"]
    best_rmse = non_base.sort_values("RMSE_mean").iloc[0]
    base_row = summary[summary["model"] == "Baseline (return=0)"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Features", len(r["feature_cols"]))
    c3.metric("Best RMSE (mean)", f"{best_rmse['RMSE_mean']:.5f}",
              delta=f"{best_rmse['RMSE_mean'] - base_row['RMSE_mean']:+.5f} vs base",
              delta_color="inverse")
    c4.metric("Best model", best_rmse["model"])

    st.divider()

    tab_pred, tab_metrics, tab_import, tab_backtest, tab_dl = st.tabs(
        ["📊 Price & predictions", "🧪 Metrics", "🔍 Feature importance",
         "💰 Backtest", "💾 Download"]
    )

    # ---------------- Predictions ----------------
    with tab_pred:
        y_true = wf["y_true"]
        preds = wf["preds"]

        split_idx = len(data) - len(y_true)
        current_prices = data["Close"].iloc[split_idx:].values
        dates = y_true.index

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Reconstructed next-day price",
                            "Predicted vs actual next-day return"),
            vertical_spacing=0.12,
        )
        fig.add_trace(go.Scatter(x=dates, y=current_prices * (1 + y_true.values),
                                 name="Actual", line=dict(width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=current_prices,
                                 name="Baseline (=today's price)",
                                 line=dict(dash="dash", width=1)), row=1, col=1)
        for name, pred in preds.items():
            if name.startswith("Baseline"):
                continue
            fig.add_trace(go.Scatter(
                x=dates, y=current_prices * (1 + pred.values),
                name=name, opacity=0.75), row=1, col=1)

        fig.add_trace(go.Scatter(x=dates, y=y_true.values,
                                 name="Actual return", opacity=0.6), row=2, col=1)
        for name, pred in preds.items():
            if name.startswith("Baseline"):
                continue
            fig.add_trace(go.Scatter(
                x=dates, y=pred.values, name=f"{name} (pred)",
                opacity=0.7), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="black", width=0.5), row=2, col=1)

        fig.update_layout(height=720, hovermode="x unified",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Reconstructed prices look almost identical because predicted "
            "returns are small — that's the honest result, not a bug. "
            "The lower panel shows where the differences actually live."
        )

    # ---------------- Metrics ----------------
    with tab_metrics:
        st.subheader("Walk-forward summary (mean ± std across folds)")
        st.dataframe(summary, use_container_width=True)

        st.subheader("Per-fold metrics")
        per_fold = wf["per_fold"]
        pivot = per_fold.pivot_table(
            index="fold", columns="model",
            values=["RMSE", "MAE", "DirAcc"],
        ).round(5)
        st.dataframe(pivot, use_container_width=True)

        # Metric bar chart
        melt = summary.melt(id_vars="model",
                            value_vars=[c for c in summary.columns
                                        if c.endswith("_mean")],
                            var_name="metric", value_name="value")
        fig = px.bar(melt, x="model", y="value", color="metric",
                     barmode="group", title="Mean metric by model")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Bootstrap: is the improvement over baseline real?")
        rows = []
        for name, b in r["bootstrap"].items():
            rows.append({
                "model": name,
                "mean RMSE diff": round(b["mean_diff"], 5),
                "95% CI low": round(b["ci_lower"], 5),
                "95% CI high": round(b["ci_upper"], 5),
                "significant?": "Yes" if b["significant"] else "No",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(
            "Positive diff = model has higher (worse) RMSE than the "
            "baseline. If the 95% interval contains 0, the difference is "
            "not statistically distinguishable."
        )

    # ---------------- Feature importance ----------------
    with tab_import:
        imp = r["importances"]
        if imp.empty:
            st.info("No importances available.")
        else:
            model_choice = st.selectbox(
                "Model", sorted(imp["model"].unique()))
            sub = (imp[imp["model"] == model_choice]
                   .sort_values("importance", ascending=True))
            fig = px.bar(sub, x="importance", y="feature",
                         orientation="h",
                         title=f"Feature importance — {model_choice}")
            fig.update_layout(height=max(400, 22 * len(sub)))
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- Backtest ----------------
    with tab_backtest:
        bt_names = sorted(r["backtests"].keys())
        if not bt_names:
            st.info("No backtests available.")
        else:
            chosen = st.selectbox("Strategy model", bt_names)
            bt = r["backtests"][chosen]
            y_true = wf["y_true"]
            dates = y_true.index

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=bt["cum_bh"],
                                     name="Buy & hold"))
            fig.add_trace(go.Scatter(x=dates, y=bt["cum_strategy"],
                                     name=f"Strategy ({chosen})"))
            fig.update_layout(
                title=f"Cumulative equity — long/flat ({cost_bps} bps cost)",
                yaxis_title="Equity (start = 1.0)",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

            s = bt["strategy_stats"]
            b = bt["bh_stats"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategy ann. return", f"{s['ann_return']:.2%}")
            c2.metric("Buy & hold ann. return", f"{b['ann_return']:.2%}")
            c3.metric("Strategy Sharpe", f"{s['sharpe']:.2f}")
            c4.metric("Trades", bt["n_trades"])

            st.caption(
                "A long/flat strategy trades on predicted sign. Costs are "
                "charged whenever the position changes. Buy-and-hold is "
                "charged one entry cost."
            )

    # ---------------- Download ----------------
    with tab_dl:
        st.subheader("Download outputs")
        st.download_button(
            "⬇️  walk_forward_summary.csv",
            summary.to_csv(index=False).encode(),
            file_name="walk_forward_summary.csv",
        )
        st.download_button(
            "⬇️  per_fold_metrics.csv",
            wf["per_fold"].to_csv(index=False).encode(),
            file_name="per_fold_metrics.csv",
        )
        st.download_button(
            "⬇️  feature_importances.csv",
            r["importances"].to_csv(index=False).encode(),
            file_name="feature_importances.csv",
        )
        st.download_button(
            "⬇️  engineered_features.csv",
            data.to_csv(index=True).encode(),
            file_name=f"{r['ticker']}_engineered.csv",
        )
else:
    st.info(
        "Set the ticker and date range in the sidebar, then click "
        "**Run pipeline**. Walk-forward validation and bootstrapping "
        "take 20–60 seconds depending on the date range."
    )