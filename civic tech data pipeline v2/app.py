"""Streamlit front-end for the Civic Tech Data Pipeline.

Run with:  streamlit run app.py
"""
import io
import os
import json
import traceback

import numpy as np
import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(
    page_title="Civic Tech Data Pipeline",
    page_icon="🏙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    mode = st.radio(
        "Mode",
        ["Offline (fixtures)", "Live (network)"],
        index=0,
        help="Offline mode reads bundled JSON fixtures; live mode hits the "
             "real Census/HUD/FCC endpoints.",
    )

    fixture_dir = None
    if mode.startswith("Offline"):
        fixture_dir = st.text_input("Fixture directory", "tests/fixtures")

    join_mode = st.radio(
        "Source combination",
        ["join", "dedup"],
        index=0,
        help="join: coalesce complementary fields across sources on geo_id.\n"
             "dedup: keep only the best-scoring source per geo_id.",
    )

    z_cutoff = st.slider(
        "Disparity z-score cutoff", 0.5, 3.0, 1.5, 0.1,
        help="Rows with |composite_disparity_score| >= cutoff are flagged.",
    )
    workers = st.slider("Parallel workers", 1, 8, 4)
    output_dir = st.text_input("Output directory", "civic_output")

    run_btn = st.button("▶️  Run pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏙️ Civic Tech Data Pipeline")
st.caption(
    "Parallel ETL over public socio-economic Open Data sources, with "
    "statistical disparity flagging and a downloadable feature extract."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_btn:
    st.session_state.error = None
    st.session_state.result = None

    # Basic pre-flight checks
    if mode.startswith("Offline") and not os.path.isdir(fixture_dir or ""):
        st.session_state.error = (
            f"Fixture directory not found: {fixture_dir!r}. "
            "Make sure tests/fixtures/*.json exist, or switch to Live mode."
        )
    else:
        progress = st.progress(0, text="Building config…")
        try:
            cfg = pl.build_config(
                output_dir=output_dir,
                fixture_dir=fixture_dir if mode.startswith("Offline") else None,
                z_cutoff=z_cutoff,
                workers=workers,
                join_mode=join_mode,
            )
            progress.progress(20, text="Extracting sources…")
            p = pl.ETLPipeline(cfg)

            frames = p.extract()
            progress.progress(55, text="Transforming…")
            df, meta = p.transform(frames)

            progress.progress(80, text="Loading outputs…")
            paths = p.load(df, meta)

            progress.progress(100, text="Done.")
            st.session_state.result = {"df": df, "meta": meta, "paths": paths}
        except Exception as exc:
            st.session_state.error = f"{type(exc).__name__}: {exc}"
            st.session_state.traceback = traceback.format_exc()

if st.session_state.error:
    st.error(st.session_state.error)
    with st.expander("Full traceback"):
        st.code(st.session_state.get("traceback", ""), language="text")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.result:
    df = st.session_state.result["df"]
    meta = st.session_state.result["meta"]
    paths = st.session_state.result["paths"]

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{meta['row_count']:,}")
    c2.metric("Sources", len(meta["sources"]))
    c3.metric("Flagged", f"{meta['disparity_flagged']:,}")
    c4.metric("Columns", len(df.columns))

    st.divider()

    tab_data, tab_disparity, tab_quality, tab_export = st.tabs(
        ["📊 Data", "🚩 Disparities", "🧪 Data quality", "💾 Export"]
    )

    # -------------------- Data tab --------------------
    with tab_data:
        st.subheader("Feature extract")
        st.caption(
            f"Join mode: **{meta['join_mode']}** · "
            f"Sources: {', '.join(meta['sources']) or '—'}"
        )

        # Column selector
        default_cols = [c for c in df.columns if not c.startswith("__")][:12]
        chosen = st.multiselect(
            "Columns to display",
            options=list(df.columns),
            default=default_cols,
        )
        st.dataframe(df[chosen] if chosen else df, use_container_width=True, height=420)

    # -------------------- Disparity tab --------------------
    with tab_disparity:
        if "composite_disparity_score" not in df.columns:
            st.info("No disparity score computed for this dataset.")
        else:
            plot_df = df.dropna(subset=["composite_disparity_score"]).copy()
            if plot_df.empty:
                st.warning("All composite disparity scores are NaN.")
            else:
                geo_col = "geo_id" if "geo_id" in plot_df.columns else None
                label_col = (
                    "county_name" if "county_name" in plot_df.columns
                    else geo_col if geo_col else None
                )

                # Sort by composite score so the bars read cleanly
                plot_df = plot_df.sort_values(
                    "composite_disparity_score", ascending=True
                )
                x_labels = (
                    plot_df[label_col].astype(str)
                    if label_col else plot_df.index.astype(str)
                )

                st.subheader("Composite disparity score by geography")
                st.caption(
                    "Positive = more disadvantaged than average. "
                    "Red bars exceed the z-score cutoff."
                )

                chart_df = pd.DataFrame({
                    "geography": x_labels.values,
                    "score": plot_df["composite_disparity_score"].values,
                    "flagged": plot_df["disparity_flag"].values,
                })
                st.bar_chart(
                    chart_df.set_index("geography")["score"],
                    use_container_width=True,
                    color="#d62728",
                )

                # Flagged-only table
                st.subheader("Flagged rows")
                flagged = df[df["disparity_flag"] == True]  # noqa: E712
                if flagged.empty:
                    st.success("No rows exceeded the z-score cutoff.")
                else:
                    show_cols = [c for c in [
                        "geo_id", "county_name",
                        "composite_disparity_score",
                        "median_income", "poverty_rate",
                        "unemployment_rate",
                        "internet_access_rate",
                        "broadband_subscription_rate",
                    ] if c in flagged.columns]
                    st.dataframe(
                        flagged[show_cols].sort_values(
                            "composite_disparity_score", ascending=False
                        ),
                        use_container_width=True,
                    )

    # -------------------- Quality tab --------------------
    with tab_quality:
        st.subheader("Column-level data quality audit")
        report = pl.build_quality_report(df)
        st.dataframe(report, use_container_width=True, height=420)

        nulls_only = report[report["null_pct"] > 0]
        if nulls_only.empty:
            st.success("No null values in any column. ")
        else:
            st.warning(
                f"{len(nulls_only)} of {len(report)} columns have null values."
            )

    # -------------------- Export tab --------------------
    with tab_export:
        st.subheader("Run metadata")
        st.json(meta)

        st.subheader("Download outputs")
        for name, path in paths.items():
            if not path or not os.path.exists(path):
                st.write(f"- **{name}**: not produced")
                continue
            with open(path, "rb") as fh:
                st.download_button(
                    label=f"⬇️  {name} — {os.path.basename(path)}",
                    data=fh.read(),
                    file_name=os.path.basename(path),
                    mime="application/octet-stream",
                    use_container_width=True,
                )

        # Also allow downloading the quality report as CSV
        st.download_button(
            label="⬇️  data_quality_report.csv",
            data=pl.build_quality_report(df).to_csv(index=False).encode(),
            file_name="data_quality_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info(
        "Configure the pipeline in the sidebar and click **Run pipeline** to begin. "
        "Offline mode uses the bundled fixtures under `tests/fixtures/` and needs no network access."
    )