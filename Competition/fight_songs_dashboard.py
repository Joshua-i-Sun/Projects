import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_data():
    df = pd.read_csv("fight-songs-updated.csv")
    
    df = df.replace(["Unknown", "unknown"], pd.NA)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")  
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df["sec_duration"] = pd.to_numeric(df["sec_duration"], errors="coerce")
    
    bool_cols = ["fight", "victory", "win_won", "victory_win_won", "rah", "nonsense",
                 "colors", "men", "opponents", "spelling", "official_song", "student_writer"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": True, "No": False, "Yes": True, True: True, False: False})
    
    return df

df = load_data()

st.sidebar.header("Filters")

conferences = ["All"] + sorted(df["conference"].dropna().unique().tolist())
selected_conf = st.sidebar.selectbox("Conference", conferences)

if selected_conf != "All":
    df_f = df[df["conference"] == selected_conf].copy()
else:
    df_f = df.copy()


show_student_only = st.sidebar.checkbox("Only student-written songs", value=False)
if show_student_only:
    df_f = df_f[df_f["student_writer"] == True]


min_tropes, max_tropes = int(df_f["trope_count"].min() or 0), int(df_f["trope_count"].max() or 8)
trope_range = st.sidebar.slider("Trope count range", 0, 8, (min_tropes, max_tropes))
df_f = df_f[(df_f["trope_count"] >= trope_range[0]) & (df_f["trope_count"] <= trope_range[1])]

if df_f.empty:
    st.warning("No songs match the current filters.")
    st.stop()

st.title("College Fight Songs Explorer")
st.caption("Data from fight-songs-updated.csv • ACC, Big 12, Big Ten, Pac-12, SEC + Notre Dame")


cols = st.columns(5)
cols[0].metric("Songs", len(df_f), delta=None)
cols[1].metric("Avg BPM", f"{df_f['bpm'].mean():.0f}" if not df_f["bpm"].isna().all() else "—")
cols[2].metric("Avg Length", f"{df_f['sec_duration'].mean():.0f} s" if not df_f["sec_duration"].isna().all() else "—")
cols[3].metric("Avg Tropes", f"{df_f['trope_count'].mean():.1f}")
cols[4].metric("% Official", f"{df_f['official_song'].mean():.0%}" if "official_song" in df_f else "—")


tab1, tab2, tab3, tab4 = st.tabs(["Tropes", "Tempo × Duration", "Timeline", "Schools"])

with tab1:
    st.subheader("Lyrical Tropes")

    trope_cols = ["fight", "victory", "win_won", "rah", "nonsense", "colors", "men", "opponents", "spelling"]
    trope_cols = [c for c in trope_cols if c in df_f.columns]

    if trope_cols:
        melted = df_f.melt(
            id_vars=["school", "conference"],
            value_vars=trope_cols,
            var_name="Trope",
            value_name="Present"
        )
        melted = melted[melted["Present"] == True]

        fig_tropes = px.histogram(
            melted, x="Trope", color="conference",
            barmode="group", title="Count of Songs Containing Each Trope",
            height=480
        )
        fig_tropes.update_layout(xaxis_title=None)
        st.plotly_chart(fig_tropes, use_container_width=True)

    st.subheader("Trope Count Distribution")
    fig_hist = px.histogram(
        df_f, x="trope_count", color="conference",
        title="How many tropes per song?", nbins=9,
        height=420
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("BPM vs Duration (bubble = # of tropes)")

    fig_scatter = px.scatter(
        df_f,
        x="sec_duration",
        y="bpm",
        size="trope_count",
        color="conference",
        hover_name="school",
        hover_data=["song_name", "year", "trope_count"],
        title="Tempo vs Length — bigger = more lyrical tropes",
        height=580
    )
    fig_scatter.update_traces(marker_opacity=0.75)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Fight Songs by Decade")

    decade_df = df_f["decade"].value_counts().sort_index().reset_index()
    decade_df.columns = ["Decade", "Count"]

    fig_line = px.line(
        decade_df, x="Decade", y="Count", markers=True,
        title="Number of fight songs composed per decade"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Raw decade breakdown")
    st.dataframe(decade_df, hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Schools sorted by trope count")

    school_stats = df_f.groupby("school").agg({
        "trope_count": "mean",
        "bpm": "mean",
        "sec_duration": "mean",
        "conference": "first",
        "song_name": "count"
    }).rename(columns={"song_name": "# Songs"}).reset_index()

    school_stats = school_stats.sort_values("trope_count", ascending=False)
    st.dataframe(
        school_stats.style.format({
            "trope_count": "{:.1f}",
            "bpm": "{:.0f}",
            "sec_duration": "{:.0f} s"
        }),
        use_container_width=True
    )

with st.expander("See full filtered data table"):
    st.dataframe(
        df_f.sort_values(["conference", "trope_count"], ascending=[True, False]),
        use_container_width=True
    )

st.markdown("---")
st.caption("Built with Streamlit • Run with:  `streamlit run fight_songs_dashboard.py`")