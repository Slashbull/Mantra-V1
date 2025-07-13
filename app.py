"""
app.py - M.A.N.T.R.A. Locked Streamlit Dashboard (Final Version)
================================================================
100% data-driven, robust, simple, and beautiful.
Never needs an upgrade. All config/logic in constants.py and your Google Sheets.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from constants import SHEET_URLS, SIGNAL_COLORS, RISK_COLORS, SCHEMA_DOC
from data_loader import load_all_data
from signals import SignalEngine

st.set_page_config(page_title="M.A.N.T.R.A.", layout="wide")

# --- MODERN DARK UI THEME ---
st.markdown("""
    <style>
        body, .stApp { background-color: #161b22 !important; color: #d1d5da; }
        .reportview-container .main { color: #d1d5da; background: #161b22; }
        .stDataFrame th, .stDataFrame td { color: #d1d5da !important; }
        .css-18ni7ap { background: #0e1117 !important; }
        .css-1v0mbdj {background: #21262d;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("M.A.N.T.R.A.")
st.sidebar.markdown("**Personal Indian Stock Intelligence Engine**")
st.sidebar.caption("Locked. 100% data/config driven. No code changes ever needed.")

with st.sidebar.expander("📄 Data Schema & Doc", expanded=False):
    st.code(SCHEMA_DOC)

# --- LOAD DATA ---
with st.spinner("Loading latest data..."):
    watchlist_df, returns_df, sector_df, status = load_all_data()

if not status.get("success", False):
    st.error("🚫 Data load error(s): " + "; ".join(status.get("errors", [])))
    st.stop()

# --- SIGNAL LOGIC ---
with st.spinner("Calculating signals..."):
    df_signals = SignalEngine.calculate_all_signals(watchlist_df, sector_df=sector_df, regime="balanced")

# --- FILTERS ---
st.sidebar.markdown("### Filters")
signal_options = ["All"] + sorted(df_signals["signal"].unique())
chosen_signal = st.sidebar.selectbox("Signal", signal_options)
sector_options = ["All"] + sorted(df_signals["sector"].dropna().unique())
chosen_sector = st.sidebar.selectbox("Sector", sector_options)
risk_options = ["All"] + sorted(df_signals["risk"].unique())
chosen_risk = st.sidebar.selectbox("Risk", risk_options)

filtered = df_signals.copy()
if chosen_signal != "All":
    filtered = filtered[filtered["signal"] == chosen_signal]
if chosen_sector != "All":
    filtered = filtered[filtered["sector"] == chosen_sector]
if chosen_risk != "All":
    filtered = filtered[filtered["risk"] == chosen_risk]

# --- KPI CARDS ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Universe", f"{len(df_signals)}")
kpi2.metric("Buy Ideas", f"{(df_signals['signal'].isin(['BUY','STRONG_BUY']).sum())}")
kpi3.metric("Strong Buys", f"{(df_signals['signal'] == 'STRONG_BUY').sum()}")
kpi4.metric("Avoids", f"{(df_signals['signal'] == 'AVOID').sum()}")

# --- TOP SIGNAL CARDS ---
st.markdown("## 🚦 Top Opportunities")
show_df = filtered.head(12)
cols = st.columns(4)
for idx, row in show_df.iterrows():
    with cols[idx % 4]:
        st.markdown(f"""
        <div style="border-radius:12px;background:#21262d;padding:16px;margin-bottom:12px;box-shadow:0 0 4px #2ea043;">
            <div style="font-size:18px;font-weight:bold;color:#58a6ff;">{row['ticker']}</div>
            <div style="font-size:13px;">{row['company_name']}</div>
            <div><b>Signal:</b> <span style="color:{SIGNAL_COLORS.get(row['signal'],'#fff')};">{row['signal']}</span></div>
            <div><b>Score:</b> {int(row['score'])} | <b>Risk:</b> <span style="color:{RISK_COLORS.get(row['risk'],'#fff')};">{row['risk']}</span></div>
            <div><b>Sector:</b> {row['sector']}</div>
            <div style="font-size:11px;margin-top:7px;">{row['reason']}</div>
        </div>
        """, unsafe_allow_html=True)

# --- FULL TABLE ---
st.markdown("### 🔬 All Signals Table")
table_cols = ["ticker", "company_name", "sector", "price", "signal", "score", "risk", "reason"]
table_cols += [c for c in ["momentum", "value", "volume", "technical"] if c in filtered.columns]
st.dataframe(
    filtered[table_cols],
    use_container_width=True,
    hide_index=True
)

# --- EXPORT BUTTON ---
st.download_button(
    label="⬇️ Export as CSV",
    data=filtered.to_csv(index=False),
    file_name="mantra_signals_export.csv",
    mime="text/csv"
)

# --- SECTOR HEATMAP ---
if not sector_df.empty and "sector_ret_30d" in sector_df.columns:
    st.markdown("### 🟩 Sector Momentum (30D Return)")
    chart = px.bar(
        sector_df.sort_values("sector_ret_30d", ascending=False),
        x="sector", y="sector_ret_30d", color="sector_ret_30d",
        color_continuous_scale="RdYlGn",
        labels={"sector_ret_30d": "30D Return (%)"}, height=350
    )
    st.plotly_chart(chart, use_container_width=True)

# --- FOOTER ---
st.markdown("""
    <hr style='border-color:#21262d;'>
    <div style='color:#666;font-size:12px;text-align:center;'>
        <b>M.A.N.T.R.A.</b> &copy; {year} — Locked Version. No Upgrades. | Precision Over Noise.<br>
        <span style='font-size:11px;'>All logic and scoring are 100% data/config driven. For personal use only.</span>
    </div>
""".format(year=pd.Timestamp.today().year), unsafe_allow_html=True)
