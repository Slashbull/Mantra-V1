#!/usr/bin/env python3
# =============================================================================
# streamlit_dashboard.py – M.A.N.T.R.A. Main Dashboard (FINAL, BUG-FREE)
# =============================================================================
"""
Launch with:

    streamlit run streamlit_dashboard.py

Dependencies: the 5 packages in requirements.txt only.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from constants import CACHE_DURATION_MINUTES
from data_loader import DataLoader
from signal_engine import SignalEngine
from ui_components import (
    dashboard_header,
    data_quality_badge,
    display_dataframe,
    gauge_chart,
    load_custom_css,
    metric_card,
    section_header,
    show_alert,
    signal_badge,  # exported for completeness – used inside stock_card
    stock_card,
    sector_heatmap,
)

# -----------------------------------------------------------------------------#
#  0. Logging
# -----------------------------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s", force=True
)
logger = logging.getLogger("MANTRA")

# -----------------------------------------------------------------------------#
#  1. Streamlit page config + CSS
# -----------------------------------------------------------------------------#
st.set_page_config(
    page_title="M.A.N.T.R.A. – Stock Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_custom_css()

# -----------------------------------------------------------------------------#
#  2. Session-state bootstrapping
# -----------------------------------------------------------------------------#
ss = st.session_state
ss.setdefault("data_loaded", False)
ss.setdefault("last_refresh", None)
ss.setdefault("tab", "📊 Overview")

# -----------------------------------------------------------------------------#
#  3. Lazy load + analyse data (cached 5 min)
# -----------------------------------------------------------------------------#
@st.cache_data(ttl=CACHE_DURATION_MINUTES * 60, show_spinner=False)
def _load_and_analyse():
    stocks, sectors, health = DataLoader.load_all_data()
    if health["status"] != "success":
        return None, None, health
    analysed = SignalEngine.calculate_all_signals(stocks, sectors)
    return analysed, sectors, health


# -----------------------------------------------------------------------------#
#  4. Sidebar – controls & filters
# -----------------------------------------------------------------------------#
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")

    rf_col, ts_col = st.columns(2)
    with rf_col:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with ts_col:
        if ss.last_refresh:
            mins = (datetime.now() - ss.last_refresh).seconds // 60
            st.caption(f"↻ {mins} min ago")

    st.markdown("---")

    tabs = ["📊 Overview", "🎯 Signals", "🔥 Top Picks", "📈 Sectors", "📋 Analysis"]
    ss.tab = st.radio("Navigation", tabs, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🔍 Filters")

    signal_filter = st.multiselect(
        "Signal",
        ["STRONG_BUY", "BUY", "WATCH", "NEUTRAL", "AVOID"],
        default=["STRONG_BUY", "BUY", "WATCH"],
    )
    min_score = st.slider("Min Score", 0, 100, 65, 5)
    risk_filter = st.multiselect(
        "Risk Level",
        ["Very Low", "Low", "Medium", "High", "Very High"],
        default=["Very Low", "Low", "Medium"],
    )
    min_volume = st.number_input(
        "Min Volume (shares)", 0, value=50_000, step=10_000, format="%d"
    )

# -----------------------------------------------------------------------------#
#  5. Header + data load spinner
# -----------------------------------------------------------------------------#
dashboard_header()

if not ss.data_loaded:
    with st.spinner("🔄 Loading market data..."):
        df_stocks, df_sector, health = _load_and_analyse()
        if df_stocks is None:
            st.error("❌ Data load failed. Check connection & Google-Sheet perms.")
            st.stop()
        ss.update(
            {
                "stocks_df": df_stocks,
                "sector_df": df_sector,
                "health": health,
                "data_loaded": True,
                "last_refresh": datetime.now(),
            }
        )

# -----------------------------------------------------------------------------#
#  6. Apply filters
# -----------------------------------------------------------------------------#
fdf = ss.stocks_df.copy()

if signal_filter:
    fdf = fdf[fdf["decision"].isin(signal_filter)]
fdf = fdf[fdf["composite_score"] >= min_score]
if risk_filter:
    fdf = fdf[fdf["risk_level"].isin(risk_filter)]
if "volume_1d" in fdf.columns:
    fdf = fdf[fdf["volume_1d"] >= min_volume]

data_quality_badge(ss.health["data_quality"])

# -----------------------------------------------------------------------------#
#  7. Tab: Overview
# -----------------------------------------------------------------------------#
if ss.tab == "📊 Overview":
    section_header("Market Overview", f"Showing {len(fdf)} stocks")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Total Universe", len(ss.stocks_df), icon="📊")
    with c2:
        buys = fdf["decision"].eq("BUY").sum()
        pct = buys / len(fdf) * 100 if len(fdf) else 0
        metric_card("Buy Signals", buys, delta=pct, icon="🟢")
    with c3:
        metric_card("Average Score", f"{fdf['composite_score'].mean():.1f}", icon="📈")
    with c4:
        metric_card("High Momentum", (fdf["momentum_score"] > 80).sum(), icon="🚀")
    with c5:
        breadth = (fdf["ret_1d"] > 0).mean() * 100 if len(fdf) else 0
        metric_card("Breadth", f"{breadth:.0f}%", icon="📊")

    st.markdown("---")
    section_header("🎯 Top Opportunities", "Highest conviction signals")

    top_buys = fdf[fdf["decision"] == "BUY"].nlargest(6, "opportunity_score")
    if top_buys.empty:
        show_alert("No BUY opportunities after applying filters", "info")
    else:
        cols = st.columns(3)
        for i, (_, row) in enumerate(top_buys.iterrows()):
            with cols[i % 3]:
                stock_card(row)

    st.markdown("---")
    section_header("🏭 Sector Performance")
    if not ss.sector_df.empty:
        st.plotly_chart(sector_heatmap(ss.sector_df), use_container_width=True)

# -----------------------------------------------------------------------------#
#  8. Tab: Signals
# -----------------------------------------------------------------------------#
elif ss.tab == "🎯 Signals":
    section_header("Trading Signals", f"{len(fdf)} stocks match filters")

    scounts = fdf["decision"].value_counts()
    cols = st.columns(4)
    for col, name in zip(cols, ["BUY", "WATCH", "NEUTRAL", "AVOID"]):
        with col:
            metric_card(name, scounts.get(name, 0), icon="🟢🟡⚪🔴"[["BUY","WATCH","NEUTRAL","AVOID"].index(name)])

    st.markdown("---")
    sel_cols = [
        "ticker",
        "company_name",
        "sector",
        "decision",
        "composite_score",
        "price",
        "ret_1d",
        "ret_30d",
        "pe",
        "volume_1d",
        "momentum_score",
        "value_score",
        "risk_level",
        "reasoning",
    ]
    display_dataframe(fdf.nlargest(100, "opportunity_score")[sel_cols], height=600)

    st.download_button(
        "📥 Download CSV",
        fdf[sel_cols].to_csv(index=False),
        f"mantra_signals_{datetime.now():%Y%m%d_%H%M}.csv",
        "text/csv",
    )

# -----------------------------------------------------------------------------#
#  9. Tab: Top Picks
# -----------------------------------------------------------------------------#
elif ss.tab == "🔥 Top Picks":
    section_header("Top Picks by Theme")
    tabs = st.tabs(["🚀 Momentum", "💎 Value", "📈 Growth", "🛡️ Safe"])

    # Momentum
    with tabs[0]:
        picks = fdf.query("momentum_score > 80 & decision in ['BUY','WATCH']")
        if picks.empty:
            show_alert("No momentum picks.", "info")
        else:
            for i, (_, row) in enumerate(picks.nlargest(12, "momentum_score").iterrows()):
                with st.columns(3)[i % 3]:
                    stock_card(row)

    # Value
    with tabs[1]:
        picks = fdf.query("value_score > 80 & 0 < pe < 20")
        if picks.empty:
            show_alert("No value picks.", "info")
        else:
            for i, (_, row) in enumerate(picks.nlargest(12, "value_score").iterrows()):
                with st.columns(3)[i % 3]:
                    stock_card(row)

    # Growth
    with tabs[2]:
        picks = fdf.query("fundamental_score > 70 & eps_change_pct > 20")
        if picks.empty:
            show_alert("No growth picks.", "info")
        else:
            for i, (_, row) in enumerate(picks.nlargest(12, "composite_score").iterrows()):
                with st.columns(3)[i % 3]:
                    stock_card(row)

    # Safe
    with tabs[3]:
        picks = fdf.query("risk_level == 'Low' & market_cap > 1e11")
        if picks.empty:
            show_alert("No low-risk large-cap picks.", "info")
        else:
            for i, (_, row) in enumerate(picks.nlargest(12, "composite_score").iterrows()):
                with st.columns(3)[i % 3]:
                    stock_card(row)

# -----------------------------------------------------------------------------#
# 10. Tab: Sectors
# -----------------------------------------------------------------------------#
elif ss.tab == "📈 Sectors":
    section_header("Sector Analysis")
    if "sector" in fdf:
        stats = (
            fdf.groupby("sector")
            .agg(
                Stocks=("ticker", "count"),
                AvgReturn30D=("ret_30d", "mean"),
                AvgScore=("composite_score", "mean"),
                BuySignals=("decision", lambda x: x.eq("BUY").sum()),
            )
            .round(2)
            .sort_values("AvgScore", ascending=False)
        )
        display_dataframe(stats.reset_index(), height=420)

    if not ss.sector_df.empty:
        st.markdown("---")
        st.plotly_chart(sector_heatmap(ss.sector_df), use_container_width=True)

    st.markdown("---")
    sel_sector = st.selectbox(
        "Sector details",
        sorted(fdf["sector"].dropna().unique()) if "sector" in fdf else [],
    )
    if sel_sector:
        sec_df = fdf.query("sector == @sel_sector").nlargest(20, "composite_score")
        display_dataframe(
            sec_df[
                ["ticker", "company_name", "decision", "composite_score", "price", "ret_30d", "pe", "risk_level"]
            ],
            f"Top in {sel_sector}",
        )

# -----------------------------------------------------------------------------#
# 11. Tab: Analysis
# -----------------------------------------------------------------------------#
else:  # 📋 Analysis
    section_header("Score Distributions")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            gauge_chart(fdf["composite_score"].mean(), "Composite (avg)"), use_container_width=True
        )
    with c2:
        st.plotly_chart(
            gauge_chart(fdf["momentum_score"].mean(), "Momentum (avg)"), use_container_width=True
        )

    st.markdown("---")
    section_header("Market Regime")

    breadth = (fdf["ret_1d"] > 0).mean() * 100 if len(fdf) else 0
    avg30 = fdf["ret_30d"].mean()
    if breadth > 70 and avg30 > 5:
        regime, desc = "🐂 Bull", "Strong up-trend"
    elif breadth < 30 and avg30 < -5:
        regime, desc = "🐻 Bear", "Broad weakness"
    elif 45 <= breadth <= 55 and abs(avg30) < 2:
        regime, desc = "↔ Sideways", "Range-bound"
    else:
        regime, desc = "🔄 Transition", "Searching direction"

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"### {regime}")
        st.markdown(f"*{desc}*")
    with s2:
        metric_card("Breadth", f"{breadth:.0f}%", icon="📊")
    with s3:
        metric_card("Avg 30-day Return", f"{avg30:.1f}%", icon="📈")

# -----------------------------------------------------------------------------#
# 12. Footer
# -----------------------------------------------------------------------------#
st.markdown(
    """
    <hr>
    <div style="text-align:center;color:#8b92a0;font-size:13px;padding:16px 0">
      🔱 <b>M.A.N.T.R.A.</b> – Market Analysis Neural Trading Research Assistant<br>
      Signals for educational use only. Do your own research.
    </div>
    """,
    unsafe_allow_html=True,
)
