#!/usr/bin/env python3
# =============================================================================
# ui_components.py – M.A.N.T.R.A. Re-usable UI Toolkit  (FINAL, BUG-FREE)
# =============================================================================
"""
All presentation helpers (CSS, metric cards, stock cards, charts, alerts,
dataframes) for **streamlit_dashboard.py**.
Only Streamlit + Plotly are required; no extra packages.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from constants import CHART_COLORS, SIGNAL_COLORS

# -----------------------------------------------------------------------------#
#  1. Global CSS  (call exactly once at app start)
# -----------------------------------------------------------------------------#
def load_custom_css() -> None:
    """Inject dark-theme tweaks + component styles into the app."""
    if st.session_state.get("_css_loaded"):
        return
    st.session_state["_css_loaded"] = True

    st.markdown(
        """
        <style>
        /* Hide Streamlit chrome */
        #MainMenu, footer, header {visibility: hidden;}

        /* Background */
        .stApp {background-color: #0e1117;}

        /* Metric card */
        .metric-card      {background:linear-gradient(135deg,#1e2329 0%,#2d3139 100%);
                           border:1px solid #2d3139;border-radius:12px;
                           padding:20px;text-align:center;transition:transform .15s}
        .metric-card:hover{transform:translateY(-2px);border-color:#00d26a}

        /* Signal badge */
        .signal-badge     {padding:6px 16px;border-radius:20px;font-weight:700;
                           font-size:14px;display:inline-block;margin:2px}
        .signal-buy   {background:#00d26a;color:#000}
        .signal-watch {background:#ffa500;color:#000}
        .signal-avoid {background:#ff4b4b;color:#fff}

        /* Stock card */
        .stock-card      {background:#1e2329;border:1px solid #2d3139;border-radius:10px;
                          padding:15px;margin:8px 0;transition:all .25s}
        .stock-card:hover{background:#252a31;border-color:#00d26a;transform:translateY(-1px)}

        /* Alerts */
        .alert-info    {background:linear-gradient(135deg,#33b5e5 0%,#4fc3f7 100%);color:#fff}
        .alert-warning {background:linear-gradient(135deg,#ffa500 0%,#ffb732 100%);color:#000}
        .alert-danger  {background:linear-gradient(135deg,#ff4b4b 0%,#ff6b6b 100%);color:#fff}
        .alert-box     {padding:12px 20px;border-radius:8px;margin:10px 0;font-weight:600}

        /* Section header */
        .section-header{border-left:4px solid #00d26a;padding-left:15px;margin:26px 0}

        /* Data-quality badge */
        .quality-badge {display:inline-flex;align-items:center;padding:4px 12px;
                        border-radius:16px;font-size:12px;font-weight:600}
        .quality-good  {background:#00d26a;color:#000}
        .quality-warning{background:#ffa500;color:#000}
        .quality-poor  {background:#ff4b4b;color:#fff}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------#
#  2. Metric card + signal badge
# -----------------------------------------------------------------------------#
def metric_card(label: str, value: Any, delta: Optional[float] = None, icon: str = "") -> None:
    """Pretty KPI tile – handles ints, floats and strings gracefully."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if abs(value) >= 1_000_000:
            val_str = f"{value/1_000_000:.1f} M"
        elif abs(value) >= 1_000:
            val_str = f"{value/1_000:.1f} K"
        else:
            val_str = f"{value:,.0f}"
    else:
        val_str = str(value)

    delta_html = ""
    if delta is not None and isinstance(delta, (int, float)) and not math.isnan(delta):
        color = SIGNAL_COLORS["BUY"] if delta > 0 else SIGNAL_COLORS["AVOID"]
        arrow = "↑" if delta > 0 else "↓"
        delta_html = f'<div style="color:{color};font-size:14px;">{arrow} {abs(delta):.1f}%</div>'

    icon_html = f'<div style="font-size:24px;margin-bottom:8px;">{icon}</div>' if icon else ""

    st.markdown(
        f"""
        <div class="metric-card">
            {icon_html}
            <div style="color:#8b92a0;font-size:14px;">{label}</div>
            <div style="color:#fff;font-size:28px;font-weight:700;margin:8px 0;">{val_str}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_badge(signal: str, score: Optional[float] = None) -> None:
    """Colour-coded badge for BUY / WATCH / AVOID etc."""
    cls = {"BUY": "buy", "WATCH": "watch", "AVOID": "avoid", "NEUTRAL": "watch"}.get(signal, "watch")
    txt = f"{signal}{f' ({score:.0f})' if score is not None else ''}"
    st.markdown(f'<span class="signal-badge signal-{cls}">{txt}</span>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------#
#  3. Stock card
# -----------------------------------------------------------------------------#
def stock_card(stock: pd.Series) -> None:
    """Compact overview tile for one stock row (expects cleaned df row)."""
    ticker = stock.get("ticker", "N/A")
    company = str(stock.get("company_name", ticker))[:30]
    price = stock.get("price", np.nan)
    change = stock.get("ret_1d", np.nan)
    signal = stock.get("decision", "NEUTRAL")
    score = stock.get("composite_score", 50)
    volume = stock.get("volume_1d", np.nan)
    pe = stock.get("pe", np.nan)
    risk = stock.get("risk_level", "N/A")

    change_col = "#00d26a" if change >= 0 else "#ff4b4b"
    sign = "↑" if change >= 0 else "↓"

    st.markdown(
        f"""
        <div class="stock-card">
        <div style="display:flex;justify-content:space-between">
            <div>
                <h4 style="margin:0;color:#fff">{ticker}</h4>
                <p style="margin:0;color:#8b92a0;font-size:12px">{company}</p>
            </div>
            <span class="signal-badge signal-{signal.lower()}">{signal} ({score:.0f})</span>
        </div>
        <div style="margin-top:14px;font-size:24px;font-weight:700;color:#fff">
            ₹{price:,.2f}
            <span style="font-size:16px;color:{change_col};margin-left:10px">
                {sign} {abs(change):.1f}%
            </span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px">
            <div><p style="margin:0;color:#8b92a0;font-size:11px">Volume</p>
                 <p style="margin:0;color:#fff;font-size:13px">{volume/1_000:,.0f} K</p></div>
            <div><p style="margin:0;color:#8b92a0;font-size:11px">P/E</p>
                 <p style="margin:0;color:#fff;font-size:13px">{pe:.1f}</p></div>
            <div><p style="margin:0;color:#8b92a0;font-size:11px">Risk</p>
                 <p style="margin:0;color:#fff;font-size:13px">{risk}</p></div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------#
#  4. Charts
# -----------------------------------------------------------------------------#
def sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
    """6-period %-return heatmap (sectors × periods)."""
    periods = ["1D", "7D", "30D", "3M", "6M", "1Y"]
    cols = ["sector_ret_1d", "sector_ret_7d", "sector_ret_30d", "sector_ret_3m", "sector_ret_6m", "sector_ret_1y"]

    z = []
    for _, row in sector_df.iterrows():
        z.append([row.get(c, 0) for c in cols])

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=periods,
            y=sector_df["sector"],
            colorscale="RdYlGn",
            zmid=0,
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
            text=[[f"{v:.1f}%" for v in r] for r in z],
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig.update_layout(
        title="Sector Performance Heatmap",
        xaxis_title="Period",
        yaxis_title="Sector",
        height=max(420, 25 * len(sector_df)),
        template="plotly_dark",
        margin=dict(l=150, r=20, t=50, b=40),
    )
    return fig


def gauge_chart(value: float, title: str) -> go.Figure:
    """Single gauge (0-100)."""
    if value >= 80:
        bar = SIGNAL_COLORS["BUY"]
    elif value >= 65:
        bar = SIGNAL_COLORS["WATCH"]
    else:
        bar = SIGNAL_COLORS["AVOID"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar},
                "steps": [
                    {"range": [0, 35], "color": "rgba(255,75,75,.25)"},
                    {"range": [35, 65], "color": "rgba(255,165,0,.25)"},
                    {"range": [65, 100], "color": "rgba(0,210,106,.25)"},
                ],
            },
        )
    )
    fig.update_layout(height=210, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    return fig


def dist_chart(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Histogram with mean line."""
    fig = go.Figure(go.Histogram(x=df[column], nbinsx=30, marker_color=SIGNAL_COLORS["BUY"], opacity=0.7))
    mean_val = df[column].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="#fff", annotation_text=f"Mean {mean_val:.1f}")
    fig.update_layout(title=title, xaxis_title=column.replace("_", " ").title(), yaxis_title="Count", template="plotly_dark", height=300)
    return fig


# -----------------------------------------------------------------------------#
#  5. Alerts & badges
# -----------------------------------------------------------------------------#
def show_alert(msg: str, typ: str = "info") -> None:
    cls_map = {"info": "info", "warning": "warning", "danger": "danger", "success": "info"}
    icon_map = {"info": "ℹ️", "warning": "⚠️", "danger": "🚨", "success": "✅"}
    cls = cls_map.get(typ, "info")
    icon = icon_map.get(typ, "ℹ️")
    st.markdown(f'<div class="alert-box alert-{cls}">{icon}&nbsp; {msg}</div>', unsafe_allow_html=True)


def data_quality_badge(score: float) -> None:
    if score >= 80:
        cls, txt = "quality-good", "Good"
    elif score >= 60:
        cls, txt = "quality-warning", "Fair"
    else:
        cls, txt = "quality-poor", "Poor"
    st.markdown(f'<span class="quality-badge {cls}">Data Quality: {txt} ({score:.0f}%)</span>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------#
#  6. Dataframe display
# -----------------------------------------------------------------------------#
def display_dataframe(df: pd.DataFrame, title: str = "", height: int = 420) -> None:
    """HTML-styled DataFrame via pandas Styler (Streamlit renders it with st.write)."""
    if title:
        st.markdown(f"### {title}")

    fmt: dict[str, str] = {}
    for c in df.columns:
        if "price" in c.lower():
            fmt[c] = "₹{0:,.2f}"
        elif any(k in c for k in ("ret_", "_pct", "score")):
            fmt[c] = "{0:+.1f}%"
        elif "volume" in c.lower():
            fmt[c] = "{0:,.0f}"

    styler = df.style.format(fmt)
    if "decision" in df:

        def style_dec(val: str) -> str:
            col = SIGNAL_COLORS.get(val.upper(), "#444")
            txt = "#000" if val.upper() == "BUY" else "#fff"
            return f"background:{col};color:{txt}"

        styler = styler.applymap(style_dec, subset=["decision"])

    st.write(styler.to_html(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------#
#  7. Layout helpers
# -----------------------------------------------------------------------------#
def section_header(title: str, subtitle: str = "") -> None:
    sub = f'<p style="margin:5px 0 0;color:#8b92a0">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="section-header"><h2 style="margin:0;color:#fff">{title}</h2>{sub}</div>', unsafe_allow_html=True)


def dashboard_header() -> None:
    col1, _, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("# 🔱 M.A.N.T.R.A.")
        st.markdown("*Market Analysis Neural Trading Research Assistant*")
    with col3:
        st.markdown(
            f"""
            <div style="text-align:right;padding-top:18px">
                <p style="margin:0;color:#8b92a0">Last Update</p>
                <p style="margin:0;color:#fff;font-weight:700">{datetime.now().strftime('%H:%M:%S')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
