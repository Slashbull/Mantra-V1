"""
M.A.N.T.R.A. - Market Analysis Neural Trading Research Assistant
================================================================
FINAL PRODUCTION VERSION 1.0.0
All signal, no noise. Every element is intentional.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Import our modules
from config import *
from data import DataHandler
from signals import SignalEngine
from ui import UI

# ============================================================================
# PAGE CONFIGURATION (MUST BE FIRST)
# ============================================================================

st.set_page_config(
    page_title="M.A.N.T.R.A. - Stock Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start collapsed for cleaner look
)

# Initialize UI styling
UI.load_css()

# ============================================================================
# SESSION STATE
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None

# ============================================================================
# HEADER
# ============================================================================

col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    st.markdown("# 🔱 M.A.N.T.R.A.")
    st.caption("Market Analysis Neural Trading Research Assistant")

with col2:
    # Data quality indicator
    if st.session_state.data_loaded and 'stocks_df' in st.session_state:
        quality = DataHandler.assess_quality(st.session_state.stocks_df)
        UI.quality_indicator(quality)

with col3:
    if st.button("🔄 Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.rerun()

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_all_data():
    """Load and process all data with caching"""
    return DataHandler.load_data()

# Load data if needed
if not st.session_state.data_loaded:
    with st.spinner("Loading market data..."):
        stocks_df, sector_df, status = load_all_data()
        
        if status['success']:
            # Calculate signals
            stocks_df = SignalEngine.calculate_signals(stocks_df, sector_df)
            
            # Store in session
            st.session_state.stocks_df = stocks_df
            st.session_state.sector_df = sector_df
            st.session_state.data_loaded = True
            st.session_state.last_refresh = datetime.now()
        else:
            st.error("Failed to load data. Please check your internet connection and try again.")
            st.stop()

# Get data
df = st.session_state.stocks_df.copy()

# ============================================================================
# QUICK FILTERS (Minimal, Essential Only)
# ============================================================================

with st.expander("🎯 Filters", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        signal_filter = st.multiselect(
            "Signals",
            options=['STRONG_BUY', 'BUY', 'WATCH'],
            default=['STRONG_BUY', 'BUY']
        )
    
    with col2:
        min_score = st.slider("Min Score", 0, 100, 70, step=10)
    
    with col3:
        risk_filter = st.multiselect(
            "Risk",
            options=['Low', 'Medium', 'High'],
            default=['Low', 'Medium']
        )
    
    with col4:
        min_volume = st.number_input(
            "Min Volume",
            min_value=0,
            value=100000,
            step=50000,
            format="%d"
        )

# Apply filters
if signal_filter:
    df = df[df['signal'].isin(signal_filter)]

df = df[df['score'] >= min_score]

if risk_filter:
    df = df[df['risk'].isin(risk_filter)]

if 'volume' in df.columns:
    df = df[df['volume'] >= min_volume]

# ============================================================================
# MARKET OVERVIEW (Single Row of Key Metrics)
# ============================================================================

st.markdown("---")

# Calculate market metrics
total_stocks = len(df)
buy_signals = len(df[df['signal'].isin(['BUY', 'STRONG_BUY'])])
avg_score = df['score'].mean() if len(df) > 0 else 0
market_breadth = (df['ret_1d'] > 0).sum() / len(df) * 100 if len(df) > 0 and 'ret_1d' in df.columns else 50

# Display metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    UI.metric_card("Stocks", f"{total_stocks:,}", icon="📊")

with col2:
    UI.metric_card("Buy Signals", buy_signals, 
                   delta=f"{buy_signals/total_stocks*100:.0f}%" if total_stocks > 0 else "0%",
                   delta_color="green")

with col3:
    UI.metric_card("Avg Score", f"{avg_score:.0f}", icon="📈")

with col4:
    UI.metric_card("Breadth", f"{market_breadth:.0f}%", 
                   delta_color="green" if market_breadth > 50 else "red")

with col5:
    strong_momentum = len(df[df['momentum'] > 80]) if 'momentum' in df.columns else 0
    UI.metric_card("Momentum", strong_momentum, icon="🚀")

with col6:
    high_volume = len(df[df['rvol'] > 2]) if 'rvol' in df.columns else 0
    UI.metric_card("Volume Spikes", high_volume, icon="📊")

# ============================================================================
# TOP OPPORTUNITIES (The Core Value)
# ============================================================================

st.markdown("---")
st.subheader("🎯 Top Opportunities")

# Get top stocks
top_stocks = df[df['signal'].isin(['STRONG_BUY', 'BUY'])].nlargest(12, 'score')

if not top_stocks.empty:
    # Display in a clean grid
    cols = st.columns(3)
    for idx, (_, stock) in enumerate(top_stocks.iterrows()):
        with cols[idx % 3]:
            UI.stock_card(stock)
else:
    st.info("No buy opportunities found. Try adjusting filters.")

# ============================================================================
# SIGNAL TABLE (Actionable Data Only)
# ============================================================================

st.markdown("---")
st.subheader(f"📊 Signals ({len(df)} stocks)")

# Prepare display dataframe
display_cols = [
    'ticker', 'name', 'signal', 'score', 'price', 
    'ret_1d', 'ret_30d', 'pe', 'volume', 'rvol', 
    'momentum', 'value', 'risk', 'sector'
]

# Only show columns that exist
available_cols = [col for col in display_cols if col in df.columns]

# Sort by score
df_display = df.nlargest(min(100, len(df)), 'score')[available_cols]

# Display with styling
st.dataframe(
    UI.style_dataframe(df_display),
    height=600,
    use_container_width=True,
    hide_index=True
)

# ============================================================================
# SECTOR PERFORMANCE (Simple Heatmap)
# ============================================================================

if not st.session_state.sector_df.empty:
    st.markdown("---")
    st.subheader("🏭 Sector Performance")
    
    fig = UI.sector_heatmap(st.session_state.sector_df)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DOWNLOAD
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    csv = df[available_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Signals (CSV)",
        data=csv,
        file_name=f"mantra_signals_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🔱 M.A.N.T.R.A. v1.0.0 Final | Data from Google Sheets</p>
        <p style="font-size: 12px;">All signals for educational purposes only. Always do your own research.</p>
    </div>
    """,
    unsafe_allow_html=True
)
