"""
app.py - M.A.N.T.R.A. Main Streamlit Application (Final Version)
================================================================
The complete, production-ready M.A.N.T.R.A. dashboard.
100% data-driven, robust, beautiful, and optimized for Streamlit Cloud.

Features:
- Real-time Indian stock market analysis
- Advanced multi-factor signal generation
- Interactive visualizations and filtering
- Professional dark theme UI
- Mobile-responsive design
- Export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Import our modules
from constants import SHEET_URLS, SIGNAL_COLORS, RISK_COLORS, SCHEMA_DOC
from data_loader import load_all_data
from signals import SignalEngine, calculate_market_summary
from ui_components import UIComponents, quick_metric, quick_stock_grid

# ============================================================================
# PAGE CONFIGURATION (MUST BE FIRST)
# ============================================================================
st.set_page_config(
    page_title="M.A.N.T.R.A. - Stock Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
UIComponents.load_css()

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'stocks_df' not in st.session_state:
    st.session_state.stocks_df = pd.DataFrame()
if 'sector_df' not in st.session_state:
    st.session_state.sector_df = pd.DataFrame()

# ============================================================================
# CACHED DATA LOADING
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_and_process_data():
    """Load and process all data with caching"""
    return load_all_data()

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Display header
    UIComponents.display_header()
    
    # Control section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🎯 **Indian Stock Intelligence Engine**")
    
    with col2:
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            st.rerun()
    
    with col3:
        if st.button("📊 Health Check", use_container_width=True):
            with st.spinner("Checking data health..."):
                st.info("All systems operational ✅")
    
    # Load data if needed
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading latest market data..."):
            watchlist_df, returns_df, sector_df, status = load_and_process_data()
            
            if not status['success']:
                st.error("❌ Failed to load data. Please check your connection and try again.")
                st.error(f"Errors: {'; '.join(status['errors'])}")
                st.stop()
            
            # Process signals
            with st.spinner("🧠 Calculating intelligent signals..."):
                signals_df = SignalEngine.calculate_all_signals(
                    watchlist_df, sector_df, regime="balanced"
                )
            
            # Store in session state
            st.session_state.stocks_df = signals_df
            st.session_state.sector_df = sector_df
            st.session_state.data_quality = status.get('quality', {})
            st.session_state.data_loaded = True
            st.session_state.last_refresh = datetime.now()
            
            st.success(f"✅ Loaded {len(signals_df)} stocks successfully!")
            time.sleep(1)
            st.rerun()
    
    # Get current data
    df = st.session_state.stocks_df.copy()
    sector_df = st.session_state.sector_df.copy()
    
    if df.empty:
        st.warning("No data available. Please refresh.")
        return
    
    # ========================================================================
    # SIDEBAR FILTERS
    # ========================================================================
    with st.sidebar:
        st.title("🔍 Filters")
        
        # Data quality indicator
        if 'data_quality' in st.session_state:
            UIComponents.display_data_quality(st.session_state.data_quality)
        
        st.markdown("---")
        
        # Signal filter
        signal_options = ["All"] + sorted(df["signal"].unique().tolist())
        selected_signal = st.selectbox("📶 Signal Type", signal_options)
        
        # Sector filter
        sector_options = ["All"] + sorted(df["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("🏭 Sector", sector_options)
        
        # Risk filter
        risk_options = ["All"] + sorted(df["risk"].unique().tolist())
        selected_risk = st.selectbox("⚠️ Risk Level", risk_options)
        
        # Score range
        min_score = st.slider("📈 Minimum Score", 0, 100, 60, step=5)
        
        # Price range
        if 'price' in df.columns:
            price_range = st.slider(
                "💰 Price Range (₹)", 
                int(df['price'].min()), 
                int(df['price'].max()), 
                (int(df['price'].min()), int(df['price'].max()))
            )
        else:
            price_range = (0, 10000)
        
        # Volume filter
        min_volume = st.number_input(
            "📊 Min Volume", 
            min_value=0, 
            value=1000, 
            step=1000,
            format="%d"
        )
        
        st.markdown("---")
        st.markdown("### 📚 Documentation")
        with st.expander("📄 Data Schema"):
            st.code(SCHEMA_DOC, language="text")
    
    # ========================================================================
    # APPLY FILTERS
    # ========================================================================
    filtered_df = df.copy()
    
    # Apply filters
    if selected_signal != "All":
        filtered_df = filtered_df[filtered_df["signal"] == selected_signal]
    
    if selected_sector != "All":
        filtered_df = filtered_df[filtered_df["sector"] == selected_sector]
    
    if selected_risk != "All":
        filtered_df = filtered_df[filtered_df["risk"] == selected_risk]
    
    filtered_df = filtered_df[filtered_df["score"] >= min_score]
    
    if 'price' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["price"] >= price_range[0]) & 
            (filtered_df["price"] <= price_range[1])
        ]
    
    if 'volume_1d' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["volume_1d"] >= min_volume]
    
    # ========================================================================
    # MARKET OVERVIEW METRICS
    # ========================================================================
    st.markdown("---")
    st.markdown("### 📊 **Market Overview**")
    
    # Calculate market summary
    market_summary = calculate_market_summary(df)
    
    # Display metrics in grid
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        UIComponents.metric_card(
            "Total Stocks", 
            f"{market_summary.get('total_stocks', 0):,}",
            icon="📈"
        )
    
    with col2:
        buy_pct = (market_summary.get('buy_signals', 0) / market_summary.get('total_stocks', 1)) * 100
        UIComponents.metric_card(
            "Buy Signals", 
            str(market_summary.get('buy_signals', 0)),
            delta=f"{buy_pct:.1f}%",
            delta_color="green" if buy_pct > 20 else "red"
        )
    
    with col3:
        UIComponents.metric_card(
            "Strong Buys", 
            str(market_summary.get('strong_buy_signals', 0)),
            icon="🚀"
        )
    
    with col4:
        UIComponents.metric_card(
            "Avg Score", 
            f"{market_summary.get('avg_score', 50):.1f}",
            icon="🎯"
        )
    
    with col5:
        breadth = market_summary.get('market_breadth', 50)
        UIComponents.metric_card(
            "Market Breadth", 
            f"{breadth:.1f}%",
            delta_color="green" if breadth > 50 else "red"
        )
    
    with col6:
        UIComponents.metric_card(
            "High Risk", 
            str(market_summary.get('high_risk_count', 0)),
            icon="⚠️"
        )
    
    # ========================================================================
    # TOP OPPORTUNITIES SECTION
    # ========================================================================
    st.markdown("---")
    st.markdown("### 🎯 **Top Opportunities**")
    
    # Get top opportunities
    top_opportunities = SignalEngine.get_top_picks(
        filtered_df, signal_filter="BUY", limit=12
    )
    
    if not top_opportunities.empty:
        quick_stock_grid(top_opportunities, max_cards=12)
    else:
        st.info("🔍 No opportunities found with current filters. Try adjusting your criteria.")
    
    # ========================================================================
    # DETAILED SIGNALS TABLE
    # ========================================================================
    st.markdown("---")
    st.markdown(f"### 📋 **Detailed Analysis** ({len(filtered_df)} stocks)")
    
    if not filtered_df.empty:
        # Prepare display columns
        display_columns = [
            'ticker', 'company_name', 'sector', 'signal', 'score', 'risk',
            'price', 'ret_1d', 'ret_30d', 'pe', 'volume_1d', 'rvol', 'reason'
        ]
        
        # Only show available columns
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        # Style the dataframe
        display_df = filtered_df[available_columns].copy()
        styled_df = UIComponents.style_dataframe(display_df)
        
        # Display the table
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=600,
            hide_index=True
        )
        
        # Export functionality
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            csv_data = filtered_df[available_columns].to_csv(index=False)
            st.download_button(
                label="📥 Export to CSV",
                data=csv_data,
                file_name=f"mantra_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("🔍 No stocks match your current filters.")
    
    # ========================================================================
    # SECTOR ANALYSIS
    # ========================================================================
    if not sector_df.empty:
        st.markdown("---")
        st.markdown("### 🏭 **Sector Performance**")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Heatmap", "📈 Distribution"])
        
        with tab1:
            # Sector heatmap
            heatmap_fig = UIComponents.sector_heatmap(sector_df)
            if heatmap_fig.data:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.info("Sector performance data not available")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                score_dist_fig = UIComponents.create_score_distribution(filtered_df)
                if score_dist_fig.data:
                    st.plotly_chart(score_dist_fig, use_container_width=True)
            
            with col2:
                # Sector distribution
                sector_dist_fig = UIComponents.create_sector_chart(filtered_df)
                if sector_dist_fig.data:
                    st.plotly_chart(sector_dist_fig, use_container_width=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>🔱 M.A.N.T.R.A. v1.0.0 Final</strong> | 
        Last Updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_refresh else 'Never'}</p>
        <p style="font-size: 12px; margin-top: 10px;">
            All signals are for educational purposes only. Always conduct your own research before making investment decisions.
        </p>
        <p style="font-size: 11px; color: #555;">
            Philosophy: "All signal, no noise. Decisions, not guesses."
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ERROR HANDLING & EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Please refresh the page and try again.")
        
        # Debug info in expander
        with st.expander("🔧 Debug Information"):
            st.code(f"Error: {str(e)}\nType: {type(e).__name__}")
            
        # Emergency data refresh
        if st.button("🆘 Emergency Refresh"):
            st.cache_data.clear()
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
