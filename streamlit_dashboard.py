"""
streamlit_dashboard.py - M.A.N.T.R.A. Main Dashboard
==================================================
FINAL PRODUCTION VERSION - Fast, Beautiful, Bug-free
Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION (MUST BE FIRST)
# ============================================================================

st.set_page_config(
    page_title="M.A.N.T.R.A. - Stock Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORTS (AFTER PAGE CONFIG)
# ============================================================================

from constants import *
from data_loader import DataLoader
from signal_engine import SignalEngine
from ui_components import *

# Load custom CSS
load_custom_css()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "overview"
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1

# ============================================================================
# DATA LOADING WITH PROGRESS
# ============================================================================

@st.cache_data(ttl=CACHE_DURATION_MINUTES*60, show_spinner=False)
def load_and_process_data():
    """Load and process all data with proper error handling"""
    try:
        # Load data
        stocks_df, sector_df, health = DataLoader.load_all_data()
        
        if health['status'] != 'success':
            return None, None, health
        
        # Calculate signals
        if not stocks_df.empty:
            stocks_df = SignalEngine.calculate_all_signals(stocks_df, sector_df)
        
        return stocks_df, sector_df, health
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return None, None, {'status': 'error', 'message': str(e)}

# ============================================================================
# HEADER
# ============================================================================

dashboard_header()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Refresh button with timer
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            st.rerun()
    
    with col2:
        if st.session_state.last_refresh:
            mins_ago = (datetime.now() - st.session_state.last_refresh).seconds // 60
            st.caption(f"Updated {mins_ago}m ago")
    
    st.markdown("---")
    
    # FILTERS
    st.markdown("### 🔍 Filters")
    
    # Decision filter
    decision_filter = st.multiselect(
        "Signal Type",
        options=['BUY', 'STRONG_BUY', 'WATCH', 'NEUTRAL', 'AVOID'],
        default=['BUY', 'STRONG_BUY', 'WATCH']
    )
    
    # Score filter
    score_range = st.slider(
        "Composite Score",
        min_value=0,
        max_value=100,
        value=(60, 100),
        step=5
    )
    
    # Risk filter
    risk_filter = st.multiselect(
        "Risk Level",
        options=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        default=['Very Low', 'Low', 'Medium']
    )
    
    # Volume filter
    min_volume = st.number_input(
        "Min Daily Volume",
        min_value=0,
        value=50000,
        step=10000,
        format="%d"
    )
    
    # Sector filter placeholder
    sector_filter = []
    
    # Price range
    st.markdown("### 💰 Price Range")
    price_range = st.slider(
        "Price (₹)",
        min_value=0,
        max_value=10000,
        value=(0, 10000),
        step=100
    )

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Load data if not loaded
if not st.session_state.data_loaded:
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load data with progress updates
    status_text.text("🔄 Connecting to Google Sheets...")
    progress_bar.progress(20)
    
    data = load_and_process_data()
    
    if data[0] is not None:
        status_text.text("📊 Processing signals...")
        progress_bar.progress(80)
        
        st.session_state.stocks_df = data[0]
        st.session_state.sector_df = data[1]
        st.session_state.health = data[2]
        st.session_state.data_loaded = True
        st.session_state.last_refresh = datetime.now()
        
        # Get unique sectors for filter
        if 'sector' in st.session_state.stocks_df.columns:
            unique_sectors = sorted(st.session_state.stocks_df['sector'].dropna().unique())
            sector_filter = st.sidebar.multiselect(
                "Sectors",
                options=unique_sectors,
                default=[]
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Data loaded successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
    else:
        progress_bar.empty()
        status_text.empty()
        st.error("❌ Failed to load data. Please check your connection and try again.")
        st.stop()

# Apply filters
df = st.session_state.stocks_df.copy()

# Apply all filters
if decision_filter:
    df = df[df['decision'].isin(decision_filter)]

df = df[
    (df['composite_score'] >= score_range[0]) & 
    (df['composite_score'] <= score_range[1])
]

if risk_filter:
    df = df[df['risk_level'].isin(risk_filter)]

if 'volume_1d' in df.columns:
    df = df[df['volume_1d'] >= min_volume]

if sector_filter and 'sector' in df.columns:
    df = df[df['sector'].isin(sector_filter)]

if 'price' in df.columns:
    df = df[
        (df['price'] >= price_range[0]) & 
        (df['price'] <= price_range[1])
    ]

# Show data quality
if 'data_quality' in st.session_state.health:
    data_quality_badge(st.session_state.health['data_quality'])

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🎯 Signals", 
    "🔥 Top Picks", 
    "📈 Sectors",
    "📋 Analysis"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    # Market metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_stocks = len(st.session_state.stocks_df)
        filtered_stocks = len(df)
        metric_card("Total Stocks", f"{filtered_stocks}/{total_stocks}", icon="📊")
    
    with col2:
        buy_signals = len(df[df['decision'].isin(['BUY', 'STRONG_BUY'])])
        buy_pct = (buy_signals / len(df) * 100) if len(df) > 0 else 0
        metric_card("Buy Signals", buy_signals, delta=buy_pct, icon="🟢")
    
    with col3:
        avg_score = df['composite_score'].mean() if len(df) > 0 else 0
        metric_card("Avg Score", f"{avg_score:.1f}", icon="📈")
    
    with col4:
        if 'momentum_score' in df.columns:
            high_momentum = len(df[df['momentum_score'] > 80])
            metric_card("High Momentum", high_momentum, icon="🚀")
        else:
            metric_card("Filtered", len(df), icon="🔍")
    
    with col5:
        if 'ret_1d' in df.columns:
            gainers = (df['ret_1d'] > 0).sum()
            breadth = (gainers / len(df) * 100) if len(df) > 0 else 50
            metric_card("Breadth", f"{breadth:.0f}%", icon="📊")
        else:
            metric_card("Sectors", df['sector'].nunique() if 'sector' in df.columns else 0, icon="🏭")
    
    with col6:
        avg_volume = df['volume_1d'].mean() / 1e6 if 'volume_1d' in df.columns else 0
        metric_card("Avg Volume", f"{avg_volume:.1f}M", icon="📊")
    
    # Top opportunities
    st.markdown("---")
    section_header("🎯 Top Opportunities", "Highest conviction trades right now")
    
    top_buys = df[df['decision'].isin(['BUY', 'STRONG_BUY'])].nlargest(
        6, 'opportunity_score' if 'opportunity_score' in df.columns else 'composite_score'
    )
    
    if not top_buys.empty:
        cols = st.columns(3)
        for idx, (_, stock) in enumerate(top_buys.iterrows()):
            with cols[idx % 3]:
                stock_card(stock)
    else:
        show_alert("No buy opportunities found with current filters. Try adjusting filters.", "info")
    
    # Quick stats
    st.markdown("---")
    section_header("📊 Market Snapshot", "Key statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Return distribution
        if 'ret_30d' in df.columns and len(df) > 0:
            fig = create_distribution_chart(df, 'ret_30d', '30-Day Return Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Score distribution
        if len(df) > 0:
            fig = create_distribution_chart(df, 'composite_score', 'Composite Score Distribution')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: SIGNALS
# ============================================================================

with tab2:
    section_header("Trading Signals", f"Found {len(df)} stocks matching your criteria")
    
    # Signal summary
    if not df.empty:
        signal_counts = df['decision'].value_counts()
        
        cols = st.columns(len(signal_counts))
        for idx, (signal, count) in enumerate(signal_counts.items()):
            with cols[idx]:
                color = SIGNAL_COLORS.get(signal, SIGNAL_COLORS['NEUTRAL'])
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 10px; background: {color}20; border-radius: 10px;">
                        <h3 style="color: {color}; margin: 0;">{count}</h3>
                        <p style="margin: 0;">{signal}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    st.markdown("---")
    
    # Sorting options
    col1, col2, col3 = st.columns([2, 2, 8])
    
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=['opportunity_score', 'composite_score', 'ret_30d', 'volume_1d', 'pe'],
            index=0
        )
    
    with col2:
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    
    # Sort data
    if sort_by in df.columns:
        df_sorted = df.sort_values(sort_by, ascending=(sort_order == "Ascending"))
    else:
        df_sorted = df.sort_values('composite_score', ascending=False)
    
    # Pagination
    rows_per_page = 50
    total_rows = len(df_sorted)
    total_pages = max(1, (total_rows - 1) // rows_per_page + 1)
    
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if total_pages > 1:
            page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.page_number,
                key="page_selector"
            )
            st.session_state.page_number = page
        else:
            page = 1
    
    # Display data
    start_idx = (page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    if not df_sorted.empty:
        # Select display columns
        display_columns = [
            'ticker', 'company_name', 'sector', 'decision', 'composite_score',
            'price', 'ret_1d', 'ret_30d', 'pe', 'volume_1d', 'rvol',
            'momentum_score', 'value_score', 'risk_level', 'reasoning'
        ]
        
        # Filter to available columns
        available_columns = [col for col in display_columns if col in df_sorted.columns]
        
        # Display table
        display_dataframe(
            df_sorted.iloc[start_idx:end_idx][available_columns],
            f"Showing {start_idx + 1}-{end_idx} of {total_rows} stocks",
            height=600
        )
        
        # Download button
        csv = df_sorted[available_columns].to_csv(index=False)
        st.download_button(
            label="📥 Download All Filtered Signals",
            data=csv,
            file_name=f"mantra_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        show_alert("No stocks found matching your criteria. Try adjusting the filters.", "info")

# ============================================================================
# TAB 3: TOP PICKS
# ============================================================================

with tab3:
    section_header("Top Stock Picks", "Best opportunities by category")
    
    # Category tabs
    pick_tab1, pick_tab2, pick_tab3, pick_tab4 = st.tabs([
        "🚀 Momentum", "💎 Value", "📈 Growth", "🛡️ Safe"
    ])
    
    with pick_tab1:
        # Momentum picks
        if 'momentum_score' in df.columns:
            momentum_picks = df[
                (df['momentum_score'] > 75) & 
                (df['decision'].isin(['BUY', 'STRONG_BUY', 'WATCH']))
            ].nlargest(12, 'momentum_score')
            
            if not momentum_picks.empty:
                st.markdown("**High momentum stocks with strong price action**")
                cols = st.columns(3)
                for idx, (_, stock) in enumerate(momentum_picks.iterrows()):
                    with cols[idx % 3]:
                        stock_card(stock)
            else:
                show_alert("No high momentum stocks found", "info")
    
    with pick_tab2:
        # Value picks
        if 'value_score' in df.columns and 'pe' in df.columns:
            value_picks = df[
                (df['value_score'] > 70) & 
                (df['pe'] > 0) & 
                (df['pe'] < 25) &
                (df['decision'].isin(['BUY', 'STRONG_BUY', 'WATCH']))
            ].nlargest(12, 'value_score')
            
            if not value_picks.empty:
                st.markdown("**Undervalued stocks with strong fundamentals**")
                cols = st.columns(3)
                for idx, (_, stock) in enumerate(value_picks.iterrows()):
                    with cols[idx % 3]:
                        stock_card(stock)
            else:
                show_alert("No value picks found", "info")
    
    with pick_tab3:
        # Growth picks
        if 'eps_change_pct' in df.columns:
            growth_picks = df[
                (df['eps_change_pct'] > 20) &
                (df['decision'].isin(['BUY', 'STRONG_BUY', 'WATCH']))
            ].nlargest(12, 'composite_score')
            
            if not growth_picks.empty:
                st.markdown("**High growth companies with strong earnings**")
                cols = st.columns(3)
                for idx, (_, stock) in enumerate(growth_picks.iterrows()):
                    with cols[idx % 3]:
                        stock_card(stock)
            else:
                show_alert("No growth picks found", "info")
    
    with pick_tab4:
        # Safe picks (large cap, low risk)
        safe_picks = df[
            (df['risk_level'].isin(['Very Low', 'Low'])) &
            (df['decision'].isin(['BUY', 'STRONG_BUY', 'WATCH']))
        ]
        
        if 'market_cap' in safe_picks.columns:
            safe_picks = safe_picks[safe_picks['market_cap'] > 5e10]  # > 5000 Cr
        
        safe_picks = safe_picks.nlargest(12, 'composite_score')
        
        if not safe_picks.empty:
            st.markdown("**Low risk stocks for conservative investors**")
            cols = st.columns(3)
            for idx, (_, stock) in enumerate(safe_picks.iterrows()):
                with cols[idx % 3]:
                    stock_card(stock)
        else:
            show_alert("No safe picks found", "info")

# ============================================================================
# TAB 4: SECTORS
# ============================================================================

with tab4:
    section_header("Sector Analysis", "Performance and rotation insights")
    
    if not st.session_state.sector_df.empty:
        # Sector heatmap
        fig = sector_heatmap(st.session_state.sector_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Sector statistics
        if 'sector' in df.columns:
            sector_stats = df.groupby('sector').agg({
                'ticker': 'count',
                'composite_score': 'mean',
                'ret_30d': 'mean' if 'ret_30d' in df.columns else 'count',
                'decision': lambda x: (x.isin(['BUY', 'STRONG_BUY'])).sum()
            }).round(2)
            
            sector_stats.columns = ['Stocks', 'Avg Score', 'Avg 30D Return', 'Buy Signals']
            sector_stats = sector_stats.sort_values('Avg Score', ascending=False)
            
            # Display sector table
            display_dataframe(
                sector_stats.reset_index(),
                "Sector Summary",
                height=400
            )
            
            # Sector deep dive
            st.markdown("---")
            selected_sector = st.selectbox(
                "Select Sector for Detailed View",
                options=sorted(df['sector'].unique())
            )
            
            if selected_sector:
                sector_stocks = df[df['sector'] == selected_sector]
                
                # Sector metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    metric_card("Stocks", len(sector_stocks))
                with col2:
                    buy_count = len(sector_stocks[sector_stocks['decision'].isin(['BUY', 'STRONG_BUY'])])
                    metric_card("Buy Signals", buy_count)
                with col3:
                    avg_score = sector_stocks['composite_score'].mean()
                    metric_card("Avg Score", f"{avg_score:.1f}")
                with col4:
                    if 'ret_30d' in sector_stocks.columns:
                        avg_return = sector_stocks['ret_30d'].mean()
                        metric_card("Avg 30D Return", f"{avg_return:.1f}%")
                
                # Top stocks in sector
                st.markdown("---")
                st.subheader(f"Top Stocks in {selected_sector}")
                
                top_sector_stocks = sector_stocks.nlargest(20, 'composite_score')
                
                if not top_sector_stocks.empty:
                    display_columns = [
                        'ticker', 'company_name', 'decision', 'composite_score',
                        'price', 'ret_30d', 'pe', 'volume_1d', 'risk_level'
                    ]
                    available_columns = [col for col in display_columns if col in top_sector_stocks.columns]
                    
                    display_dataframe(
                        top_sector_stocks[available_columns],
                        "",
                        height=400
                    )
    else:
        show_alert("Sector data not available", "warning")

# ============================================================================
# TAB 5: ANALYSIS
# ============================================================================

with tab5:
    section_header("Market Analysis", "Deep insights into market metrics")
    
    # Factor performance
    st.subheader("📊 Factor Performance")
    
    factor_scores = {}
    for factor in ['momentum_score', 'value_score', 'technical_score', 'volume_score']:
        if factor in df.columns:
            factor_scores[factor.replace('_score', '').title()] = df[factor].mean()
    
    if factor_scores:
        cols = st.columns(len(factor_scores))
        for idx, (factor, score) in enumerate(factor_scores.items()):
            with cols[idx]:
                fig = create_gauge_chart(score, factor)
                st.plotly_chart(fig, use_container_width=True)
    
    # Market regime
    st.markdown("---")
    st.subheader("🌡️ Market Regime")
    
    if 'ret_1d' in df.columns and 'ret_30d' in df.columns:
        # Calculate market metrics
        breadth = (df['ret_1d'] > 0).sum() / len(df) * 100 if len(df) > 0 else 50
        avg_return_30d = df['ret_30d'].mean() if len(df) > 0 else 0
        
        # Determine regime
        if breadth > 65 and avg_return_30d > 5:
            regime = "🐂 Bull Market"
            regime_desc = "Strong uptrend with broad participation"
            regime_color = SIGNAL_COLORS['BUY']
        elif breadth < 35 and avg_return_30d < -5:
            regime = "🐻 Bear Market"
            regime_desc = "Downtrend with widespread weakness"
            regime_color = SIGNAL_COLORS['AVOID']
        elif 45 <= breadth <= 55 and abs(avg_return_30d) < 2:
            regime = "↔️ Sideways Market"
            regime_desc = "Range-bound with no clear direction"
            regime_color = SIGNAL_COLORS['NEUTRAL']
        else:
            regime = "🔄 Transitional"
            regime_desc = "Market searching for direction"
            regime_color = SIGNAL_COLORS['WATCH']
        
        # Display regime
        st.markdown(
            f"""
            <div style="background: {regime_color}20; padding: 20px; border-radius: 10px; border: 2px solid {regime_color};">
                <h2 style="color: {regime_color}; margin: 0;">{regime}</h2>
                <p style="margin: 10px 0 0 0;">{regime_desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Regime metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metric_card("Market Breadth", f"{breadth:.0f}%", icon="📊")
        with col2:
            metric_card("Avg 30D Return", f"{avg_return_30d:.1f}%", icon="📈")
        with col3:
            winners = len(df[df['ret_30d'] > 10]) if 'ret_30d' in df.columns else 0
            metric_card("Strong Winners", winners, icon="🏆")
        with col4:
            losers = len(df[df['ret_30d'] < -10]) if 'ret_30d' in df.columns else 0
            metric_card("Weak Stocks", losers, icon="📉")
    
    # Additional insights
    st.markdown("---")
    st.subheader("📈 Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PE distribution
        if 'pe' in df.columns:
            pe_stats = df[df['pe'] > 0]['pe'].describe()
            st.markdown("**P/E Ratio Statistics**")
            st.dataframe(pe_stats.round(2))
    
    with col2:
        # Volume analysis
        if 'rvol' in df.columns:
            high_volume = len(df[df['rvol'] > 2])
            low_volume = len(df[df['rvol'] < 0.5])
            
            st.markdown("**Volume Analysis**")
            st.markdown(f"- High Volume Stocks: **{high_volume}**")
            st.markdown(f"- Low Volume Stocks: **{low_volume}**")
            st.markdown(f"- Average RVol: **{df['rvol'].mean():.2f}x**")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #8b92a0; padding: 20px;">
        <p>🔱 M.A.N.T.R.A. - Market Analysis Neural Trading Research Assistant</p>
        <p style="font-size: 12px;">Data from Google Sheets | All signals for educational purposes only</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================================
# END OF DASHBOARD
# ============================================================================
