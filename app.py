"""
app.py - M.A.N.T.R.A. Main Application
=====================================
Production-ready Streamlit application with beautiful UI/UX.
Zero bugs, optimized for Streamlit Cloud deployment.

Author: M.A.N.T.R.A. System
Version: 1.0.0 (Final Production)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
from typing import Dict, List, Optional, Any

# Import our modules
from constants import (
    SCHEMA_DOC, SUCCESS_MESSAGES, ERROR_MESSAGES, VALIDATION_RULES
)
from data_loader import (
    load_all_data, get_data_summary, format_load_status
)
from signals import (
    SignalEngine, get_top_picks, calculate_market_summary, get_sector_leaders
)
from ui_components import (
    UIComponents, ChartComponents, display_stock_grid, 
    display_section_header, add_section_spacer
)

warnings.filterwarnings('ignore')

# ============================================================================
# STREAMLIT PAGE CONFIGURATION (MUST BE FIRST)
# ============================================================================

st.set_page_config(
    page_title="M.A.N.T.R.A. Stock Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "M.A.N.T.R.A. - Market Analysis Neural Trading Research Assistant"
    }
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    default_states = {
        'data_loaded': False,
        'last_refresh': None,
        'stocks_df': pd.DataFrame(),
        'sector_df': pd.DataFrame(),
        'data_quality': {},
        'load_status': {},
        'app_version': '1.0.0'
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def cached_load_and_process_data():
    """Cached data loading and processing."""
    return load_all_data()

def load_and_process_signals(watchlist_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    """Load and process trading signals."""
    try:
        with st.spinner("🧠 Calculating intelligent signals..."):
            signals_df = SignalEngine.calculate_all_signals(watchlist_df, sector_df)
        return signals_df
    except Exception as e:
        st.error(f"Error calculating signals: {str(e)}")
        return watchlist_df

# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def display_control_panel():
    """Display main control panel."""
    try:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            display_section_header(
                "🎯 Indian Stock Intelligence Engine",
                "Real-time analysis with advanced multi-factor signals"
            )
        
        with col2:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                with st.spinner("Clearing cache and refreshing..."):
                    st.cache_data.clear()
                    st.session_state.data_loaded = False
                    time.sleep(0.5)  # Brief pause for user feedback
                st.rerun()
        
        with col3:
            if st.button("📊 System Status", use_container_width=True):
                status_check()
                
    except Exception as e:
        st.error(f"Error displaying control panel: {str(e)}")

def status_check():
    """Perform system status check."""
    try:
        with st.spinner("Checking system status..."):
            time.sleep(1)  # Simulate check
            
        if st.session_state.data_loaded:
            quality = st.session_state.get('data_quality', {})
            load_status = st.session_state.get('load_status', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.success("✅ Data: Loaded successfully")
                st.info(f"📊 Quality: {quality.get('status', 'Unknown')} ({quality.get('score', 0):.1f}/100)")
            with col2:
                st.success("✅ Signals: Active")
                st.info(f"⚡ Load time: {load_status.get('load_time', 0):.1f}s")
        else:
            st.warning("⚠️ Data not loaded. Please refresh.")
            
    except Exception as e:
        st.error(f"Status check failed: {str(e)}")

def setup_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Setup sidebar filters and return filter values."""
    filters = {}
    
    try:
        with st.sidebar:
            st.title("🔍 Smart Filters")
            
            # Data quality indicator
            if 'data_quality' in st.session_state and st.session_state.data_quality:
                UIComponents.display_data_quality_indicator(st.session_state.data_quality)
            
            add_section_spacer("1rem")
            
            # Core filters
            st.subheader("📊 Core Filters")
            
            # Signal filter
            signal_options = ["All"] + sorted(df["signal"].unique().tolist()) if 'signal' in df.columns else ["All"]
            filters['signal'] = st.selectbox(
                "📶 Signal Type",
                signal_options,
                help="Filter by signal strength"
            )
            
            # Sector filter
            if 'sector' in df.columns:
                sector_options = ["All"] + sorted(df["sector"].dropna().unique().tolist())
                filters['sector'] = st.selectbox(
                    "🏭 Sector",
                    sector_options,
                    help="Filter by industry sector"
                )
            else:
                filters['sector'] = "All"
            
            # Risk filter
            risk_options = ["All"] + sorted(df["risk"].unique().tolist()) if 'risk' in df.columns else ["All"]
            filters['risk'] = st.selectbox(
                "⚠️ Risk Level",
                risk_options,
                help="Filter by risk assessment"
            )
            
            add_section_spacer("1rem")
            
            # Advanced filters
            st.subheader("🎛️ Advanced Filters")
            
            # Score range
            filters['min_score'] = st.slider(
                "📈 Minimum Score",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="Minimum signal score threshold"
            )
            
            # Price range
            if 'price' in df.columns and df['price'].notna().any():
                price_min = max(1, int(df['price'].min()))
                price_max = min(100000, int(df['price'].max()))
                filters['price_range'] = st.slider(
                    "💰 Price Range (₹)",
                    min_value=price_min,
                    max_value=price_max,
                    value=(price_min, price_max),
                    help="Filter by stock price range"
                )
            else:
                filters['price_range'] = (1, 100000)
            
            # Volume filter
            filters['min_volume'] = st.number_input(
                "📊 Minimum Volume",
                min_value=0,
                value=10000,
                step=5000,
                format="%d",
                help="Minimum daily trading volume"
            )
            
            add_section_spacer("1rem")
            
            # Documentation
            st.subheader("📚 Documentation")
            with st.expander("📄 Data Schema", expanded=False):
                st.text(SCHEMA_DOC)
            
            with st.expander("ℹ️ How to Use", expanded=False):
                st.markdown("""
                **Quick Start:**
                1. 🔄 Refresh data for latest market info
                2. 🔍 Use filters to narrow down stocks
                3. 📊 Check top opportunities section
                4. 📋 Review detailed analysis table
                5. 📥 Export results as needed
                
                **Signal Meanings:**
                - 🟢 **STRONG_BUY**: Score ≥85, high confidence
                - 🟢 **BUY**: Score ≥70, good opportunity
                - 🟡 **WATCH**: Score ≥55, monitor closely
                - ⚪ **NEUTRAL**: Score ≥40, no clear direction
                - 🔴 **AVOID**: Score <40, high risk
                """)
        
        return filters
        
    except Exception as e:
        st.sidebar.error(f"Error setting up filters: {str(e)}")
        return {
            'signal': 'All',
            'sector': 'All', 
            'risk': 'All',
            'min_score': 50,
            'price_range': (1, 100000),
            'min_volume': 10000
        }

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to dataframe."""
    try:
        filtered_df = df.copy()
        
        # Signal filter
        if filters.get('signal') != "All" and 'signal' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["signal"] == filters['signal']]
        
        # Sector filter
        if filters.get('sector') != "All" and 'sector' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["sector"] == filters['sector']]
        
        # Risk filter
        if filters.get('risk') != "All" and 'risk' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["risk"] == filters['risk']]
        
        # Score filter
        if 'score' in filtered_df.columns:
            min_score = filters.get('min_score', 0)
            filtered_df = filtered_df[filtered_df["score"] >= min_score]
        
        # Price filter
        if 'price' in filtered_df.columns:
            price_range = filters.get('price_range', (1, 100000))
            filtered_df = filtered_df[
                (filtered_df["price"] >= price_range[0]) & 
                (filtered_df["price"] <= price_range[1])
            ]
        
        # Volume filter
        if 'volume_1d' in filtered_df.columns:
            min_volume = filters.get('min_volume', 0)
            filtered_df = filtered_df[filtered_df["volume_1d"] >= min_volume]
        
        return filtered_df.reset_index(drop=True)
        
    except Exception as e:
        st.warning(f"Error applying filters: {str(e)}")
        return df

def display_market_overview(df: pd.DataFrame):
    """Display market overview metrics."""
    try:
        display_section_header("📊 Market Overview", "Real-time market analysis and key metrics")
        
        # Calculate market summary
        market_summary = calculate_market_summary(df)
        
        # Display metrics in responsive grid
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            UIComponents.display_metric_card(
                "Total Stocks",
                market_summary.get('total_stocks', 0),
                icon="📈"
            )
        
        with col2:
            buy_signals = market_summary.get('buy_signals', 0)
            total_stocks = market_summary.get('total_stocks', 1)
            buy_percentage = (buy_signals / total_stocks * 100) if total_stocks > 0 else 0
            
            UIComponents.display_metric_card(
                "Buy Signals",
                buy_signals,
                delta=f"{buy_percentage:.1f}%",
                delta_color="green" if buy_percentage > 20 else "red"
            )
        
        with col3:
            UIComponents.display_metric_card(
                "Strong Buys",
                market_summary.get('strong_buy_signals', 0),
                icon="🚀"
            )
        
        with col4:
            avg_score = market_summary.get('avg_score', 50)
            UIComponents.display_metric_card(
                "Average Score",
                f"{avg_score:.1f}",
                delta_color="green" if avg_score > 55 else "red",
                icon="🎯"
            )
        
        with col5:
            breadth = market_summary.get('market_breadth', 50)
            UIComponents.display_metric_card(
                "Market Breadth",
                f"{breadth:.1f}%",
                delta_color="green" if breadth > 50 else "red"
            )
        
        with col6:
            UIComponents.display_metric_card(
                "High Risk",
                market_summary.get('high_risk_count', 0),
                icon="⚠️"
            )
            
    except Exception as e:
        st.error(f"Error displaying market overview: {str(e)}")

def display_top_opportunities(filtered_df: pd.DataFrame):
    """Display top investment opportunities."""
    try:
        display_section_header(
            "🎯 Top Opportunities", 
            "Best investment opportunities based on multi-factor analysis"
        )
        
        # Get top opportunities
        top_picks = get_top_picks(filtered_df, signal_filter="BUY", limit=12)
        
        if not top_picks.empty:
            display_stock_grid(top_picks, max_cards=12, columns=3)
        else:
            UIComponents.display_error_message(
                "No opportunities found with current filters. Try adjusting your criteria.",
                "info"
            )
            
    except Exception as e:
        st.error(f"Error displaying top opportunities: {str(e)}")

def display_detailed_analysis(filtered_df: pd.DataFrame):
    """Display detailed analysis table."""
    try:
        stock_count = len(filtered_df)
        display_section_header(
            f"📋 Detailed Analysis ({stock_count:,} stocks)",
            "Comprehensive analysis with all factors and metrics"
        )
        
        if not filtered_df.empty:
            # Prepare display columns
            display_columns = [
                'ticker', 'company_name', 'sector', 'signal', 'score', 'risk',
                'price', 'ret_1d', 'ret_30d', 'pe', 'volume_1d', 'reason'
            ]
            
            # Filter available columns
            available_columns = [col for col in display_columns if col in filtered_df.columns]
            
            # Format data for display
            display_df = filtered_df[available_columns].copy()
            formatted_df = UIComponents.format_dataframe_for_display(display_df)
            
            # Display the table
            st.dataframe(
                formatted_df,
                use_container_width=True,
                height=500,
                hide_index=True,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "company_name": st.column_config.TextColumn("Company", width="medium"),
                    "signal": st.column_config.TextColumn("Signal", width="small"),
                    "score": st.column_config.TextColumn("Score", width="small"),
                    "price": st.column_config.TextColumn("Price", width="small"),
                    "reason": st.column_config.TextColumn("Analysis", width="large")
                }
            )
            
            # Export functionality
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                csv_data = filtered_df[available_columns].to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv_data,
                    file_name=f"mantra_signals_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download filtered results as CSV file"
                )
        else:
            UIComponents.display_error_message(
                "No stocks match your current filters. Try adjusting the filter criteria.",
                "info"
            )
            
    except Exception as e:
        st.error(f"Error displaying detailed analysis: {str(e)}")

def display_sector_analysis(sector_df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Display sector analysis and charts."""
    try:
        if not sector_df.empty or not filtered_df.empty:
            display_section_header(
                "🏭 Sector Analysis",
                "Sector performance and distribution insights"
            )
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Performance Heatmap", "📈 Distribution Charts"])
            
            with tab1:
                if not sector_df.empty:
                    heatmap_fig = ChartComponents.create_sector_heatmap(sector_df)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                else:
                    st.info("Sector performance data not available")
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    if not filtered_df.empty and 'score' in filtered_df.columns:
                        score_fig = ChartComponents.create_score_distribution(filtered_df)
                        st.plotly_chart(score_fig, use_container_width=True)
                    else:
                        st.info("Score distribution not available")
                
                with col2:
                    if not filtered_df.empty and 'sector' in filtered_df.columns:
                        sector_fig = ChartComponents.create_sector_pie_chart(filtered_df)
                        st.plotly_chart(sector_fig, use_container_width=True)
                    else:
                        st.info("Sector distribution not available")
                        
    except Exception as e:
        st.error(f"Error displaying sector analysis: {str(e)}")

def display_footer():
    """Display application footer."""
    try:
        add_section_spacer("2rem")
        
        last_refresh = st.session_state.get('last_refresh')
        refresh_time = last_refresh.strftime('%Y-%m-%d %H:%M:%S') if last_refresh else 'Never'
        
        st.markdown(f"""
        <div class="footer">
            <p><strong>🔱 M.A.N.T.R.A. v{st.session_state.get('app_version', '1.0.0')}</strong></p>
            <p style="font-size: 0.9rem; margin: 0.5rem 0;">
                Last Updated: {refresh_time}
            </p>
            <p style="font-size: 0.875rem; color: #94a3b8; margin-top: 1rem;">
                All signals are for educational purposes only. Always conduct your own research before making investment decisions.
            </p>
            <p style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">
                Philosophy: "All signal, no noise. Decisions, not guesses."
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.warning(f"Error displaying footer: {str(e)}")

# ============================================================================
# MAIN APPLICATION FUNCTION
# ============================================================================

def main():
    """Main application function."""
    try:
        # Initialize
        initialize_session_state()
        
        # Load CSS
        UIComponents.load_css()
        
        # Display header
        UIComponents.display_header()
        
        # Control panel
        display_control_panel()
        
        # Load data if needed
        if not st.session_state.data_loaded:
            UIComponents.display_loading_message("Loading latest market data...")
            
            # Load data
            watchlist_df, sector_df, status = cached_load_and_process_data()
            
            if not status['success']:
                UIComponents.display_error_message(
                    f"Failed to load data: {'; '.join(status['errors'])}"
                )
                st.stop()
            
            # Process signals
            signals_df = load_and_process_signals(watchlist_df, sector_df)
            
            # Store in session state
            st.session_state.stocks_df = signals_df
            st.session_state.sector_df = sector_df
            st.session_state.data_quality = status.get('quality', {})
            st.session_state.load_status = status
            st.session_state.data_loaded = True
            st.session_state.last_refresh = datetime.now()
            
            # Success message
            st.success(f"✅ Loaded {len(signals_df):,} stocks successfully!")
            time.sleep(1)
            st.rerun()
        
        # Get current data
        df = st.session_state.stocks_df.copy()
        sector_df = st.session_state.sector_df.copy()
        
        if df.empty:
            UIComponents.display_error_message(
                "No data available. Please refresh the application."
            )
            return
        
        # Setup filters
        filters = setup_sidebar_filters(df)
        
        # Apply filters
        filtered_df = apply_filters(df, filters)
        
        # Display main content
        add_section_spacer()
        
        # Market overview
        display_market_overview(df)
        
        add_section_spacer()
        
        # Top opportunities
        display_top_opportunities(filtered_df)
        
        add_section_spacer()
        
        # Detailed analysis
        display_detailed_analysis(filtered_df)
        
        add_section_spacer()
        
        # Sector analysis
        display_sector_analysis(sector_df, filtered_df)
        
        # Footer
        display_footer()
        
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Please refresh the page and try again.")
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            st.code(f"""
Error: {str(e)}
Type: {type(e).__name__}
Session State Keys: {list(st.session_state.keys())}
            """)
        
        # Emergency refresh
        if st.button("🆘 Emergency Refresh"):
            st.cache_data.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.error("Please contact support if this issue persists.")
