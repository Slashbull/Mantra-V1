"""
ui_components.py - M.A.N.T.R.A. UI Components
============================================
FINAL VERSION - Beautiful, Fast, Bug-Free UI Components
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime
from constants import SIGNAL_COLORS, CHART_COLORS

# ============================================================================
# CUSTOM CSS - BEAUTIFUL DARK THEME
# ============================================================================

def load_custom_css():
    """Load custom CSS for beautiful dark theme styling"""
    st.markdown("""
    <style>
    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, #1e2329 0%, #2d3139 100%);
        border: 1px solid #2d3139;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00d26a;
        box-shadow: 0 6px 12px rgba(0, 210, 106, 0.1);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #ffffff;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #8b92a0;
        font-weight: 500;
    }
    
    .metric-delta {
        font-size: 14px;
        font-weight: 600;
    }
    
    .delta-positive {
        color: #00d26a;
    }
    
    .delta-negative {
        color: #ff4b4b;
    }
    
    /* ===== SIGNAL BADGES ===== */
    .signal-badge {
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 13px;
        display: inline-block;
        margin: 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .signal-strong-buy {
        background: linear-gradient(135deg, #00ff00 0%, #00d26a 100%);
        color: #000;
        box-shadow: 0 2px 8px rgba(0, 255, 0, 0.3);
    }
    
    .signal-buy {
        background-color: #00d26a;
        color: #000;
    }
    
    .signal-watch {
        background-color: #ffa500;
        color: #000;
    }
    
    .signal-neutral {
        background-color: #808080;
        color: #fff;
    }
    
    .signal-avoid {
        background-color: #ff4b4b;
        color: #fff;
    }
    
    /* ===== STOCK CARDS ===== */
    .stock-card {
        background: #1e2329;
        border: 1px solid #2d3139;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stock-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d26a 0%, #00ff00 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .stock-card:hover {
        background: #252a31;
        border-color: #00d26a;
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .stock-card:hover::before {
        opacity: 1;
    }
    
    /* ===== ALERTS ===== */
    .alert-banner {
        padding: 16px 24px;
        border-radius: 10px;
        margin: 12px 0;
        font-weight: 500;
        display: flex;
        align-items: center;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .alert-success {
        background: linear-gradient(135deg, #00d26a 0%, #00a854 100%);
        color: #000;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #33b5e5 0%, #0099cc 100%);
        color: #fff;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffa500 0%, #ff8800 100%);
        color: #000;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #ff4b4b 0%, #cc0000 100%);
        color: #fff;
    }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        margin: 30px 0 20px 0;
        position: relative;
        padding-left: 20px;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #00d26a 0%, #00a854 100%);
        border-radius: 2px;
    }
    
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    .section-subtitle {
        font-size: 14px;
        color: #8b92a0;
        margin: 5px 0 0 0;
    }
    
    /* ===== DATA QUALITY BADGE ===== */
    .quality-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .quality-excellent {
        background-color: #00d26a;
        color: #000;
    }
    
    .quality-good {
        background-color: #33b5e5;
        color: #fff;
    }
    
    .quality-warning {
        background-color: #ffa500;
        color: #000;
    }
    
    .quality-poor {
        background-color: #ff4b4b;
        color: #fff;
    }
    
    /* ===== TABLES ===== */
    .dataframe {
        font-size: 14px !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #00d26a 0%, #00a854 100%);
        color: #000;
        border: none;
        padding: 8px 24px;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 210, 106, 0.3);
    }
    
    /* ===== CUSTOM SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e2329;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2d3139;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00d26a;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# METRIC CARDS
# ============================================================================

def metric_card(
    label: str, 
    value: Union[int, float, str], 
    delta: Optional[float] = None,
    icon: str = "",
    format_value: bool = True
):
    """Create a beautiful metric card with optional delta"""
    # Format value
    if format_value and isinstance(value, (int, float)):
        if value >= 1000000:
            formatted_value = f"{value/1000000:.1f}M"
        elif value >= 1000:
            formatted_value = f"{value/1000:.1f}K"
        else:
            formatted_value = f"{value:,.0f}" if isinstance(value, int) else f"{value:.1f}"
    else:
        formatted_value = str(value)
    
    # Build HTML
    delta_html = ""
    if delta is not None:
        delta_class = "delta-positive" if delta >= 0 else "delta-negative"
        delta_symbol = "↑" if delta >= 0 else "↓"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_symbol} {abs(delta):.1f}%</div>'
    
    html = f"""
    <div class="metric-card">
        {f'<div style="font-size: 28px; margin-bottom: 10px;">{icon}</div>' if icon else ''}
        <div class="metric-label">{label}</div>
        <div class="metric-value">{formatted_value}</div>
        {delta_html}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# SIGNAL BADGES
# ============================================================================

def signal_badge(signal: str, score: Optional[float] = None):
    """Create a signal badge with optional score"""
    signal_upper = signal.upper()
    
    # Map signal to CSS class
    signal_class_map = {
        'STRONG_BUY': 'signal-strong-buy',
        'BUY': 'signal-buy',
        'WATCH': 'signal-watch',
        'NEUTRAL': 'signal-neutral',
        'AVOID': 'signal-avoid'
    }
    
    signal_class = signal_class_map.get(signal_upper, 'signal-neutral')
    
    # Format text
    display_text = signal_upper.replace('_', ' ')
    if score is not None:
        display_text += f" ({score:.0f})"
    
    html = f'<span class="signal-badge {signal_class}">{display_text}</span>'
    return html

# ============================================================================
# STOCK CARDS
# ============================================================================

def stock_card(stock: pd.Series, show_details: bool = True):
    """Create a detailed stock card"""
    # Extract data with defaults
    ticker = stock.get('ticker', 'N/A')
    company = stock.get('company_name', ticker)
    price = stock.get('price', 0)
    change = stock.get('ret_1d', 0)
    signal = stock.get('decision', 'NEUTRAL')
    score = stock.get('composite_score', 50)
    
    # Limit company name length
    if len(company) > 30:
        company = company[:27] + "..."
    
    # Additional metrics
    volume = stock.get('volume_1d', 0)
    pe = stock.get('pe', 0)
    risk = stock.get('risk_level', 'Medium')
    rvol = stock.get('rvol', 1.0)
    
    # Build card HTML
    html = f"""
    <div class="stock-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
            <div>
                <h3 style="margin: 0; color: #fff; font-size: 20px;">{ticker}</h3>
                <p style="margin: 0; color: #8b92a0; font-size: 13px;">{company}</p>
            </div>
            <div>{signal_badge(signal, score)}</div>
        </div>
        
        <div style="margin: 20px 0;">
            <div style="font-size: 28px; font-weight: bold; color: #fff;">
                ₹{price:,.2f}
                <span style="font-size: 16px; color: {'#00d26a' if change >= 0 else '#ff4b4b'}; margin-left: 10px;">
                    {'↑' if change >= 0 else '↓'} {abs(change):.1f}%
                </span>
            </div>
        </div>
    """
    
    if show_details:
        html += f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; padding-top: 20px; border-top: 1px solid #2d3139;">
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 11px; text-transform: uppercase;">Volume</p>
                <p style="margin: 5px 0 0 0; color: #fff; font-size: 14px; font-weight: 600;">{volume/1000:.0f}K</p>
                <p style="margin: 2px 0 0 0; color: #00d26a; font-size: 11px;">{rvol:.1f}x avg</p>
            </div>
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 11px; text-transform: uppercase;">P/E Ratio</p>
                <p style="margin: 5px 0 0 0; color: #fff; font-size: 14px; font-weight: 600;">{pe:.1f}</p>
            </div>
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 11px; text-transform: uppercase;">Risk Level</p>
                <p style="margin: 5px 0 0 0; color: #fff; font-size: 14px; font-weight: 600;">{risk}</p>
            </div>
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 11px; text-transform: uppercase;">30D Return</p>
                <p style="margin: 5px 0 0 0; color: {'#00d26a' if stock.get('ret_30d', 0) >= 0 else '#ff4b4b'}; font-size: 14px; font-weight: 600;">
                    {stock.get('ret_30d', 0):+.1f}%
                </p>
            </div>
        </div>
        """
    
    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# CHARTS
# ============================================================================

def sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
    """Create an interactive sector performance heatmap"""
    if sector_df.empty:
        return go.Figure()
    
    # Prepare data - ensure we have the columns
    sectors = sector_df['sector'].tolist()
    
    # Define periods and columns
    periods = ['1D', '7D', '30D', '3M', '6M', '1Y']
    columns = ['sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d', 
               'sector_ret_3m', 'sector_ret_6m', 'sector_ret_1y']
    
    # Build value matrix
    z_values = []
    hover_text = []
    
    for _, row in sector_df.iterrows():
        row_values = []
        row_hover = []
        
        for col in columns:
            if col in sector_df.columns:
                val = row[col]
                # Handle string percentages
                if isinstance(val, str):
                    val = float(val.replace('%', ''))
                elif pd.isna(val):
                    val = 0
                row_values.append(val)
                row_hover.append(f"{val:.1f}%")
            else:
                row_values.append(0)
                row_hover.append("N/A")
        
        z_values.append(row_values)
        hover_text.append(row_hover)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=periods,
        y=sectors,
        text=hover_text,
        texttemplate='%{text}',
        textfont={"size": 12, "color": "white"},
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(
            title="Return %",
            titleside="right",
            tickmode="linear",
            tick0=-20,
            dtick=10
        ),
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Sector Performance Heatmap',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title='Time Period',
        yaxis_title='Sector',
        height=max(400, len(sectors) * 30),
        template='plotly_dark',
        margin=dict(l=150, r=50, t=50, b=50),
        xaxis={'side': 'top'},
        font={'size': 12}
    )
    
    return fig

def create_gauge_chart(value: float, title: str, max_value: float = 100) -> go.Figure:
    """Create a gauge chart for scores"""
    # Determine color based on value
    if value >= 80:
        bar_color = SIGNAL_COLORS['BUY']
    elif value >= 60:
        bar_color = SIGNAL_COLORS['WATCH']
    else:
        bar_color = SIGNAL_COLORS['AVOID']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': 'white'}},
        delta={'reference': 50, 'increasing': {'color': "#00d26a"}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#2d3139",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255, 75, 75, 0.1)'},
                {'range': [40, 60], 'color': 'rgba(255, 165, 0, 0.1)'},
                {'range': [60, 100], 'color': 'rgba(0, 210, 106, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        template='plotly_dark',
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_distribution_chart(
    df: pd.DataFrame, 
    column: str, 
    title: str,
    bins: int = 30
) -> go.Figure:
    """Create a distribution histogram"""
    if column not in df.columns:
        return go.Figure()
    
    # Remove NaN values
    data = df[column].dropna()
    
    # Create histogram
    fig = go.Figure(data=[
        go.Histogram(
            x=data,
            nbinsx=bins,
            marker=dict(
                color=SIGNAL_COLORS['BUY'],
                line=dict(color='white', width=1)
            ),
            opacity=0.8
        )
    ])
    
    # Add mean line
    mean_val = data.mean()
    median_val = data.median()
    
    fig.add_vline(
        x=mean_val, 
        line_dash="dash", 
        line_color="white",
        annotation_text=f"Mean: {mean_val:.1f}",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=median_val, 
        line_dash="dot", 
        line_color="#ffa500",
        annotation_text=f"Median: {median_val:.1f}",
        annotation_position="top left"
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title='Count',
        template='plotly_dark',
        showlegend=False,
        height=350,
        margin=dict(l=50, r=50, t=70, b=50)
    )
    
    return fig

# ============================================================================
# ALERTS AND MESSAGES
# ============================================================================

def show_alert(message: str, alert_type: str = "info", icon: Optional[str] = None):
    """Display an alert message"""
    # Map alert types to classes and default icons
    alert_config = {
        'success': ('alert-success', '✅'),
        'info': ('alert-info', 'ℹ️'),
        'warning': ('alert-warning', '⚠️'),
        'danger': ('alert-danger', '🚨'),
        'error': ('alert-danger', '❌')
    }
    
    alert_class, default_icon = alert_config.get(alert_type, ('alert-info', 'ℹ️'))
    display_icon = icon or default_icon
    
    html = f"""
    <div class="alert-banner {alert_class}">
        <span style="font-size: 20px; margin-right: 12px;">{display_icon}</span>
        <span style="flex: 1;">{message}</span>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# DATA DISPLAY
# ============================================================================

def display_dataframe(
    df: pd.DataFrame, 
    title: str = "",
    height: int = 400,
    hide_index: bool = True,
    highlight_columns: Optional[List[str]] = None
):
    """Display a styled dataframe with formatting"""
    if df.empty:
        show_alert("No data to display", "info")
        return
    
    if title:
        st.markdown(f"### {title}")
    
    # Create a copy to avoid modifying original
    display_df = df.copy()
    
    # Format numeric columns
    format_dict = {}
    
    for col in display_df.columns:
        if 'price' in col.lower() or col in ['target_1', 'target_2', 'stop_loss']:
            format_dict[col] = lambda x: f'₹{x:,.2f}' if pd.notna(x) else ''
        elif 'ret_' in col or '_pct' in col or col.endswith('score'):
            format_dict[col] = lambda x: f'{x:+.1f}%' if pd.notna(x) else ''
        elif 'volume' in col.lower():
            format_dict[col] = lambda x: f'{x:,.0f}' if pd.notna(x) else ''
        elif col == 'pe':
            format_dict[col] = lambda x: f'{x:.1f}' if pd.notna(x) and x > 0 else 'N/A'
    
    # Apply styling
    styled_df = display_df.style
    
    # Format columns
    if format_dict:
        styled_df = styled_df.format(format_dict)
    
    # Highlight columns
    if highlight_columns:
        def highlight_cols(x):
            return ['background-color: rgba(0, 210, 106, 0.1)' if x.name in highlight_columns else '' for _ in x]
        styled_df = styled_df.apply(highlight_cols, axis=0)
    
    # Color code decision column if present
    if 'decision' in display_df.columns:
        def color_decision(val):
            colors = {
                'STRONG_BUY': 'background-color: #00ff00; color: black; font-weight: bold;',
                'BUY': 'background-color: #00d26a; color: black; font-weight: bold;',
                'WATCH': 'background-color: #ffa500; color: black;',
                'NEUTRAL': 'background-color: #808080; color: white;',
                'AVOID': 'background-color: #ff4b4b; color: white;'
            }
            return colors.get(val, '')
        
        styled_df = styled_df.applymap(lambda x: color_decision(x) if x in colors else '', subset=['decision'])
    
    # Display
    st.dataframe(
        styled_df,
        height=height,
        use_container_width=True,
        hide_index=hide_index
    )

# ============================================================================
# LAYOUT HELPERS
# ============================================================================

def section_header(title: str, subtitle: str = ""):
    """Create a section header with optional subtitle"""
    html = f"""
    <div class="section-header">
        <h2 class="section-title">{title}</h2>
        {f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def data_quality_badge(quality_score: float):
    """Display a data quality badge"""
    if quality_score >= 90:
        badge_class = "quality-excellent"
        text = "Excellent"
    elif quality_score >= 75:
        badge_class = "quality-good"
        text = "Good"
    elif quality_score >= 60:
        badge_class = "quality-warning"
        text = "Fair"
    else:
        badge_class = "quality-poor"
        text = "Poor"
    
    html = f'<span class="quality-badge {badge_class}">Data Quality: {text} ({quality_score:.0f}%)</span>'
    st.markdown(html, unsafe_allow_html=True)

def create_columns_adaptive(num_items: int, max_cols: int = 4) -> List:
    """Create adaptive column layout based on number of items"""
    if num_items == 0:
        return []
    elif num_items <= max_cols:
        return st.columns(num_items)
    else:
        # For many items, use max columns
        return st.columns(min(max_cols, num_items))

# ============================================================================
# SUMMARY CARDS
# ============================================================================

def market_summary_card(df: pd.DataFrame):
    """Create a market summary card"""
    if df.empty:
        return
    
    # Calculate metrics
    total_stocks = len(df)
    advancing = (df['ret_1d'] > 0).sum() if 'ret_1d' in df.columns else 0
    declining = (df['ret_1d'] < 0).sum() if 'ret_1d' in df.columns else 0
    unchanged = total_stocks - advancing - declining
    
    breadth = (advancing / total_stocks * 100) if total_stocks > 0 else 0
    
    # Determine market mood
    if breadth > 65:
        mood = "Bullish"
        mood_color = "#00d26a"
        mood_icon = "📈"
    elif breadth < 35:
        mood = "Bearish"
        mood_color = "#ff4b4b"
        mood_icon = "📉"
    else:
        mood = "Neutral"
        mood_color = "#ffa500"
        mood_icon = "➡️"
    
    html = f"""
    <div class="metric-card" style="padding: 30px;">
        <h3 style="margin: 0 0 20px 0; color: #fff; font-size: 24px;">
            Market Summary {mood_icon}
        </h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 14px;">Market Mood</p>
                <p style="margin: 5px 0; color: {mood_color}; font-size: 20px; font-weight: bold;">{mood}</p>
            </div>
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 14px;">Breadth</p>
                <p style="margin: 5px 0; color: #fff; font-size: 20px; font-weight: bold;">{breadth:.1f}%</p>
            </div>
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 14px;">Advancing</p>
                <p style="margin: 5px 0; color: #00d26a; font-size: 20px; font-weight: bold;">{advancing}</p>
            </div>
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 14px;">Declining</p>
                <p style="margin: 5px 0; color: #ff4b4b; font-size: 20px; font-weight: bold;">{declining}</p>
            </div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# MAIN HEADER
# ============================================================================

def dashboard_header(show_time: bool = True):
    """Create main dashboard header"""
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("# 🔱 M.A.N.T.R.A.")
        st.markdown("*Market Analysis Neural Trading Research Assistant*")
    
    with col2:
        # Empty for spacing or additional info
        pass
    
    with col3:
        if show_time:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.markdown(
                f"""
                <div style="text-align: right; padding-top: 20px;">
                    <p style="margin: 0; color: #8b92a0; font-size: 12px;">Last Update</p>
                    <p style="margin: 0; color: #fff; font-weight: bold; font-size: 16px;">{current_time}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================================
# FOOTER
# ============================================================================

def dashboard_footer():
    """Create dashboard footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0; color: #8b92a0;">
            <p style="margin: 0;">🔱 M.A.N.T.R.A. v1.0.0</p>
            <p style="margin: 5px 0; font-size: 12px;">All signals are for educational purposes only. Always do your own research.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_ui():
    """Initialize UI with all configurations"""
    # Load custom CSS
    load_custom_css()
    
    # Any other UI initialization
    pass

# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    # Initialize Streamlit page
    st.set_page_config(
        page_title="M.A.N.T.R.A. UI Test",
        page_icon="🔱",
        layout="wide"
    )
    
    # Initialize UI
    initialize_ui()
    
    # Test components
    st.markdown("# UI Components Test")
    
    # Test metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Stocks", 2487, delta=5.2, icon="📊")
    with col2:
        metric_card("Buy Signals", 142, delta=-2.1, icon="🎯")
    with col3:
        metric_card("Avg Score", 67.8, icon="📈")
    with col4:
        metric_card("Market Cap", 1250000, icon="💰")
    
    # Test alerts
    st.markdown("### Alerts")
    show_alert("System loaded successfully", "success")
    show_alert("52 new opportunities found", "info")
    show_alert("High volatility detected", "warning")
    
    st.markdown("### Badges")
    st.markdown(signal_badge("BUY", 85), unsafe_allow_html=True)
    st.markdown(signal_badge("WATCH", 70), unsafe_allow_html=True)
    st.markdown(signal_badge("AVOID", 35), unsafe_allow_html=True)
