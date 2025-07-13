"""
ui.py - Clean, Fast UI Components for M.A.N.T.R.A.
==================================================
Minimal, beautiful, performance-optimized
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Optional

class UI:
    """Clean UI components - only essentials"""
    
    @staticmethod
    def load_css():
        """Minimal, clean CSS for dark theme"""
        st.markdown("""
        <style>
        /* Clean dark theme */
        .stApp {
            background-color: #0e1117;
        }
        
        /* Hide Streamlit branding */
        #MainMenu, footer, header {visibility: hidden;}
        
        /* Metric cards */
        .metric-card {
            background: #1a1f2e;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            height: 100%;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #fff;
            margin: 8px 0;
        }
        
        .metric-label {
            font-size: 13px;
            color: #8b949e;
            font-weight: 500;
        }
        
        .metric-delta {
            font-size: 13px;
            font-weight: 600;
        }
        
        /* Stock cards */
        .stock-card {
            background: #1a1f2e;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            transition: border-color 0.2s;
        }
        
        .stock-card:hover {
            border-color: #58a6ff;
        }
        
        /* Signal badges */
        .signal-badge {
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
        }
        
        .signal-strong-buy {
            background: #2ea043;
            color: #fff;
        }
        
        .signal-buy {
            background: #3fb950;
            color: #000;
        }
        
        .signal-watch {
            background: #d29922;
            color: #000;
        }
        
        .signal-neutral {
            background: #6e7681;
            color: #fff;
        }
        
        .signal-avoid {
            background: #da3633;
            color: #fff;
        }
        
        /* Quality indicator */
        .quality-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
        }
        
        .quality-excellent {
            background: rgba(46, 160, 67, 0.15);
            color: #3fb950;
            border: 1px solid #3fb950;
        }
        
        .quality-good {
            background: rgba(88, 166, 255, 0.15);
            color: #58a6ff;
            border: 1px solid #58a6ff;
        }
        
        .quality-fair {
            background: rgba(210, 153, 34, 0.15);
            color: #d29922;
            border: 1px solid #d29922;
        }
        
        /* Clean scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0e1117;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #30363d;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #58a6ff;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card(label: str, value: str, delta: str = "", 
                    delta_color: str = "", icon: str = ""):
        """Clean metric card"""
        delta_html = ""
        if delta:
            color = "#3fb950" if delta_color == "green" else "#da3633" if delta_color == "red" else "#8b949e"
            delta_html = f'<div class="metric-delta" style="color: {color};">{delta}</div>'
        
        st.markdown(f"""
        <div class="metric-card">
            {f'<div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>' if icon else ''}
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def stock_card(stock: pd.Series):
        """Clean stock card with essential info"""
        ticker = stock.get('ticker', 'N/A')
        name = stock.get('name', ticker)[:30]
        price = stock.get('price', 0)
        change = stock.get('ret_1d', 0)
        signal = stock.get('signal', 'NEUTRAL')
        score = stock.get('score', 50)
        volume = stock.get('volume', 0)
        pe = stock.get('pe', 0)
        
        # Signal badge
        signal_class = f"signal-{signal.lower().replace('_', '-')}"
        
        # Price color
        price_color = "#3fb950" if change >= 0 else "#da3633"
        
        st.markdown(f"""
        <div class="stock-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <div style="font-weight: 700; font-size: 16px; color: #fff;">{ticker}</div>
                    <div style="font-size: 12px; color: #8b949e;">{name}</div>
                </div>
                <span class="signal-badge {signal_class}">{signal} ({int(score)})</span>
            </div>
            
            <div style="font-size: 24px; font-weight: 700; color: #fff; margin: 12px 0;">
                ₹{price:,.2f}
                <span style="font-size: 14px; color: {price_color}; margin-left: 8px;">
                    {'+' if change >= 0 else ''}{change:.1f}%
                </span>
            </div>
            
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #8b949e;">
                <span>Vol: {volume/1000:.0f}K</span>
                <span>P/E: {pe:.1f if pe > 0 else 'N/A'}</span>
                <span>30D: {stock.get('ret_30d', 0):+.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def quality_indicator(quality: Dict):
        """Data quality indicator"""
        score = quality.get('score', 0)
        status = quality.get('status', 'Unknown')
        
        if score >= 90:
            badge_class = "quality-excellent"
            icon = "✓"
        elif score >= 75:
            badge_class = "quality-good"
            icon = "✓"
        else:
            badge_class = "quality-fair"
            icon = "!"
        
        st.markdown(f"""
        <div class="quality-badge {badge_class}">
            <span>{icon}</span>
            <span>Data: {status} ({score:.0f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Style dataframe for display"""
        # Format functions
        def format_price(val):
            return f'₹{val:,.2f}' if pd.notna(val) else ''
        
        def format_pct(val):
            if pd.notna(val):
                color = '#3fb950' if val >= 0 else '#da3633'
                return f'<span style="color: {color}">{val:+.1f}%</span>'
            return ''
        
        def format_signal(val):
            colors = {
                'STRONG_BUY': 'background: #2ea043; color: white; font-weight: bold;',
                'BUY': 'background: #3fb950; color: black; font-weight: bold;',
                'WATCH': 'background: #d29922; color: black;',
                'NEUTRAL': 'background: #6e7681; color: white;',
                'AVOID': 'background: #da3633; color: white;'
            }
            style = colors.get(val, '')
            return f'<div style="{style} padding: 2px 8px; border-radius: 4px; text-align: center;">{val}</div>'
        
        # Apply formatting
        formatted = df.copy()
        
        # Format columns if they exist
        if 'price' in formatted.columns:
            formatted['price'] = formatted['price'].apply(format_price)
        
        for col in ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m']:
            if col in formatted.columns:
                formatted[col] = formatted[col].apply(format_pct)
        
        if 'signal' in formatted.columns:
            formatted['signal'] = formatted['signal'].apply(format_signal)
        
        if 'volume' in formatted.columns:
            formatted['volume'] = formatted['volume'].apply(lambda x: f'{x/1000:.0f}K' if pd.notna(x) else '')
        
        if 'pe' in formatted.columns:
            formatted['pe'] = formatted['pe'].apply(lambda x: f'{x:.1f}' if pd.notna(x) and x > 0 else 'N/A')
        
        return formatted
    
    @staticmethod
    def sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
        """Clean sector performance heatmap"""
        if sector_df.empty:
            return go.Figure()
        
        # Prepare data
        sectors = sector_df['sector'].tolist()
        periods = ['1D', '7D', '30D']
        columns = ['sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d']
        
        # Build matrix
        z_values = []
        for _, row in sector_df.iterrows():
            row_values = []
            for col in columns:
                if col in sector_df.columns:
                    val = row[col] if pd.notna(row[col]) else 0
                    row_values.append(float(val))
                else:
                    row_values.append(0)
            z_values.append(row_values)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=periods,
            y=sectors,
            colorscale=[
                [0, '#da3633'],      # Red
                [0.5, '#0e1117'],    # Dark neutral
                [1, '#3fb950']       # Green
            ],
            zmid=0,
            text=[[f'{val:.1f}%' for val in row] for row in z_values],
            texttemplate='%{text}',
            textfont={"size": 12, "color": "white"},
            hovertemplate='%{y}<br>%{x}: %{z:.1f}%<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            height=max(300, len(sectors) * 25),
            margin=dict(l=100, r=20, t=20, b=20),
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='#8b949e', size=12),
            xaxis=dict(side='top', tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=11))
        )
        
        return fig
