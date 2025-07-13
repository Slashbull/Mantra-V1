"""
ui_components.py - M.A.N.T.R.A. UI Components (Final Version)
============================================================
Beautiful, modern UI components for the M.A.N.T.R.A. dashboard.
Optimized for both desktop and mobile viewing.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Optional, Any

from constants import SIGNAL_COLORS, RISK_COLORS, SECTOR_COLORS

class UIComponents:
    """Modern UI components for M.A.N.T.R.A. dashboard"""
    
    @staticmethod
    def load_css():
        """Load custom CSS for modern dark theme"""
        st.markdown("""
        <style>
        /* Main app styling */
        .stApp {
            background-color: #0e1117;
            color: #d1d5da;
        }
        
        /* Hide Streamlit branding */
        #MainMenu, footer, header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Custom metric cards */
        .metric-card {
            background: linear-gradient(135deg, #1a1f2e 0%, #21262d 100%);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .metric-card:hover {
            border-color: #58a6ff;
            box-shadow: 0 6px 12px rgba(88, 166, 255, 0.2);
            transform: translateY(-2px);
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            color: #fff;
            margin: 8px 0;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .metric-label {
            font-size: 14px;
            color: #8b949e;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-delta {
            font-size: 14px;
            font-weight: 600;
            margin-top: 4px;
        }
        
        /* Stock cards */
        .stock-card {
            background: linear-gradient(135deg, #1a1f2e 0%, #21262d 100%);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        .stock-card:hover {
            border-color: #58a6ff;
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(88, 166, 255, 0.15);
        }
        
        .stock-ticker {
            font-size: 20px;
            font-weight: 700;
            color: #58a6ff;
            margin-bottom: 4px;
        }
        
        .stock-name {
            font-size: 13px;
            color: #8b949e;
            margin-bottom: 12px;
        }
        
        .stock-price {
            font-size: 24px;
            font-weight: 700;
            color: #fff;
            margin: 8px 0;
        }
        
        .stock-details {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #8b949e;
            margin-top: 12px;
        }
        
        /* Signal badges */
        .signal-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        /* Custom buttons */
        .stButton button {
            background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(46, 160, 67, 0.2);
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #2ea043 0%, #238636 100%);
            box-shadow: 0 4px 8px rgba(46, 160, 67, 0.3);
            transform: translateY(-1px);
        }
        
        /* Data tables */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #161b22;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #0e1117 0%, #1a1f2e 50%, #0e1117 100%);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid #30363d;
        }
        
        .main-title {
            font-size: 42px;
            font-weight: 700;
            color: #58a6ff;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            margin: 0;
        }
        
        .main-subtitle {
            font-size: 16px;
            color: #8b949e;
            text-align: center;
            margin-top: 8px;
            font-style: italic;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .metric-card {
                height: 100px;
                padding: 15px;
            }
            
            .metric-value {
                font-size: 24px;
            }
            
            .stock-card {
                padding: 15px;
            }
            
            .main-title {
                font-size: 32px;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_header():
        """Display the main header with branding"""
        st.markdown("""
        <div class="main-header">
            <h1 class="main-title">🔱 M.A.N.T.R.A.</h1>
            <p class="main-subtitle">Market Analysis Neural Trading Research Assistant</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card(label: str, value: str, delta: str = "", 
                    delta_color: str = "", icon: str = ""):
        """Display a modern metric card"""
        delta_html = ""
        if delta:
            color = "#2ea043" if delta_color == "green" else "#da3633" if delta_color == "red" else "#8b949e"
            delta_html = f'<div class="metric-delta" style="color: {color};">{delta}</div>'
        
        icon_html = f'<div style="font-size: 20px; margin-bottom: 8px;">{icon}</div>' if icon else ''
        
        st.markdown(f"""
        <div class="metric-card">
            {icon_html}
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def stock_card(stock_data: Dict[str, Any]):
        """Display an individual stock card"""
        ticker = stock_data.get('ticker', 'N/A')
        name = str(stock_data.get('company_name', ticker))[:40] + ('...' if len(str(stock_data.get('company_name', ''))) > 40 else '')
        price = stock_data.get('price', 0)
        ret_1d = stock_data.get('ret_1d', 0)
        signal = stock_data.get('signal', 'NEUTRAL')
        score = stock_data.get('score', 50)
        risk = stock_data.get('risk', 'Medium')
        sector = stock_data.get('sector', 'Unknown')
        
        # Price change color
        price_color = "#2ea043" if ret_1d >= 0 else "#da3633"
        price_symbol = "+" if ret_1d >= 0 else ""
        
        # Signal badge color
        signal_color = SIGNAL_COLORS.get(signal, "#6e7681")
        
        # Risk badge color  
        risk_color = RISK_COLORS.get(risk, "#6e7681")
        
        st.markdown(f"""
        <div class="stock-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <div class="stock-ticker">{ticker}</div>
                    <div class="stock-name">{name}</div>
                </div>
                <div style="text-align: right;">
                    <span class="signal-badge" style="background-color: {signal_color}; color: white;">
                        {signal} ({int(score)})
                    </span>
                </div>
            </div>
            
            <div class="stock-price">
                ₹{price:,.2f}
                <span style="color: {price_color}; font-size: 14px; margin-left: 8px;">
                    {price_symbol}{ret_1d:.1f}%
                </span>
            </div>
            
            <div class="stock-details">
                <span>Risk: <span style="color: {risk_color}; font-weight: 600;">{risk}</span></span>
                <span>Sector: {sector[:15]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def signal_badge(signal: str, score: float = None) -> str:
        """Generate HTML for signal badge"""
        color = SIGNAL_COLORS.get(signal, "#6e7681")
        score_text = f" ({int(score)})" if score is not None else ""
        return f'<span class="signal-badge" style="background-color: {color}; color: white;">{signal}{score_text}</span>'
    
    @staticmethod
    def risk_badge(risk: str) -> str:
        """Generate HTML for risk badge"""
        color = RISK_COLORS.get(risk, "#6e7681")
        return f'<span class="signal-badge" style="background-color: {color}; color: white;">{risk}</span>'
    
    @staticmethod
    def sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
        """Create an interactive sector performance heatmap"""
        if sector_df.empty:
            return go.Figure()
        
        # Prepare data for heatmap
        sectors = sector_df['sector'].tolist()
        periods = ['1D', '7D', '30D']
        
        # Get return columns
        ret_columns = ['sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d']
        available_columns = [col for col in ret_columns if col in sector_df.columns]
        
        if not available_columns:
            return go.Figure()
        
        # Build matrix
        z_values = []
        hover_text = []
        
        for _, row in sector_df.iterrows():
            row_values = []
            row_hover = []
            for i, col in enumerate(available_columns):
                val = row.get(col, 0)
                if pd.notna(val):
                    row_values.append(float(val))
                    row_hover.append(f"{periods[i]}: {val:.1f}%")
                else:
                    row_values.append(0)
                    row_hover.append(f"{periods[i]}: N/A")
            z_values.append(row_values)
            hover_text.append(row_hover)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=periods[:len(available_columns)],
            y=sectors,
            colorscale=[
                [0, '#da3633'],      # Red for negative
                [0.5, '#0e1117'],    # Dark for neutral
                [1, '#2ea043']       # Green for positive
            ],
            zmid=0,
            text=[[f'{val:.1f}%' for val in row] for row in z_values],
            texttemplate='%{text}',
            textfont={"size": 11, "color": "white", "family": "monospace"},
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>',
            colorbar=dict(
                title="Return %",
                titlefont=dict(color="white"),
                tickfont=dict(color="white")
            )
        ))
        
        # Update layout for dark theme
        fig.update_layout(
            height=max(400, len(sectors) * 30),
            margin=dict(l=120, r=40, t=40, b=40),
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white', size=12, family="Arial"),
            title=dict(
                text="Sector Performance Heatmap",
                font=dict(size=18, color="white"),
                x=0.5
            ),
            xaxis=dict(
                side='top',
                tickfont=dict(size=12, color="white"),
                gridcolor='#30363d'
            ),
            yaxis=dict(
                tickfont=dict(size=11, color="white"),
                gridcolor='#30363d'
            )
        )
        
        return fig
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart"""
        if df.empty or 'score' not in df.columns:
            return go.Figure()
        
        fig = px.histogram(
            df, 
            x='score', 
            nbins=20,
            title="Signal Score Distribution",
            color_discrete_sequence=['#58a6ff']
        )
        
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            title=dict(font=dict(size=16, color="white")),
            xaxis=dict(
                title="Signal Score",
                gridcolor='#30363d',
                tickfont=dict(color="white")
            ),
            yaxis=dict(
                title="Number of Stocks",
                gridcolor='#30363d',
                tickfont=dict(color="white")
            )
        )
        
        return fig
    
    @staticmethod
    def create_sector_chart(df: pd.DataFrame) -> go.Figure:
        """Create sector distribution pie chart"""
        if df.empty or 'sector' not in df.columns:
            return go.Figure()
        
        sector_counts = df['sector'].value_counts().head(10)
        
        fig = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="Top 10 Sectors by Stock Count",
            color_discrete_sequence=SECTOR_COLORS
        )
        
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            title=dict(font=dict(size=16, color="white")),
            showlegend=True,
            legend=dict(font=dict(color="white"))
        )
        
        return fig
    
    @staticmethod
    def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Apply styling to dataframe for display"""
        if df.empty:
            return df
        
        styled_df = df.copy()
        
        # Format price columns
        price_cols = [col for col in ['price', 'prev_close'] if col in styled_df.columns]
        for col in price_cols:
            styled_df[col] = styled_df[col].apply(lambda x: f'₹{x:,.2f}' if pd.notna(x) else 'N/A')
        
        # Format percentage columns
        pct_cols = [col for col in styled_df.columns if 'ret_' in col or '_pct' in col]
        for col in pct_cols:
            styled_df[col] = styled_df[col].apply(
                lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A'
            )
        
        # Format volume
        if 'volume_1d' in styled_df.columns:
            styled_df['volume_1d'] = styled_df['volume_1d'].apply(
                lambda x: f'{x/1000:.0f}K' if pd.notna(x) and x > 0 else 'N/A'
            )
        
        # Format PE ratio
        if 'pe' in styled_df.columns:
            styled_df['pe'] = styled_df['pe'].apply(
                lambda x: f'{x:.1f}' if pd.notna(x) and x > 0 else 'N/A'
            )
        
        return styled_df
    
    @staticmethod
    def display_data_quality(quality_info: Dict[str, Any]):
        """Display data quality information"""
        score = quality_info.get('score', 0)
        status = quality_info.get('status', 'Unknown')
        
        if score >= 90:
            color = "#2ea043"
            icon = "✅"
        elif score >= 75:
            color = "#d29922"
            icon = "⚠️"
        else:
            color = "#da3633"
            icon = "❌"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1f2e 0%, #21262d 100%);
            border: 1px solid {color};
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <span style="font-size: 20px;">{icon}</span>
            <div>
                <strong style="color: {color};">Data Quality: {status}</strong>
                <div style="font-size: 12px; color: #8b949e;">
                    Score: {score:.1f}/100 | Rows: {quality_info.get('rows', 0):,}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def quick_metric(label: str, value: Any, delta: str = "") -> None:
    """Quick metric display"""
    UIComponents.metric_card(label, str(value), delta)

def quick_stock_grid(stocks_df: pd.DataFrame, max_cards: int = 12) -> None:
    """Quick stock cards grid display"""
    if stocks_df.empty:
        st.info("No stocks to display")
        return
    
    # Create grid layout
    cols_per_row = 3
    stocks_to_show = stocks_df.head(max_cards)
    
    for i in range(0, len(stocks_to_show), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (_, stock) in enumerate(stocks_to_show.iloc[i:i+cols_per_row].iterrows()):
            with cols[j]:
                UIComponents.stock_card(stock.to_dict())
