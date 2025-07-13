"""
ui_components.py - M.A.N.T.R.A. UI Components Module
===================================================
Beautiful, responsive UI components with modern design.
Production-ready with comprehensive error handling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Union
import warnings

from constants import (
    SIGNAL_COLORS, RISK_COLORS, SECTOR_COLORS, CUSTOM_CSS
)

warnings.filterwarnings('ignore')

# ============================================================================
# CORE UI COMPONENT CLASS
# ============================================================================

class UIComponents:
    """Modern UI components for M.A.N.T.R.A. dashboard."""
    
    @staticmethod
    def load_css():
        """Load modern CSS styling."""
        try:
            st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Error loading CSS: {str(e)}")
    
    @staticmethod
    def display_header():
        """Display main application header."""
        try:
            st.markdown("""
            <div class="main-header animate-fade-in">
                <h1 class="main-title">🔱 M.A.N.T.R.A.</h1>
                <p class="main-subtitle">Market Analysis Neural Trading Research Assistant</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying header: {str(e)}")
            st.title("🔱 M.A.N.T.R.A.")
            st.caption("Market Analysis Neural Trading Research Assistant")
    
    @staticmethod
    def display_metric_card(
        label: str, 
        value: Union[str, int, float], 
        delta: str = "", 
        delta_color: str = "",
        icon: str = ""
    ):
        """Display animated metric card."""
        try:
            # Format value
            if isinstance(value, (int, float)):
                if value >= 1000:
                    formatted_value = f"{value:,.0f}"
                else:
                    formatted_value = f"{value:.1f}" if isinstance(value, float) else str(value)
            else:
                formatted_value = str(value)
            
            # Delta styling
            delta_html = ""
            if delta:
                if delta_color == "green":
                    color = "#00cc66"
                elif delta_color == "red":
                    color = "#ff4444"
                else:
                    color = "#888888"
                delta_html = f'<div class="metric-delta" style="color: {color};">{delta}</div>'
            
            # Icon
            icon_html = f'<div style="font-size: 1.5rem; margin-bottom: 0.5rem; opacity: 0.8;">{icon}</div>' if icon else ''
            
            st.markdown(f"""
            <div class="metric-card animate-fade-in">
                {icon_html}
                <div class="metric-label">{label}</div>
                <div class="metric-value">{formatted_value}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Error displaying metric card: {str(e)}")
            st.metric(label, value, delta)
    
    @staticmethod
    def display_stock_card(stock_data: Union[pd.Series, Dict[str, Any]]):
        """Display individual stock card with comprehensive data."""
        try:
            # Convert to dict if Series
            if isinstance(stock_data, pd.Series):
                stock = stock_data.to_dict()
            else:
                stock = stock_data
            
            # Extract data with safe defaults
            ticker = str(stock.get('ticker', 'N/A'))
            name = str(stock.get('company_name', ticker))[:45]
            if len(str(stock.get('company_name', ''))) > 45:
                name += "..."
            
            price = pd.to_numeric(stock.get('price', 0), errors='coerce') or 0
            ret_1d = pd.to_numeric(stock.get('ret_1d', 0), errors='coerce') or 0
            ret_30d = pd.to_numeric(stock.get('ret_30d', 0), errors='coerce') or 0
            
            signal = str(stock.get('signal', 'NEUTRAL'))
            score = pd.to_numeric(stock.get('score', 50), errors='coerce') or 50
            risk = str(stock.get('risk', 'Medium'))
            sector = str(stock.get('sector', 'Unknown'))[:15]
            
            pe_ratio = pd.to_numeric(stock.get('pe', 0), errors='coerce') or 0
            volume = pd.to_numeric(stock.get('volume_1d', 0), errors='coerce') or 0
            
            # Color calculations
            price_color = "#00cc66" if ret_1d >= 0 else "#ff4444"
            price_symbol = "+" if ret_1d >= 0 else ""
            
            signal_color = SIGNAL_COLORS.get(signal, "#888888")
            risk_color = RISK_COLORS.get(risk, "#888888")
            
            # Format volume
            if volume >= 1000000:
                volume_str = f"{volume/1000000:.1f}M"
            elif volume >= 1000:
                volume_str = f"{volume/1000:.0f}K"
            else:
                volume_str = f"{volume:.0f}"
            
            st.markdown(f"""
            <div class="stock-card animate-fade-in">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                    <div style="flex: 1;">
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
                    <span style="color: {price_color}; font-size: 1rem; margin-left: 0.5rem;">
                        {price_symbol}{ret_1d:.1f}%
                    </span>
                </div>
                
                <div style="display: flex; justify-content: space-between; margin: 0.75rem 0; font-size: 0.875rem; color: #94a3b8;">
                    <span>30D: <span style="color: {'#00cc66' if ret_30d >= 0 else '#ff4444'};">{ret_30d:+.1f}%</span></span>
                    <span>PE: {pe_ratio:.1f if pe_ratio > 0 else 'N/A'}</span>
                    <span>Vol: {volume_str}</span>
                </div>
                
                <div class="stock-details">
                    <span>Risk: <span style="color: {risk_color}; font-weight: 600;">{risk}</span></span>
                    <span>Sector: {sector}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Error displaying stock card: {str(e)}")
            # Fallback simple display
            ticker = stock_data.get('ticker', 'N/A') if isinstance(stock_data, dict) else getattr(stock_data, 'ticker', 'N/A')
            st.info(f"Stock: {ticker}")
    
    @staticmethod
    def display_data_quality_indicator(quality_data: Dict[str, Any]):
        """Display data quality indicator with detailed metrics."""
        try:
            score = quality_data.get('score', 0)
            status = quality_data.get('status', 'Unknown')
            rows = quality_data.get('rows', 0)
            issues = quality_data.get('issues', [])
            
            # Determine quality class
            if score >= 90:
                quality_class = "quality-excellent"
                icon = "✅"
            elif score >= 75:
                quality_class = "quality-good"
                icon = "✅"
            elif score >= 60:
                quality_class = "quality-fair"
                icon = "⚠️"
            else:
                quality_class = "quality-poor"
                icon = "❌"
            
            # Issues summary
            issues_text = ""
            if issues:
                issues_text = f"<div style='font-size: 0.75rem; color: #fbbf24; margin-top: 0.5rem;'>Issues: {len(issues)}</div>"
            
            st.markdown(f"""
            <div class="quality-indicator {quality_class}">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div>
                    <div style="font-weight: 600; color: #f8fafc;">Data Quality: {status}</div>
                    <div style="font-size: 0.875rem; color: #94a3b8;">
                        Score: {score:.1f}/100 | Stocks: {rows:,}
                    </div>
                    {issues_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Error displaying quality indicator: {str(e)}")
            st.info("Data quality check pending")
    
    @staticmethod
    def display_loading_message(message: str = "Loading data..."):
        """Display loading message with animation."""
        try:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; color: #94a3b8;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">⚡</div>
                <div style="font-size: 1.1rem; font-weight: 500;">{message}</div>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.info(message)
    
    @staticmethod
    def display_error_message(message: str, error_type: str = "error"):
        """Display error message with appropriate styling."""
        try:
            icons = {
                "error": "❌",
                "warning": "⚠️",
                "info": "ℹ️"
            }
            
            colors = {
                "error": "#ff4444",
                "warning": "#fbbf24",
                "info": "#60a5fa"
            }
            
            icon = icons.get(error_type, "❌")
            color = colors.get(error_type, "#ff4444")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e293b, #334155); 
                        border: 1px solid {color}; border-radius: 12px; padding: 1rem; margin: 1rem 0;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <span style="color: #f8fafc; font-weight: 500;">{message}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except:
            if error_type == "error":
                st.error(message)
            elif error_type == "warning":
                st.warning(message)
            else:
                st.info(message)
    
    @staticmethod
    def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
        """Format dataframe for beautiful display."""
        try:
            if df.empty:
                return df
            
            # Work on copy
            display_df = df.copy()
            
            # Format price columns
            price_columns = ['price', 'prev_close']
            for col in price_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f'₹{x:,.2f}' if pd.notna(x) and x > 0 else 'N/A'
                    )
            
            # Format percentage columns
            percentage_columns = [col for col in display_df.columns if 'ret_' in col or '_pct' in col]
            for col in percentage_columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A'
                )
            
            # Format volume
            if 'volume_1d' in display_df.columns:
                display_df['volume_1d'] = display_df['volume_1d'].apply(
                    lambda x: f'{x/1000000:.1f}M' if pd.notna(x) and x >= 1000000
                    else f'{x/1000:.0f}K' if pd.notna(x) and x >= 1000
                    else f'{x:.0f}' if pd.notna(x) else 'N/A'
                )
            
            # Format PE ratio
            if 'pe' in display_df.columns:
                display_df['pe'] = display_df['pe'].apply(
                    lambda x: f'{x:.1f}' if pd.notna(x) and x > 0 else 'N/A'
                )
            
            # Format scores
            score_columns = ['score', 'momentum', 'value', 'volume_score', 'technical']
            for col in score_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A'
                    )
            
            return display_df
            
        except Exception as e:
            st.warning(f"Error formatting dataframe: {str(e)}")
            return df

# ============================================================================
# CHART COMPONENTS
# ============================================================================

class ChartComponents:
    """Chart and visualization components."""
    
    @staticmethod
    def create_sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
        """Create interactive sector performance heatmap."""
        try:
            if sector_df.empty:
                return UIComponents._create_empty_chart("No sector data available")
            
            # Prepare data
            sectors = sector_df['sector'].tolist()
            periods = ['1D', '7D', '30D']
            columns = ['sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d']
            
            # Filter available columns
            available_columns = [col for col in columns if col in sector_df.columns]
            if not available_columns:
                return UIComponents._create_empty_chart("No return data available")
            
            # Build data matrix
            z_values = []
            text_values = []
            
            for _, row in sector_df.iterrows():
                row_values = []
                row_text = []
                
                for i, col in enumerate(available_columns):
                    val = pd.to_numeric(row.get(col, 0), errors='coerce')
                    if pd.notna(val):
                        row_values.append(float(val))
                        row_text.append(f'{val:.1f}%')
                    else:
                        row_values.append(0)
                        row_text.append('N/A')
                
                z_values.append(row_values)
                text_values.append(row_text)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=periods[:len(available_columns)],
                y=sectors,
                text=text_values,
                texttemplate='%{text}',
                textfont={"size": 11, "color": "white", "family": "Inter"},
                colorscale=[
                    [0, '#ff4444'],      # Red for negative
                    [0.5, '#1a202c'],    # Dark for neutral
                    [1, '#00cc66']       # Green for positive
                ],
                zmid=0,
                hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>',
                colorbar=dict(
                    title=dict(text="Return %", font=dict(color="white", size=14)),
                    tickfont=dict(color="white", size=12),
                    bgcolor="rgba(26, 32, 44, 0.8)",
                    bordercolor="rgba(255, 255, 255, 0.2)",
                    borderwidth=1
                )
            ))
            
            # Update layout
            fig.update_layout(
                height=max(400, len(sectors) * 30),
                margin=dict(l=120, r=60, t=40, b=40),
                plot_bgcolor='#0f1419',
                paper_bgcolor='#0f1419',
                font=dict(color='white', size=12, family="Inter"),
                title=dict(
                    text="Sector Performance Heatmap",
                    font=dict(size=18, color="white", family="Inter"),
                    x=0.5,
                    y=0.98
                ),
                xaxis=dict(
                    side='top',
                    tickfont=dict(size=12, color="white"),
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis=dict(
                    tickfont=dict(size=11, color="white"),
                    gridcolor='rgba(255, 255, 255, 0.1)'
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Error creating sector heatmap: {str(e)}")
            return UIComponents._create_empty_chart("Error creating heatmap")
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create signal score distribution chart."""
        try:
            if df.empty or 'score' not in df.columns:
                return UIComponents._create_empty_chart("No score data available")
            
            scores = df['score'].dropna()
            if scores.empty:
                return UIComponents._create_empty_chart("No valid scores")
            
            fig = px.histogram(
                x=scores,
                nbins=20,
                title="Signal Score Distribution",
                color_discrete_sequence=['#60a5fa'],
                opacity=0.8
            )
            
            fig.update_layout(
                plot_bgcolor='#0f1419',
                paper_bgcolor='#0f1419',
                font=dict(color='white', family="Inter"),
                title=dict(
                    font=dict(size=16, color="white", family="Inter"),
                    x=0.5
                ),
                xaxis=dict(
                    title="Signal Score",
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    tickfont=dict(color="white"),
                    titlefont=dict(color="white")
                ),
                yaxis=dict(
                    title="Number of Stocks",
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    tickfont=dict(color="white"),
                    titlefont=dict(color="white")
                ),
                bargap=0.1,
                margin=dict(l=60, r=40, t=60, b=60)
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Error creating score distribution: {str(e)}")
            return UIComponents._create_empty_chart("Error creating distribution")
    
    @staticmethod
    def create_sector_pie_chart(df: pd.DataFrame) -> go.Figure:
        """Create sector distribution pie chart."""
        try:
            if df.empty or 'sector' not in df.columns:
                return UIComponents._create_empty_chart("No sector data available")
            
            sector_counts = df['sector'].value_counts().head(10)
            
            if sector_counts.empty:
                return UIComponents._create_empty_chart("No valid sector data")
            
            fig = px.pie(
                values=sector_counts.values,
                names=sector_counts.index,
                title="Top 10 Sectors by Stock Count",
                color_discrete_sequence=SECTOR_COLORS,
                opacity=0.9
            )
            
            fig.update_layout(
                plot_bgcolor='#0f1419',
                paper_bgcolor='#0f1419',
                font=dict(color='white', family="Inter"),
                title=dict(
                    font=dict(size=16, color="white", family="Inter"),
                    x=0.5
                ),
                legend=dict(
                    font=dict(color="white", size=11),
                    bgcolor="rgba(26, 32, 44, 0.8)",
                    bordercolor="rgba(255, 255, 255, 0.2)",
                    borderwidth=1
                ),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=10,
                marker=dict(line=dict(color='#0f1419', width=2))
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Error creating sector pie chart: {str(e)}")
            return UIComponents._create_empty_chart("Error creating pie chart")

    @staticmethod
    def _create_empty_chart(message: str) -> go.Figure:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="white"),
            showarrow=False
        )
        fig.update_layout(
            plot_bgcolor='#0f1419',
            paper_bgcolor='#0f1419',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_signal_badge_html(signal: str, score: Optional[float] = None) -> str:
    """Create HTML for signal badge."""
    try:
        color = SIGNAL_COLORS.get(signal, "#888888")
        score_text = f" ({int(score)})" if score is not None else ""
        return f'''
        <span class="signal-badge" style="background-color: {color}; color: white;">
            {signal}{score_text}
        </span>
        '''
    except:
        return signal

def create_risk_badge_html(risk: str) -> str:
    """Create HTML for risk badge."""
    try:
        color = RISK_COLORS.get(risk, "#888888")
        return f'''
        <span class="signal-badge" style="background-color: {color}; color: white;">
            {risk}
        </span>
        '''
    except:
        return risk

def display_stock_grid(stocks_df: pd.DataFrame, max_cards: int = 12, columns: int = 3):
    """Display stocks in a responsive grid layout."""
    try:
        if stocks_df.empty:
            UIComponents.display_error_message("No stocks to display", "info")
            return
        
        stocks_to_show = stocks_df.head(max_cards)
        
        # Create responsive grid
        for i in range(0, len(stocks_to_show), columns):
            cols = st.columns(columns)
            for j, (_, stock) in enumerate(stocks_to_show.iloc[i:i+columns].iterrows()):
                if j < len(cols):
                    with cols[j]:
                        UIComponents.display_stock_card(stock)
                        
    except Exception as e:
        st.warning(f"Error displaying stock grid: {str(e)}")
        st.dataframe(stocks_df.head(max_cards))

def display_section_header(title: str, subtitle: str = "", icon: str = ""):
    """Display section header with optional subtitle and icon."""
    try:
        icon_html = f'<span style="margin-right: 0.5rem;">{icon}</span>' if icon else ''
        subtitle_html = f'<p style="color: #94a3b8; font-size: 1rem; margin: 0.5rem 0 0 0;">{subtitle}</p>' if subtitle else ''
        
        st.markdown(f"""
        <div style="margin: 2rem 0 1rem 0;">
            <h3 style="color: #f8fafc; font-size: 1.5rem; font-weight: 600; margin: 0; display: flex; align-items: center;">
                {icon_html}{title}
            </h3>
            {subtitle_html}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.warning(f"Error displaying section header: {str(e)}")
        st.subheader(f"{icon} {title}")
        if subtitle:
            st.caption(subtitle)

def add_section_spacer(height: str = "2rem"):
    """Add visual spacer between sections."""
    try:
        st.markdown(f'<div style="height: {height};"></div>', unsafe_allow_html=True)
    except:
        st.markdown("---")
