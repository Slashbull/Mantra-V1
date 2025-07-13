"""
signals.py - M.A.N.T.R.A. Signal Engine Module
==============================================
Advanced signal calculation with comprehensive error handling.
Production-ready multi-factor analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import warnings

from constants import (
    SIGNAL_THRESHOLDS, FACTOR_WEIGHTS, SIGNAL_COLORS, RISK_COLORS
)

warnings.filterwarnings('ignore')

# ============================================================================
# CORE SIGNAL CALCULATION ENGINE
# ============================================================================

class SignalEngine:
    """Production-ready signal calculation engine."""
    
    @staticmethod
    def calculate_all_signals(
        df: pd.DataFrame, 
        sector_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive trading signals for all stocks.
        
        Args:
            df: Stock data DataFrame
            sector_df: Optional sector performance data
            
        Returns:
            DataFrame with calculated signals, scores, and explanations
        """
        if df.empty:
            return df
        
        try:
            # Work on copy to avoid modifying original
            result_df = df.copy()
            
            # Calculate individual factor scores
            result_df = SignalEngine._calculate_factor_scores(result_df)
            
            # Calculate composite score
            result_df = SignalEngine._calculate_composite_score(result_df)
            
            # Generate signal tags
            result_df = SignalEngine._generate_signal_tags(result_df)
            
            # Assess risk levels
            result_df = SignalEngine._assess_risk_levels(result_df)
            
            # Add sector context if available
            if sector_df is not None and not sector_df.empty:
                result_df = SignalEngine._add_sector_context(result_df, sector_df)
            
            # Generate explanations
            result_df = SignalEngine._generate_explanations(result_df)
            
            # Final cleanup and sorting
            result_df = SignalEngine._finalize_results(result_df)
            
            return result_df
            
        except Exception as e:
            st.error(f"Error calculating signals: {str(e)}")
            return df
    
    @staticmethod
    def _calculate_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate individual factor scores."""
        try:
            # Momentum Score
            df['momentum'] = SignalEngine._calculate_momentum_score(df)
            
            # Value Score
            df['value'] = SignalEngine._calculate_value_score(df)
            
            # Volume Score
            df['volume_score'] = SignalEngine._calculate_volume_score(df)
            
            # Technical Score
            df['technical'] = SignalEngine._calculate_technical_score(df)
            
            return df
            
        except Exception as e:
            st.warning(f"Error calculating factor scores: {str(e)}")
            # Set default scores
            for factor in ['momentum', 'value', 'volume_score', 'technical']:
                if factor not in df.columns:
                    df[factor] = 50.0
            return df
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on price returns."""
        try:
            score = pd.Series(50.0, index=df.index)
            
            # Return weights (recent returns weighted more heavily)
            return_weights = {
                'ret_1d': 0.10,
                'ret_7d': 0.20,
                'ret_30d': 0.40,
                'ret_3m': 0.30
            }
            
            momentum_value = pd.Series(0.0, index=df.index)
            total_weight = 0
            
            for ret_col, weight in return_weights.items():
                if ret_col in df.columns:
                    # Clean returns data
                    returns = pd.to_numeric(df[ret_col], errors='coerce').fillna(0)
                    
                    # Normalize to 0-100 scale (±100% return = 0-100 score)
                    normalized = 50 + np.clip(returns / 2, -50, 50)
                    
                    momentum_value += normalized * weight
                    total_weight += weight
            
            if total_weight > 0:
                score = momentum_value / total_weight
            
            # Bonus for consistent uptrend
            if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
                returns_7d = pd.to_numeric(df['ret_7d'], errors='coerce').fillna(0)
                returns_30d = pd.to_numeric(df['ret_30d'], errors='coerce').fillna(0)
                uptrend_bonus = ((returns_7d > 0) & (returns_30d > 0)) * 10
                score += uptrend_bonus
            
            # Bonus for strong recent performance
            if 'ret_1d' in df.columns:
                ret_1d = pd.to_numeric(df['ret_1d'], errors='coerce').fillna(0)
                strong_day_bonus = (ret_1d > 5) * 5
                score += strong_day_bonus
            
            # Penalty for reversal patterns
            if all(col in df.columns for col in ['ret_1d', 'ret_30d']):
                ret_1d = pd.to_numeric(df['ret_1d'], errors='coerce').fillna(0)
                ret_30d = pd.to_numeric(df['ret_30d'], errors='coerce').fillna(0)
                reversal_penalty = ((ret_1d < -3) & (ret_30d > 15)) * 10
                score -= reversal_penalty
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            st.warning(f"Error calculating momentum score: {str(e)}")
            return pd.Series(50.0, index=df.index)
    
    @staticmethod
    def _calculate_value_score(df: pd.DataFrame) -> pd.Series:
        """Calculate value score based on PE ratio and earnings."""
        try:
            score = pd.Series(50.0, index=df.index)
            
            if 'pe' not in df.columns:
                return score
            
            pe_ratio = pd.to_numeric(df['pe'], errors='coerce').fillna(0)
            
            # PE-based scoring with Indian market context
            pe_conditions = [
                (pe_ratio > 0) & (pe_ratio <= 12),    # Deep value
                (pe_ratio > 12) & (pe_ratio <= 20),   # Good value
                (pe_ratio > 20) & (pe_ratio <= 30),   # Fair value
                (pe_ratio > 30) & (pe_ratio <= 50),   # Getting expensive
                (pe_ratio > 50),                      # Very expensive
                (pe_ratio <= 0)                       # Loss-making/invalid
            ]
            pe_scores = [95, 80, 60, 40, 20, 25]
            
            score = pd.Series(
                np.select(pe_conditions, pe_scores, default=50),
                index=df.index
            ).astype(float)
            
            # EPS growth bonus
            if 'eps_change_pct' in df.columns:
                eps_change = pd.to_numeric(df['eps_change_pct'], errors='coerce').fillna(0)
                
                # Strong growth bonus
                growth_bonus = np.where(eps_change > 25, 15,
                               np.where(eps_change > 10, 10,
                               np.where(eps_change > 0, 5, 0)))
                score += growth_bonus
                
                # Decline penalty
                decline_penalty = np.where(eps_change < -20, 20,
                                 np.where(eps_change < -10, 15,
                                 np.where(eps_change < 0, 5, 0)))
                score -= decline_penalty
            
            # Current EPS validation
            if 'eps_current' in df.columns:
                eps_current = pd.to_numeric(df['eps_current'], errors='coerce').fillna(0)
                # Bonus for positive, growing EPS
                positive_eps_bonus = (eps_current > 0) * 5
                score += positive_eps_bonus
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            st.warning(f"Error calculating value score: {str(e)}")
            return pd.Series(50.0, index=df.index)
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate volume score based on relative volume."""
        try:
            score = pd.Series(50.0, index=df.index)
            
            if 'rvol' not in df.columns:
                return score
            
            rvol = pd.to_numeric(df['rvol'], errors='coerce').fillna(1.0)
            
            # Volume-based scoring
            volume_conditions = [
                rvol >= 5.0,    # Extreme volume
                rvol >= 3.0,    # Very high volume
                rvol >= 2.0,    # High volume
                rvol >= 1.5,    # Elevated volume
                rvol >= 1.0,    # Normal volume
                rvol >= 0.5,    # Low volume
                rvol < 0.5      # Very low volume
            ]
            volume_scores = [95, 85, 75, 65, 55, 35, 20]
            
            score = pd.Series(
                np.select(volume_conditions, volume_scores, default=50),
                index=df.index
            ).astype(float)
            
            # Context-based adjustments
            if 'ret_1d' in df.columns:
                ret_1d = pd.to_numeric(df['ret_1d'], errors='coerce').fillna(0)
                
                # Volume surge with price up (positive signal)
                bullish_volume = ((rvol > 2) & (ret_1d > 2)) * 10
                score += bullish_volume
                
                # Volume surge with price down (negative signal)
                bearish_volume = ((rvol > 2) & (ret_1d < -2)) * 10
                score -= bearish_volume
            
            # Sustained volume bonus
            if 'volume_spike' in df.columns:
                sustained_volume = df['volume_spike'].fillna(False) * 5
                score += sustained_volume
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            st.warning(f"Error calculating volume score: {str(e)}")
            return pd.Series(50.0, index=df.index)
    
    @staticmethod
    def _calculate_technical_score(df: pd.DataFrame) -> pd.Series:
        """Calculate technical score based on price patterns and indicators."""
        try:
            score = pd.Series(50.0, index=df.index)
            
            # Price above moving averages
            sma_indicators = ['above_sma_20d', 'above_sma_50d']
            for indicator in sma_indicators:
                if indicator in df.columns:
                    sma_bonus = df[indicator].fillna(False).astype(int) * 15
                    score += sma_bonus
            
            # 52-week position
            if 'position_52w' in df.columns:
                position = pd.to_numeric(df['position_52w'], errors='coerce').fillna(50)
                
                # Position-based scoring
                position_score = np.where(position > 90, 20,    # Near highs
                                np.where(position > 70, 15,     # Upper range
                                np.where(position > 30, 5,      # Middle range
                                np.where(position > 10, -5,     # Lower range
                                -10))))                          # Near lows
                score += position_score
                
                # Breakout bonus
                breakout_bonus = (position > 95) * 10
                score += breakout_bonus
            
            # Price trend indicators
            if 'uptrend' in df.columns:
                trend_bonus = df['uptrend'].fillna(False).astype(int) * 10
                score += trend_bonus
            
            # Near highs/lows
            if 'near_high' in df.columns:
                high_bonus = df['near_high'].fillna(False).astype(int) * 8
                score += high_bonus
            
            if 'near_low' in df.columns:
                low_penalty = df['near_low'].fillna(False).astype(int) * 8
                score -= low_penalty
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            st.warning(f"Error calculating technical score: {str(e)}")
            return pd.Series(50.0, index=df.index)
    
    @staticmethod
    def _calculate_composite_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted composite score."""
        try:
            factors = ['momentum', 'value', 'volume_score', 'technical']
            
            # Ensure all factors exist
            for factor in factors:
                if factor not in df.columns:
                    df[factor] = 50.0
            
            # Calculate weighted score
            composite_score = (
                df['momentum'].fillna(50) * FACTOR_WEIGHTS['momentum'] +
                df['value'].fillna(50) * FACTOR_WEIGHTS['value'] +
                df['volume_score'].fillna(50) * FACTOR_WEIGHTS['volume'] +
                df['technical'].fillna(50) * FACTOR_WEIGHTS['technical']
            )
            
            df['score'] = composite_score.round(1)
            
            return df
            
        except Exception as e:
            st.warning(f"Error calculating composite score: {str(e)}")
            df['score'] = 50.0
            return df
    
    @staticmethod
    def _generate_signal_tags(df: pd.DataFrame) -> pd.DataFrame:
        """Generate signal tags based on composite scores."""
        try:
            def score_to_signal(score):
                if pd.isna(score):
                    return "NEUTRAL"
                elif score >= SIGNAL_THRESHOLDS['STRONG_BUY']:
                    return "STRONG_BUY"
                elif score >= SIGNAL_THRESHOLDS['BUY']:
                    return "BUY"
                elif score >= SIGNAL_THRESHOLDS['WATCH']:
                    return "WATCH"
                elif score >= SIGNAL_THRESHOLDS['NEUTRAL']:
                    return "NEUTRAL"
                else:
                    return "AVOID"
            
            df['signal'] = df['score'].apply(score_to_signal)
            
            return df
            
        except Exception as e:
            st.warning(f"Error generating signal tags: {str(e)}")
            df['signal'] = 'NEUTRAL'
            return df
    
    @staticmethod
    def _assess_risk_levels(df: pd.DataFrame) -> pd.DataFrame:
        """Assess risk level for each stock."""
        try:
            def calculate_risk(row):
                risk_factors = 0
                
                # PE ratio risk
                pe = pd.to_numeric(row.get('pe', 0), errors='coerce')
                if pd.notna(pe):
                    if pe > 50:
                        risk_factors += 3
                    elif pe > 30:
                        risk_factors += 2
                    elif pe <= 0:
                        risk_factors += 3  # Loss-making
                
                # Volume risk
                volume = pd.to_numeric(row.get('volume_1d', 0), errors='coerce')
                if pd.notna(volume) and volume < 10000:
                    risk_factors += 2
                
                # Relative volume risk
                rvol = pd.to_numeric(row.get('rvol', 1), errors='coerce')
                if pd.notna(rvol) and rvol < 0.5:
                    risk_factors += 1
                
                # Price risk (penny stocks)
                price = pd.to_numeric(row.get('price', 0), errors='coerce')
                if pd.notna(price):
                    if price < 20:
                        risk_factors += 3
                    elif price < 50:
                        risk_factors += 1
                
                # EPS decline risk
                eps_change = pd.to_numeric(row.get('eps_change_pct', 0), errors='coerce')
                if pd.notna(eps_change):
                    if eps_change < -30:
                        risk_factors += 3
                    elif eps_change < -10:
                        risk_factors += 2
                
                # Market cap risk
                try:
                    category = str(row.get('category', '')).lower()
                    if 'micro' in category or 'nano' in category:
                        risk_factors += 2
                    elif 'small' in category:
                        risk_factors += 1
                except:
                    pass
                
                # Position risk (near lows)
                position_52w = pd.to_numeric(row.get('position_52w', 50), errors='coerce')
                if pd.notna(position_52w) and position_52w < 20:
                    risk_factors += 1
                
                # Risk categorization
                if risk_factors >= 6:
                    return "High"
                elif risk_factors >= 3:
                    return "Medium"
                else:
                    return "Low"
            
            df['risk'] = df.apply(calculate_risk, axis=1)
            
            return df
            
        except Exception as e:
            st.warning(f"Error assessing risk levels: {str(e)}")
            df['risk'] = 'Medium'
            return df
    
    @staticmethod
    def _add_sector_context(df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Add sector performance context."""
        try:
            if 'sector' not in df.columns or 'sector' not in sector_df.columns:
                return df
            
            # Create sector strength mapping
            sector_strength = {}
            
            if 'sector_ret_30d' in sector_df.columns:
                for _, row in sector_df.iterrows():
                    sector = row['sector']
                    ret_30d = pd.to_numeric(row.get('sector_ret_30d', 0), errors='coerce')
                    if pd.notna(ret_30d):
                        sector_strength[sector] = ret_30d
            
            # Map sector strength to stocks
            df['sector_strength'] = df['sector'].map(sector_strength).fillna(0)
            
            # Adjust score slightly based on sector momentum
            if 'score' in df.columns:
                sector_adjustment = np.clip(df['sector_strength'] / 10, -5, 5)
                df['score'] = (df['score'] + sector_adjustment).clip(0, 100).round(1)
            
            return df
            
        except Exception as e:
            st.warning(f"Error adding sector context: {str(e)}")
            return df
    
    @staticmethod
    def _generate_explanations(df: pd.DataFrame) -> pd.DataFrame:
        """Generate human-readable explanations for signals."""
        try:
            def build_explanation(row):
                explanations = []
                
                # Factor-based explanations
                factors = {
                    'momentum': 'momentum trends',
                    'value': 'valuation metrics',
                    'volume_score': 'volume activity',
                    'technical': 'technical patterns'
                }
                
                for factor, description in factors.items():
                    if factor in row:
                        score = pd.to_numeric(row[factor], errors='coerce')
                        if pd.notna(score):
                            if score >= 80:
                                explanations.append(f"Strong {description}")
                            elif score >= 70:
                                explanations.append(f"Good {description}")
                            elif score <= 30:
                                explanations.append(f"Weak {description}")
                
                # Specific conditions
                try:
                    # High PE warning
                    pe = pd.to_numeric(row.get('pe', 0), errors='coerce')
                    if pd.notna(pe) and pe > 40:
                        explanations.append("High PE ratio")
                    
                    # Strong earnings growth
                    eps_change = pd.to_numeric(row.get('eps_change_pct', 0), errors='coerce')
                    if pd.notna(eps_change) and eps_change > 30:
                        explanations.append("Strong earnings growth")
                    
                    # Volume surge
                    rvol = pd.to_numeric(row.get('rvol', 1), errors='coerce')
                    if pd.notna(rvol) and rvol > 3:
                        explanations.append("Volume surge")
                    
                    # Risk warning
                    if row.get('risk') == 'High':
                        explanations.append("⚠️ High risk factors")
                    
                    # Near highs
                    position = pd.to_numeric(row.get('position_52w', 50), errors='coerce')
                    if pd.notna(position) and position > 90:
                        explanations.append("Near 52W highs")
                
                except:
                    pass
                
                return "; ".join(explanations) if explanations else "Mixed signals"
            
            df['reason'] = df.apply(build_explanation, axis=1)
            
            return df
            
        except Exception as e:
            st.warning(f"Error generating explanations: {str(e)}")
            df['reason'] = 'Analysis pending'
            return df
    
    @staticmethod
    def _finalize_results(df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and optimization."""
        try:
            # Fill any remaining NaN values
            essential_columns = {
                'score': 50.0,
                'signal': 'NEUTRAL',
                'risk': 'Medium',
                'reason': 'No analysis',
                'momentum': 50.0,
                'value': 50.0,
                'volume_score': 50.0,
                'technical': 50.0
            }
            
            for col, default_value in essential_columns.items():
                if col in df.columns:
                    df[col] = df[col].fillna(default_value)
                else:
                    df[col] = default_value
            
            # Sort by score descending
            df = df.sort_values('score', ascending=False, na_last=True)
            
            # Reset index
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            st.warning(f"Error finalizing results: {str(e)}")
            return df

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_top_picks(df: pd.DataFrame, signal_filter: str = "BUY", limit: int = 20) -> pd.DataFrame:
    """Get top stock picks based on signals."""
    try:
        if df.empty or 'signal' not in df.columns:
            return pd.DataFrame()
        
        if signal_filter == "BUY":
            filtered = df[df['signal'].isin(['STRONG_BUY', 'BUY'])]
        elif signal_filter == "ALL":
            filtered = df
        else:
            filtered = df[df['signal'] == signal_filter]
        
        if 'score' in filtered.columns:
            return filtered.nlargest(limit, 'score')
        else:
            return filtered.head(limit)
            
    except Exception as e:
        st.warning(f"Error getting top picks: {str(e)}")
        return pd.DataFrame()

def calculate_market_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate market summary statistics."""
    try:
        if df.empty:
            return {
                'total_stocks': 0,
                'buy_signals': 0,
                'strong_buy_signals': 0,
                'avoid_signals': 0,
                'avg_score': 50,
                'high_risk_count': 0,
                'market_breadth': 50
            }
        
        summary = {
            'total_stocks': len(df),
            'buy_signals': len(df[df['signal'].isin(['STRONG_BUY', 'BUY'])]) if 'signal' in df.columns else 0,
            'strong_buy_signals': len(df[df['signal'] == 'STRONG_BUY']) if 'signal' in df.columns else 0,
            'avoid_signals': len(df[df['signal'] == 'AVOID']) if 'signal' in df.columns else 0,
            'avg_score': df['score'].mean() if 'score' in df.columns else 50,
            'high_risk_count': len(df[df['risk'] == 'High']) if 'risk' in df.columns else 0
        }
        
        # Market breadth (% of stocks up today)
        if 'ret_1d' in df.columns:
            ret_1d = pd.to_numeric(df['ret_1d'], errors='coerce')
            positive_returns = (ret_1d > 0).sum()
            summary['market_breadth'] = (positive_returns / len(df) * 100) if len(df) > 0 else 50
        else:
            summary['market_breadth'] = 50
        
        return summary
        
    except Exception as e:
        st.warning(f"Error calculating market summary: {str(e)}")
        return {
            'total_stocks': len(df) if not df.empty else 0,
            'buy_signals': 0,
            'strong_buy_signals': 0,
            'avoid_signals': 0,
            'avg_score': 50,
            'high_risk_count': 0,
            'market_breadth': 50
        }

def get_sector_leaders(df: pd.DataFrame) -> pd.DataFrame:
    """Get top stock from each sector."""
    try:
        if df.empty or 'sector' not in df.columns or 'score' not in df.columns:
            return pd.DataFrame()
        
        sector_leaders = []
        for sector in df['sector'].dropna().unique():
            sector_stocks = df[df['sector'] == sector]
            if not sector_stocks.empty:
                leader = sector_stocks.nlargest(1, 'score')
                sector_leaders.append(leader)
        
        return pd.concat(sector_leaders, ignore_index=True) if sector_leaders else pd.DataFrame()
        
    except Exception as e:
        st.warning(f"Error getting sector leaders: {str(e)}")
        return pd.DataFrame()

def format_signal_for_display(signal: str, score: float = None) -> str:
    """Format signal for HTML display."""
    try:
        color = SIGNAL_COLORS.get(signal, "#888888")
        score_text = f" ({int(score)})" if score is not None else ""
        return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;">{signal}{score_text}</span>'
    except:
        return signal

def format_risk_for_display(risk: str) -> str:
    """Format risk for HTML display."""
    try:
        color = RISK_COLORS.get(risk, "#888888")
        return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;">{risk}</span>'
    except:
        return risk
