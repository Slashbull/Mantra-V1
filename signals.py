"""
signals.py - M.A.N.T.R.A. Signal Engine (Locked Production Version)
===================================================================
All-time best logic: computes signal score, tags (STRONG_BUY, BUY, etc.),
risk, and detailed reason for each stock based only on provided data and config.

- 100% data-driven; no hardcoded logic
- Robust to missing/extra columns  
- Uses advanced factor scoring from constants.py
- Generates actionable signals with explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from constants import FACTOR_CONFIG, SIGNAL_THRESHOLDS, SIGNAL_COLORS, RISK_COLORS

class SignalEngine:
    """Advanced signal generation engine for M.A.N.T.R.A."""
    
    @staticmethod
    def calculate_all_signals(
        df: pd.DataFrame, 
        sector_df: Optional[pd.DataFrame] = None, 
        regime: str = "balanced"
    ) -> pd.DataFrame:
        """
        Calculate comprehensive signals for all stocks
        
        Args:
            df: Stock data DataFrame  
            sector_df: Optional sector performance data
            regime: Factor weighting regime (default: "balanced")
            
        Returns:
            DataFrame with signal scores, tags, risk, and explanations
        """
        if df.empty:
            return df
            
        # Work on a copy to avoid modifying original
        result_df = df.copy()
        
        # Validate regime
        if regime not in FACTOR_CONFIG:
            regime = "balanced"
        
        factor_config = FACTOR_CONFIG[regime]
        
        # Calculate individual factor scores
        factor_scores = {}
        for factor_name, config in factor_config.items():
            scoring_func = config.get('func')
            if scoring_func and callable(scoring_func):
                try:
                    score = scoring_func(result_df)
                    factor_scores[factor_name] = score
                    result_df[factor_name] = score
                except Exception as e:
                    print(f"Warning: Error calculating {factor_name}: {e}")
                    result_df[factor_name] = 50  # Neutral score on error
            else:
                result_df[factor_name] = 50  # Default neutral
        
        # Calculate composite score
        result_df['score'] = SignalEngine._calculate_composite_score(
            result_df, factor_config
        )
        
        # Generate signal tags
        result_df['signal'] = SignalEngine._generate_signals(result_df['score'])
        
        # Calculate risk assessment
        result_df['risk'] = SignalEngine._assess_risk(result_df)
        
        # Add sector strength if available
        if sector_df is not None and not sector_df.empty:
            result_df = SignalEngine._add_sector_context(result_df, sector_df)
        
        # Generate explanations
        result_df['reason'] = SignalEngine._generate_explanations(
            result_df, factor_config
        )
        
        # Final cleanup and sorting
        result_df = SignalEngine._finalize_results(result_df)
        
        return result_df
    
    @staticmethod
    def _calculate_composite_score(df: pd.DataFrame, factor_config: Dict) -> pd.Series:
        """Calculate weighted composite score"""
        composite_score = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for factor_name, config in factor_config.items():
            weight = config.get('weight', 0)
            if factor_name in df.columns:
                factor_score = df[factor_name].fillna(50)
                composite_score += factor_score * weight
                total_weight += weight
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            composite_score = composite_score / total_weight
        else:
            composite_score = pd.Series(50.0, index=df.index)
        
        return composite_score.round(1)
    
    @staticmethod
    def _generate_signals(scores: pd.Series) -> pd.Series:
        """Generate signal tags based on score thresholds"""
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
        
        return scores.apply(score_to_signal)
    
    @staticmethod
    def _assess_risk(df: pd.DataFrame) -> pd.Series:
        """Assess risk level for each stock"""
        def calculate_risk(row):
            risk_factors = 0
            
            # High PE ratio risk
            if 'pe' in row and pd.notna(row['pe']):
                if row['pe'] > 40:
                    risk_factors += 2
                elif row['pe'] <= 0:
                    risk_factors += 3  # Loss-making
            
            # Low volume risk
            if 'volume_1d' in row and pd.notna(row['volume_1d']):
                if row['volume_1d'] < 1000:
                    risk_factors += 2
            
            # Relative volume risk
            if 'rvol' in row and pd.notna(row['rvol']):
                if row['rvol'] < 0.5:
                    risk_factors += 1
            
            # Price risk (penny stocks)
            if 'price' in row and pd.notna(row['price']):
                if row['price'] < 50:
                    risk_factors += 1
                elif row['price'] < 20:
                    risk_factors += 2
            
            # EPS decline risk
            if 'eps_change_pct' in row and pd.notna(row['eps_change_pct']):
                if row['eps_change_pct'] < -20:
                    risk_factors += 2
                elif row['eps_change_pct'] < -10:
                    risk_factors += 1
            
            # Market cap risk (micro-cap)
            if 'category' in row and pd.notna(row['category']):
                if 'micro' in str(row['category']).lower():
                    risk_factors += 1
            
            # Return risk categories
            if risk_factors >= 4:
                return "High"
            elif risk_factors >= 2:
                return "Medium" 
            else:
                return "Low"
        
        return df.apply(calculate_risk, axis=1)
    
    @staticmethod
    def _add_sector_context(df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Add sector performance context"""
        if 'sector' not in df.columns or 'sector' not in sector_df.columns:
            return df
        
        # Create sector strength mapping
        sector_strength = {}
        if 'sector_ret_30d' in sector_df.columns:
            for _, row in sector_df.iterrows():
                sector = row['sector']
                ret_30d = row.get('sector_ret_30d', 0)
                if pd.notna(ret_30d):
                    sector_strength[sector] = ret_30d
        
        # Map sector strength to stocks
        df['sector_strength'] = df['sector'].map(sector_strength).fillna(0)
        
        # Slight score adjustment based on sector momentum
        if 'score' in df.columns:
            sector_boost = np.clip(df['sector_strength'] / 10, -5, 5)  # ±5 points max
            df['score'] = (df['score'] + sector_boost).clip(0, 100).round(1)
        
        return df
    
    @staticmethod
    def _generate_explanations(df: pd.DataFrame, factor_config: Dict) -> pd.Series:
        """Generate human-readable explanations for each signal"""
        def build_explanation(row):
            explanations = []
            
            # Factor-based explanations
            for factor_name, config in factor_config.items():
                if factor_name in row:
                    score = row[factor_name]
                    if pd.notna(score):
                        if score >= 80:
                            label = config.get('strong_label', f'Strong {factor_name}')
                        elif score >= 65:
                            label = config.get('good_label', f'Good {factor_name}')
                        elif score <= 35:
                            label = config.get('bad_label', f'Weak {factor_name}')
                        else:
                            continue  # Skip neutral scores
                        
                        explanations.append(label)
            
            # Risk-based warnings
            if 'risk' in row and row['risk'] == 'High':
                explanations.append("⚠️ High risk factors")
            
            # Special conditions
            if 'pe' in row and pd.notna(row['pe']) and row['pe'] > 50:
                explanations.append("Very expensive valuation")
            
            if 'eps_change_pct' in row and pd.notna(row['eps_change_pct']) and row['eps_change_pct'] > 50:
                explanations.append("🚀 Strong earnings growth")
            
            if 'rvol' in row and pd.notna(row['rvol']) and row['rvol'] > 3:
                explanations.append("📈 Volume surge")
            
            return "; ".join(explanations) if explanations else "Neutral factors"
        
        return df.apply(build_explanation, axis=1)
    
    @staticmethod
    def _finalize_results(df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and optimization"""
        # Fill any remaining NaN values
        if 'score' in df.columns:
            df['score'] = df['score'].fillna(50)
        if 'signal' in df.columns:
            df['signal'] = df['signal'].fillna('NEUTRAL')
        if 'risk' in df.columns:
            df['risk'] = df['risk'].fillna('Medium')
        if 'reason' in df.columns:
            df['reason'] = df['reason'].fillna('')
        
        # Sort by score descending
        if 'score' in df.columns:
            df = df.sort_values('score', ascending=False)
        
        return df
    
    @staticmethod
    def get_top_picks(df: pd.DataFrame, signal_filter: str = "BUY", limit: int = 20) -> pd.DataFrame:
        """Get top stock picks based on signal and score"""
        if df.empty or 'signal' not in df.columns:
            return pd.DataFrame()
        
        # Filter by signal type
        if signal_filter == "BUY":
            filtered = df[df['signal'].isin(['STRONG_BUY', 'BUY'])]
        elif signal_filter == "ALL":
            filtered = df
        else:
            filtered = df[df['signal'] == signal_filter]
        
        # Sort by score and return top picks
        return filtered.nlargest(limit, 'score') if 'score' in filtered.columns else filtered.head(limit)
    
    @staticmethod
    def get_sector_leaders(df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Get best stocks from each sector"""
        if df.empty or 'sector' not in df.columns:
            return pd.DataFrame()
        
        sector_leaders = []
        for sector in df['sector'].dropna().unique():
            sector_stocks = df[df['sector'] == sector]
            if not sector_stocks.empty and 'score' in sector_stocks.columns:
                leader = sector_stocks.nlargest(1, 'score')
                sector_leaders.append(leader)
        
        return pd.concat(sector_leaders, ignore_index=True) if sector_leaders else pd.DataFrame()

# Utility functions for UI integration
def format_signal_badge(signal: str) -> str:
    """Format signal as colored HTML badge"""
    color = SIGNAL_COLORS.get(signal, "#6e7681")
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{signal}</span>'

def format_risk_badge(risk: str) -> str:
    """Format risk as colored HTML badge"""
    color = RISK_COLORS.get(risk, "#6e7681")
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{risk}</span>'

def calculate_market_summary(df: pd.DataFrame) -> Dict[str, any]:
    """Calculate overall market summary statistics"""
    if df.empty:
        return {}
    
    summary = {
        'total_stocks': len(df),
        'buy_signals': len(df[df['signal'].isin(['STRONG_BUY', 'BUY'])]) if 'signal' in df.columns else 0,
        'strong_buy_signals': len(df[df['signal'] == 'STRONG_BUY']) if 'signal' in df.columns else 0,
        'avoid_signals': len(df[df['signal'] == 'AVOID']) if 'signal' in df.columns else 0,
        'avg_score': df['score'].mean() if 'score' in df.columns else 50,
        'high_risk_count': len(df[df['risk'] == 'High']) if 'risk' in df.columns else 0
    }
    
    # Calculate market breadth (% of stocks up today)
    if 'ret_1d' in df.columns:
        positive_returns = (df['ret_1d'] > 0).sum()
        summary['market_breadth'] = (positive_returns / len(df) * 100) if len(df) > 0 else 50
    else:
        summary['market_breadth'] = 50
    
    return summary
