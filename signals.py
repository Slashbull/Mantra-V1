"""
signals.py - Streamlined Signal Engine for M.A.N.T.R.A.
======================================================
Fast, simple, effective signal generation
"""

import pandas as pd
import numpy as np
from config import SIGNAL_THRESHOLDS, WEIGHTS

class SignalEngine:
    """Simplified signal engine - only what matters"""
    
    @staticmethod
    def calculate_signals(df: pd.DataFrame, sector_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate all signals using vectorized operations
        Focus on actionable signals only
        """
        if df.empty:
            return df
        
        # Calculate component scores
        df['momentum'] = SignalEngine._momentum_score(df)
        df['value'] = SignalEngine._value_score(df)
        df['volume_activity'] = SignalEngine._volume_score(df)
        df['technical'] = SignalEngine._technical_score(df)
        
        # Composite score (weighted average)
        df['score'] = (
            df['momentum'] * WEIGHTS['momentum'] +
            df['value'] * WEIGHTS['value'] +
            df['volume_activity'] * WEIGHTS['volume'] +
            df['technical'] * WEIGHTS['technical']
        ).round(0)
        
        # Generate signals
        df['signal'] = SignalEngine._get_signal(df['score'])
        
        # Risk assessment (simplified)
        df['risk'] = SignalEngine._assess_risk(df)
        
        # Add sector strength if available
        if sector_df is not None and not sector_df.empty:
            df = SignalEngine._add_sector_strength(df, sector_df)
        
        # Sort by score
        df = df.sort_values('score', ascending=False)
        
        return df
    
    @staticmethod
    def _momentum_score(df: pd.DataFrame) -> pd.Series:
        """Simple momentum scoring based on returns"""
        score = pd.Series(50.0, index=df.index)
        
        # Weight recent returns more heavily
        weights = {
            'ret_1d': 0.1,
            'ret_7d': 0.2,
            'ret_30d': 0.4,
            'ret_3m': 0.3
        }
        
        momentum = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for col, weight in weights.items():
            if col in df.columns:
                # Normalize returns to 0-100 scale
                ret = df[col].fillna(0)
                normalized = 50 + np.clip(ret / 2, -50, 50)  # ±100% return = 0-100 score
                momentum += normalized * weight
                total_weight += weight
        
        if total_weight > 0:
            score = momentum / total_weight
        
        # Bonus for consistent uptrend
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            uptrend = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            score += uptrend * 10
        
        return score.clip(0, 100).round(0)
    
    @staticmethod
    def _value_score(df: pd.DataFrame) -> pd.Series:
        """Simple value scoring based on PE ratio"""
        score = pd.Series(50.0, index=df.index)
        
        if 'pe' not in df.columns:
            return score
        
        pe = df['pe'].fillna(0)
        
        # Simple PE-based scoring
        conditions = [
            (pe > 0) & (pe <= 15),    # Deep value
            (pe > 15) & (pe <= 25),   # Fair value
            (pe > 25) & (pe <= 40),   # Growth premium
            (pe > 40),                # Expensive
            pe <= 0                   # No earnings
        ]
        
        scores = [90, 70, 50, 30, 20]
        
        score = pd.Series(
            np.select(conditions, scores, default=50),
            index=df.index
        )
        
        # Bonus for profitable companies
        if 'eps' in df.columns:
            profitable = df['eps'] > 0
            score += profitable * 10
        
        return score.clip(0, 100).round(0)
    
    @staticmethod
    def _volume_score(df: pd.DataFrame) -> pd.Series:
        """Simple volume activity scoring"""
        score = pd.Series(50.0, index=df.index)
        
        if 'rvol' not in df.columns:
            return score
        
        rvol = df['rvol'].fillna(1.0)
        
        # Volume-based scoring
        conditions = [
            rvol >= 3.0,    # Extreme volume
            rvol >= 2.0,    # High volume
            rvol >= 1.5,    # Elevated
            rvol >= 0.8,    # Normal
            rvol < 0.8      # Low volume
        ]
        
        scores = [90, 75, 65, 50, 30]
        
        score = pd.Series(
            np.select(conditions, scores, default=50),
            index=df.index
        )
        
        # Bonus for volume spike with positive price action
        if 'ret_1d' in df.columns:
            spike_up = (rvol > 2) & (df['ret_1d'] > 1)
            score += spike_up * 10
        
        return score.clip(0, 100).round(0)
    
    @staticmethod
    def _technical_score(df: pd.DataFrame) -> pd.Series:
        """Simple technical scoring"""
        score = pd.Series(50.0, index=df.index)
        
        # Price above SMA20 (trend following)
        if 'above_sma20' in df.columns:
            score += df['above_sma20'] * 20
        
        # Position in 52-week range
        if 'pos_52w' in df.columns:
            pos = df['pos_52w'].fillna(50)
            # Higher position = stronger
            score += (pos - 50) / 5  # ±10 points for position
        
        return score.clip(0, 100).round(0)
    
    @staticmethod
    def _get_signal(scores: pd.Series) -> pd.Series:
        """Generate clear buy/sell signals"""
        conditions = [
            scores >= SIGNAL_THRESHOLDS['STRONG_BUY'],
            scores >= SIGNAL_THRESHOLDS['BUY'],
            scores >= SIGNAL_THRESHOLDS['WATCH'],
            scores >= SIGNAL_THRESHOLDS['NEUTRAL']
        ]
        
        signals = ['STRONG_BUY', 'BUY', 'WATCH', 'NEUTRAL']
        
        return pd.Series(
            np.select(conditions, signals, default='AVOID'),
            index=scores.index
        )
    
    @staticmethod
    def _assess_risk(df: pd.DataFrame) -> pd.Series:
        """Simple risk assessment"""
        risk_score = pd.Series(0.0, index=df.index)
        
        # High PE = higher risk
        if 'pe' in df.columns:
            risk_score += (df['pe'] > 40) * 30
            risk_score += (df['pe'] <= 0) * 40  # No earnings
        
        # Low volume = higher risk
        if 'volume' in df.columns:
            risk_score += (df['volume'] < 50000) * 20
        
        # Poor momentum = higher risk
        if 'ret_30d' in df.columns:
            risk_score += (df['ret_30d'] < -20) * 30
        
        # Convert to categories
        conditions = [
            risk_score <= 30,
            risk_score <= 60,
            risk_score > 60
        ]
        
        categories = ['Low', 'Medium', 'High']
        
        return pd.Series(
            np.select(conditions, categories, default='Medium'),
            index=df.index
        )
    
    @staticmethod
    def _add_sector_strength(df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Add sector performance data"""
        if 'sector' not in df.columns or 'sector' not in sector_df.columns:
            return df
        
        # Map sector returns
        sector_map = sector_df.set_index('sector')['sector_ret_30d'].to_dict()
        df['sector_strength'] = df['sector'].map(sector_map).fillna(0)
        
        # Adjust score slightly for sector strength
        sector_bonus = df['sector_strength'].clip(-20, 20) / 4  # ±5 points max
        df['score'] = (df['score'] + sector_bonus).clip(0, 100).round(0)
        
        return df
