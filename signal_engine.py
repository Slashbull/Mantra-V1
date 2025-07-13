"""
signal_engine.py - M.A.N.T.R.A. Signal Generation Engine
======================================================
Multi-factor scoring perfectly aligned with your data columns
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Optional, Tuple
import logging
from constants import (
    FACTOR_WEIGHTS, SIGNAL_LEVELS, MOMENTUM_THRESHOLDS,
    VOLUME_THRESHOLDS, PE_RANGES, EPS_GROWTH_RANGES,
    POSITION_52W_RANGES, SMA_DISTANCE_THRESHOLDS,
    EPS_TIERS, PRICE_TIERS, RISK_FACTORS
)

logger = logging.getLogger(__name__)

class SignalEngine:
    """Signal generation using your exact data structure"""
    
    @staticmethod
    @st.cache_data(ttl=60, show_spinner=False)
    def calculate_all_signals(stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all signals and scores for stocks
        Returns enhanced dataframe with scores and decisions
        """
        if stocks_df.empty:
            return stocks_df
        
        df = stocks_df.copy()
        
        # Calculate individual factor scores
        df['momentum_score'] = SignalEngine._calculate_momentum_score(df)
        df['value_score'] = SignalEngine._calculate_value_score(df)
        df['technical_score'] = SignalEngine._calculate_technical_score(df)
        df['volume_score'] = SignalEngine._calculate_volume_score(df)
        df['fundamental_score'] = SignalEngine._calculate_fundamental_score(df)
        
        # Add sector strength
        df = SignalEngine._add_sector_strength(df, sector_df)
        
        # Calculate composite score
        df['composite_score'] = SignalEngine._calculate_composite_score(df)
        
        # Make decisions
        df['decision'] = df['composite_score'].apply(SignalEngine._get_decision)
        
        # Calculate risk
        df['risk_score'] = SignalEngine._calculate_risk_score(df)
        df['risk_level'] = df['risk_score'].apply(SignalEngine._get_risk_level)
        
        # Calculate opportunity score
        df['opportunity_score'] = SignalEngine._calculate_opportunity_score(df)
        
        # Generate reasoning
        df['reasoning'] = df.apply(SignalEngine._generate_reasoning, axis=1)
        
        # Add targets and stop loss
        df = SignalEngine._calculate_targets(df)
        
        # Rank stocks
        df['rank'] = df['composite_score'].rank(ascending=False, method='min')
        df['percentile'] = (df['composite_score'].rank(pct=True) * 100).round(1)
        
        return df
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score using all return columns"""
        score = pd.Series(50.0, index=df.index)
        
        # Use all available return periods with appropriate weights
        return_weights = {
            'ret_1d': 0.05,
            'ret_3d': 0.05,
            'ret_7d': 0.10,
            'ret_30d': 0.20,
            'ret_3m': 0.30,
            'ret_6m': 0.20,
            'ret_1y': 0.10
        }
        
        total_weight = 0
        momentum_components = pd.Series(0.0, index=df.index)
        
        for period, weight in return_weights.items():
            if period in df.columns:
                returns = df[period].fillna(0)
                
                # Map period to threshold
                period_map = {
                    'ret_1d': '1d',
                    'ret_3d': '3d',
                    'ret_7d': '7d',
                    'ret_30d': '30d',
                    'ret_3m': '3m',
                    'ret_6m': '6m',
                    'ret_1y': '1y'
                }
                
                threshold_key = period_map.get(period, '30d')
                threshold = MOMENTUM_THRESHOLDS['STRONG'].get(threshold_key, 10)
                
                # Score based on return strength
                period_score = 50 + (returns / threshold) * 25
                period_score = period_score.clip(0, 100)
                
                momentum_components += period_score * weight
                total_weight += weight
        
        if total_weight > 0:
            score = momentum_components / total_weight
        
        # Bonus for trend consistency (using average returns if available)
        if 'avg_ret_30d' in df.columns:
            trend_bonus = pd.Series(0.0, index=df.index)
            trend_bonus[df['avg_ret_30d'] > 10] = 10
            trend_bonus[df['avg_ret_30d'] > 5] = 5
            score = (score + trend_bonus).clip(0, 100)
        
        return score.round(1)
    
    @staticmethod
    def _calculate_value_score(df: pd.DataFrame) -> pd.Series:
        """Calculate value score using PE and EPS data"""
        score = pd.Series(50.0, index=df.index)
        
        # PE Ratio Score (50% weight)
        if 'pe' in df.columns:
            pe = df['pe'].fillna(df['pe'].median())
            pe_score = pd.Series(50.0, index=df.index)
            
            # Score based on PE ranges
            for (min_pe, max_pe), score_val in [
                (PE_RANGES['DEEP_VALUE'], 95),
                (PE_RANGES['VALUE'], 85),
                (PE_RANGES['FAIR'], 70),
                (PE_RANGES['GROWTH'], 50),
                (PE_RANGES['EXPENSIVE'], 30),
                (PE_RANGES['BUBBLE'], 20)
            ]:
                mask = (pe >= min_pe) & (pe < max_pe)
                pe_score[mask] = score_val
            
            # Negative PE gets low score
            pe_score[pe <= 0] = 20
            
            score = pe_score * 0.5
        
        # EPS Growth Score (30% weight)
        if 'eps_change_pct' in df.columns:
            eps_growth = df['eps_change_pct'].fillna(0)
            growth_score = pd.Series(50.0, index=df.index)
            
            growth_score[eps_growth > EPS_GROWTH_RANGES['HYPER']] = 95
            growth_score[eps_growth > EPS_GROWTH_RANGES['HIGH']] = 80
            growth_score[eps_growth > EPS_GROWTH_RANGES['MODERATE']] = 65
            growth_score[eps_growth > EPS_GROWTH_RANGES['LOW']] = 50
            growth_score[eps_growth > EPS_GROWTH_RANGES['NEGATIVE']] = 35
            growth_score[eps_growth <= EPS_GROWTH_RANGES['DECLINING']] = 20
            
            score += growth_score * 0.3
        
        # EPS Tier Bonus (20% weight)
        if 'eps_tier' in df.columns:
            tier_score = pd.Series(50.0, index=df.index)
            
            for tier, config in EPS_TIERS.items():
                mask = df['eps_tier'] == tier
                tier_score[mask] = 50 + config['score_boost']
            
            score += tier_score * 0.2
        
        return score.clip(0, 100).round(1)
    
    @staticmethod
    def _calculate_technical_score(df: pd.DataFrame) -> pd.Series:
        """Calculate technical score using price position and SMAs"""
        score = pd.Series(50.0, index=df.index)
        
        # Price vs SMAs (40% weight)
        sma_score = pd.Series(50.0, index=df.index)
        sma_count = 0
        
        if 'price' in df.columns:
            # Check each SMA
            for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
                if sma in df.columns:
                    above_sma = df['price'] > df[sma]
                    sma_score += above_sma * 10
                    sma_count += 1
            
            # Penalty if trading_under any average
            if 'trading_under' in df.columns:
                has_trading_under = df['trading_under'].notna() & (df['trading_under'] != '')
                sma_score[has_trading_under] -= 20
        
        if sma_count > 0:
            score = sma_score * 0.4
        
        # 52-week position (30% weight)
        if 'position_52w' in df.columns:
            pos_score = pd.Series(50.0, index=df.index)
            
            for (min_pos, max_pos), score_val in [
                (POSITION_52W_RANGES['NEAR_HIGH'], 75),
                (POSITION_52W_RANGES['UPPER'], 65),
                (POSITION_52W_RANGES['MIDDLE_HIGH'], 55),
                (POSITION_52W_RANGES['MIDDLE_LOW'], 50),
                (POSITION_52W_RANGES['LOWER'], 40),
                (POSITION_52W_RANGES['NEAR_LOW'], 35)
            ]:
                mask = (df['position_52w'] >= min_pos) & (df['position_52w'] < max_pos)
                pos_score[mask] = score_val
            
            score += pos_score * 0.3
        
        # From high/low percentages (30% weight)
        if 'from_high_pct' in df.columns and 'from_low_pct' in df.columns:
            position_score = pd.Series(50.0, index=df.index)
            
            # Good: Far from high, close to middle
            optimal_from_high = df['from_high_pct'] > -30  # Not too far from high
            optimal_from_low = df['from_low_pct'] > 30     # Well above low
            
            position_score[optimal_from_high & optimal_from_low] = 70
            position_score[df['from_high_pct'] > -10] = 60  # Near highs
            position_score[df['from_low_pct'] < 20] = 40    # Near lows
            
            score += position_score * 0.3
        
        return score.clip(0, 100).round(1)
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate volume score using rvol and volume ratios"""
        score = pd.Series(50.0, index=df.index)
        
        # Relative volume (rvol) - 50% weight
        if 'rvol' in df.columns:
            rvol = df['rvol'].fillna(1.0)
            rvol_score = pd.Series(50.0, index=df.index)
            
            rvol_score[rvol >= VOLUME_THRESHOLDS['SPIKE']] = 90
            rvol_score[rvol >= VOLUME_THRESHOLDS['HIGH']] = 75
            rvol_score[rvol >= VOLUME_THRESHOLDS['ELEVATED']] = 65
            rvol_score[rvol >= VOLUME_THRESHOLDS['NORMAL']] = 50
            rvol_score[rvol >= VOLUME_THRESHOLDS['LOW']] = 35
            rvol_score[rvol < VOLUME_THRESHOLDS['LOW']] = 20
            
            score = rvol_score * 0.5
        
        # Volume ratios - 50% weight
        ratio_score = pd.Series(50.0, index=df.index)
        ratio_count = 0
        
        for ratio_col, weight in [
            ('vol_ratio_1d_90d', 0.4),
            ('vol_ratio_7d_90d', 0.3),
            ('vol_ratio_30d_90d', 0.3)
        ]:
            if ratio_col in df.columns:
                ratio_val = df[ratio_col].fillna(100)
                
                col_score = pd.Series(50.0, index=df.index)
                col_score[ratio_val >= VOLUME_RATIO_THRESHOLDS['SURGE']] = 85
                col_score[ratio_val >= VOLUME_RATIO_THRESHOLDS['INCREASING']] = 70
                col_score[ratio_val >= VOLUME_RATIO_THRESHOLDS['DECREASING']] = 50
                col_score[ratio_val < VOLUME_RATIO_THRESHOLDS['DRYING']] = 30
                
                ratio_score += col_score * weight
                ratio_count += weight
        
        if ratio_count > 0:
            score += (ratio_score - 50) * 0.5
        
        # Bonus for volume spike with positive price action
        if 'rvol' in df.columns and 'ret_1d' in df.columns:
            spike_bonus = ((df['rvol'] > 2) & (df['ret_1d'] > 1)) * 10
            score = (score + spike_bonus).clip(0, 100)
        
        return score.round(1)
    
    @staticmethod
    def _calculate_fundamental_score(df: pd.DataFrame) -> pd.Series:
        """Calculate fundamental score using quality metrics"""
        score = pd.Series(50.0, index=df.index)
        components = 0
        
        # EPS trend (25% weight)
        if 'eps_change_pct' in df.columns:
            eps_score = pd.Series(50.0, index=df.index)
            eps_score[df['eps_change_pct'] > 30] = 80
            eps_score[df['eps_change_pct'] > 10] = 65
            eps_score[df['eps_change_pct'] > 0] = 55
            eps_score[df['eps_change_pct'] < -20] = 30
            
            score += eps_score * 0.25
            components += 0.25
        
        # Market cap quality (25% weight)
        if 'market_cap' in df.columns:
            mcap_score = pd.Series(50.0, index=df.index)
            mcap_score[df['market_cap'] > 1e12] = 75    # Mega cap
            mcap_score[df['market_cap'] > 2e11] = 70    # Large cap
            mcap_score[df['market_cap'] > 5e10] = 60    # Mid cap
            mcap_score[df['market_cap'] > 5e9] = 50     # Small cap
            mcap_score[df['market_cap'] <= 5e9] = 40    # Micro cap
            
            score += mcap_score * 0.25
            components += 0.25
        
        # Category quality (25% weight)
        if 'category' in df.columns:
            cat_score = pd.Series(50.0, index=df.index)
            cat_score[df['category'] == 'Large Cap'] = 70
            cat_score[df['category'] == 'Mid Cap'] = 60
            cat_score[df['category'] == 'Small Cap'] = 50
            cat_score[df['category'] == 'Micro Cap'] = 40
            
            score += cat_score * 0.25
            components += 0.25
        
        # Profitability (25% weight)
        if 'pe' in df.columns:
            profit_score = pd.Series(50.0, index=df.index)
            profit_score[(df['pe'] > 0) & (df['pe'] < 30)] = 75
            profit_score[(df['pe'] >= 30) & (df['pe'] < 50)] = 60
            profit_score[df['pe'] <= 0] = 30  # Loss making
            
            score += profit_score * 0.25
            components += 0.25
        
        # Normalize if not all components present
        if components > 0 and components < 1:
            score = 50 + (score - 50) / components
        
        return score.clip(0, 100).round(1)
    
    @staticmethod
    def _add_sector_strength(df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Add sector performance to individual stocks"""
        if sector_df.empty or 'sector' not in df.columns:
            df['sector_score'] = 50.0
            return df
        
        # Use 30-day sector performance as the main metric
        if 'sector_ret_30d' in sector_df.columns:
            sector_perf = sector_df.set_index('sector')['sector_ret_30d'].to_dict()
            df['sector_performance'] = df['sector'].map(sector_perf).fillna(0)
            
            # Convert to score (normalize)
            if df['sector_performance'].std() > 0:
                df['sector_score'] = (
                    50 + (df['sector_performance'] - df['sector_performance'].mean()) / 
                    df['sector_performance'].std() * 20
                ).clip(0, 100)
            else:
                df['sector_score'] = 50.0
        
        # Add sector momentum if available
        if 'sector_avg_3m' in sector_df.columns:
            sector_momentum = sector_df.set_index('sector')['sector_avg_3m'].to_dict()
            df['sector_momentum'] = df['sector'].map(sector_momentum).fillna(0)
        
        return df
    
    @staticmethod
    def _calculate_composite_score(df: pd.DataFrame) -> pd.Series:
        """Calculate weighted composite score"""
        score_columns = {
            'momentum': 'momentum_score',
            'value': 'value_score',
            'technical': 'technical_score',
            'volume': 'volume_score',
            'fundamentals': 'fundamental_score'
        }
        
        composite = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for factor, col in score_columns.items():
            if col in df.columns and factor in FACTOR_WEIGHTS:
                weight = FACTOR_WEIGHTS[factor]
                composite += df[col].fillna(50) * weight
                total_weight += weight
        
        # Add sector bonus (not part of main weights)
        if 'sector_score' in df.columns:
            sector_bonus = (df['sector_score'] - 50) * 0.1  # Max ±5 points
            composite += sector_bonus
        
        # Normalize
        if total_weight > 0:
            composite = composite / total_weight
        
        return composite.clip(0, 100).round(1)
    
    @staticmethod
    def _get_decision(score: float) -> str:
        """Convert score to trading decision"""
        if score >= SIGNAL_LEVELS['STRONG_BUY']:
            return 'STRONG_BUY'
        elif score >= SIGNAL_LEVELS['BUY']:
            return 'BUY'
        elif score >= SIGNAL_LEVELS['WATCH']:
            return 'WATCH'
        elif score >= SIGNAL_LEVELS['NEUTRAL']:
            return 'NEUTRAL'
        else:
            return 'AVOID'
    
    @staticmethod
    def _calculate_risk_score(df: pd.DataFrame) -> pd.Series:
        """Calculate risk score (0-100, higher is riskier)"""
        risk = pd.Series(0.0, index=df.index)
        
        # Volatility risk (using return std)
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            returns_std = df[['ret_1d', 'ret_7d', 'ret_30d']].std(axis=1)
            volatility_risk = (returns_std / 10 * 20).clip(0, 20)
            risk += volatility_risk
        
        # PE risk
        if 'pe' in df.columns:
            pe_risk = pd.Series(0.0, index=df.index)
            pe_risk[df['pe'] > 40] = RISK_FACTORS['high_pe']
            pe_risk[df['pe'] <= 0] = RISK_FACTORS['no_profit']
            risk += pe_risk
        
        # Volume risk
        if 'volume_1d' in df.columns:
            volume_risk = pd.Series(0.0, index=df.index)
            volume_risk[df['volume_1d'] < 50000] = RISK_FACTORS['low_volume']
            risk += volume_risk
        
        # Price risk
        if 'price' in df.columns:
            price_risk = pd.Series(0.0, index=df.index)
            price_risk[df['price'] < 50] = RISK_FACTORS['penny_stock']
            risk += price_risk
        
        # Position risk
        if 'position_52w' in df.columns:
            position_risk = pd.Series(0.0, index=df.index)
            position_risk[df['position_52w'] < 10] = RISK_FACTORS['near_52w_low']
            risk += position_risk
        
        # Negative EPS
        if 'eps_current' in df.columns:
            eps_risk = pd.Series(0.0, index=df.index)
            eps_risk[df['eps_current'] < 0] = RISK_FACTORS['negative_eps']
            risk += eps_risk
        
        return risk.clip(0, 100).round(1)
    
    @staticmethod
    def _get_risk_level(risk_score: float) -> str:
        """Convert risk score to risk level"""
        for level, (min_risk, max_risk) in RISK_LEVELS.items():
            if min_risk <= risk_score < max_risk:
                return level.replace('_', ' ').title()
        return 'Unknown'
    
    @staticmethod
    def _calculate_opportunity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate opportunity score (upside potential)"""
        score = df.get('composite_score', 50)
        risk = df.get('risk_score', 50)
        momentum = df.get('momentum_score', 50)
        
        # High score + low risk + good momentum = high opportunity
        opportunity = (score * 0.5 + (100 - risk) * 0.3 + momentum * 0.2)
        
        # Bonus for value plays
        if 'value_score' in df.columns:
            value_bonus = (df['value_score'] > 80) * 5
            opportunity += value_bonus
        
        return opportunity.clip(0, 100).round(1)
    
    @staticmethod
    def _calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate target prices and stop losses"""
        df['target_1'] = np.nan
        df['target_2'] = np.nan
        df['stop_loss'] = np.nan
        
        if 'price' not in df.columns:
            return df
        
        # Calculate based on volatility and score
        for idx, row in df.iterrows():
            price = row['price']
            if pd.isna(price) or price <= 0:
                continue
            
            score = row.get('composite_score', 50)
            risk_level = row.get('risk_level', 'Medium')
            
            # Target calculation
            if score >= 80:
                target_pct_1 = 10
                target_pct_2 = 20
            elif score >= 70:
                target_pct_1 = 8
                target_pct_2 = 15
            else:
                target_pct_1 = 5
                target_pct_2 = 10
            
            # Adjust for volatility
            if 'ret_30d' in row and abs(row['ret_30d']) > 20:
                target_pct_1 *= 1.5
                target_pct_2 *= 1.5
            
            df.at[idx, 'target_1'] = round(price * (1 + target_pct_1/100), 2)
            df.at[idx, 'target_2'] = round(price * (1 + target_pct_2/100), 2)
            
            # Stop loss based on risk
            if risk_level in ['Very Low', 'Low']:
                stop_pct = 5
            elif risk_level == 'Medium':
                stop_pct = 7
            else:
                stop_pct = 10
            
            df.at[idx, 'stop_loss'] = round(price * (1 - stop_pct/100), 2)
        
        return df
    
    @staticmethod
    def _generate_reasoning(row: pd.Series) -> str:
        """Generate human-readable reasoning for decision"""
        reasons = []
        
        # Score-based reasoning
        score = row.get('composite_score', 50)
        if score >= 85:
            reasons.append("Excellent overall score")
        elif score >= 75:
            reasons.append("Strong fundamentals")
        elif score < 40:
            reasons.append("Weak indicators")
        
        # Factor-specific reasons
        if row.get('momentum_score', 50) > 80:
            reasons.append("Strong momentum")
        elif row.get('momentum_score', 50) < 30:
            reasons.append("Poor momentum")
        
        if row.get('value_score', 50) > 80:
            reasons.append("Attractive valuation")
        
        if row.get('volume_score', 50) > 80:
            reasons.append("High volume activity")
        
        # Special conditions
        if row.get('rvol', 1) > 3:
            reasons.append("Volume spike")
        
        if row.get('position_52w', 50) > 85:
            reasons.append("Near 52W high")
        elif row.get('position_52w', 50) < 15:
            reasons.append("Near 52W low")
        
        # Risk warning
        risk_level = row.get('risk_level', 'Medium')
        if risk_level in ['High', 'Very High']:
            reasons.append(f"{risk_level} risk")
        
        return " | ".join(reasons[:3]) if reasons else "Mixed signals"
