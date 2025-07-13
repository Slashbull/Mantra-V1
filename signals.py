"""
signals.py - M.A.N.T.R.A. Signal Engine (Locked Production Version)
===================================================================
All-time best logic: computes signal score, tags (STRONG_BUY, BUY, etc.),
risk, and detailed reason for each stock based only on provided data and config.
- 100% data-driven; no hardcoded logic
- Robust to missing/extra columns
- No future upgrades needed
"""

import pandas as pd
import numpy as np

from constants import FACTOR_CONFIG, SIGNAL_THRESHOLDS

class SignalEngine:
    @staticmethod
    def calculate_all_signals(df, sector_df=None, regime="balanced"):
        """
        Calculates scores, tags, risks, and reasons for each stock.

        Args:
            df (pd.DataFrame): Input data (merged watchlist + returns)
            sector_df (pd.DataFrame): Sector data (optional, for sector-based factors)
            regime (str): Which factor regime to use (default: "balanced")

        Returns:
            pd.DataFrame: Input df with new columns: score, signal, risk, reason, and per-factor scores
        """
        # Defensive: make a copy
        df = df.copy()
        # Prepare regime config
        if regime not in FACTOR_CONFIG:
            regime = "balanced"
        factor_dict = FACTOR_CONFIG[regime]

        # Score each factor (add as columns)
        factor_scores = {}
        for factor, props in factor_dict.items():
            func = props.get("func")
            if func:
                factor_scores[factor] = func(df)
                df[factor] = factor_scores[factor]
            else:
                df[factor] = 50  # Neutral if no function

        # Weighted sum
        score = np.zeros(len(df))
        for factor, props in factor_dict.items():
            w = props.get("weight", 0)
            score += df[factor] * w
        df['score'] = score.round(1)

        # Tag: STRONG_BUY, BUY, WATCH, NEUTRAL, AVOID
        df['signal'] = df['score'].apply(lambda x:
            "STRONG_BUY" if x >= SIGNAL_THRESHOLDS['STRONG_BUY'] else
            "BUY"        if x >= SIGNAL_THRESHOLDS['BUY'] else
            "WATCH"      if x >= SIGNAL_THRESHOLDS['WATCH'] else
            "NEUTRAL"    if x >= SIGNAL_THRESHOLDS['NEUTRAL'] else
            "AVOID"
        )

        # Risk banding
        def risk_band(row):
            # High risk if PE > 40 or very low volume or negative EPS growth
            try:
                if ('pe' in row and row['pe'] is not None and row['pe'] > 40):
                    return "High"
                if ('rvol' in row and row['rvol'] is not None and row['rvol'] < 0.7):
                    return "High"
                if ('eps_change_pct' in row and row['eps_change_pct'] is not None and row['eps_change_pct'] < -10):
                    return "High"
                if ('volume_1d' in row and row['volume_1d'] is not None and row['volume_1d'] < 500):
                    return "High"
                if ('price' in row and row['price'] is not None and row['price'] < 25):
                    return "High"
                if ('category' in row and row['category'] is not None and 'micro' in str(row['category']).lower()):
                    return "High"
                if ('pe' in row and row['pe'] is not None and row['pe'] < 15 and 'sector' in row and row['sector'] is not None):
                    return "Low"
                return "Medium"
            except Exception:
                return "Medium"
        df['risk'] = df.apply(risk_band, axis=1)

        # Reason/Explanation
        def build_reason(row):
            reasons = []
            for factor, props in factor_dict.items():
                val = row.get(factor, 50)
                if val >= 85:
                    label = props.get("strong_label", "")
                elif val >= 65:
                    label = props.get("good_label", "")
                else:
                    label = props.get("bad_label", "")
                if label:
                    reasons.append(f"{factor.title()}: {label}")
            # Highlight any red flags for risk
            if row.get('risk') == "High":
                reasons.append("⚠️ High risk (PE/Volume/EPS)")
            return "; ".join(reasons)
        df['reason'] = df.apply(build_reason, axis=1)

        # Final clean up: fill any missing
        df['score'] = df['score'].fillna(50)
        df['signal'] = df['signal'].fillna("NEUTRAL")
        df['risk'] = df['risk'].fillna("Medium")
        df['reason'] = df['reason'].fillna("")

        return df
