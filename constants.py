"""
constants.py - M.A.N.T.R.A. Central Configuration (Locked Version)
==================================================================
Holds all configuration, scoring logic, thresholds, factor functions,
Google Sheets URLs, column headers, regime definitions, UI color settings,
and schema documentation.

- This file controls all logic, scoring, and UI settings for the app.
- Change here = system-wide effect. Never edit other files for business logic.
"""

import pandas as pd
import numpy as np

# ============================================================================
# GOOGLE SHEETS CONFIGURATION
# ============================================================================
GOOGLE_SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
SHEET_GIDS = {
    "watchlist": "2026492216",
    "returns":   "100734077",
    "sector":    "140104095"
}
SHEET_URLS = {
    name: f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={gid}"
    for name, gid in SHEET_GIDS.items()
}

# ============================================================================
# REQUIRED COLUMN HEADERS (for validation and cleaning)
# ============================================================================
WATCHLIST_COLUMNS = [
    'ticker', 'exchange', 'company_name', 'year', 'market_cap', 'category',
    'sector', 'eps_tier', 'price', 'ret_1d', 'low_52w', 'high_52w', 'from_low_pct',
    'from_high_pct', 'sma_20d', 'sma_50d', 'sma_200d', 'trading_under', 'ret_3d',
    'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
    'volume_1d', 'volume_7d', 'volume_30d', 'volume_3m', 'vol_ratio_1d_90d',
    'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'rvol', 'price_tier', 'prev_close',
    'pe', 'eps_current', 'eps_last_qtr', 'eps_duplicate', 'eps_change_pct'
]
RETURNS_COLUMNS = [
    'ticker', 'company_name', 'returns_ret_1d', 'returns_ret_3d', 'returns_ret_7d',
    'returns_ret_30d', 'returns_ret_3m', 'returns_ret_6m', 'returns_ret_1y',
    'returns_ret_3y', 'returns_ret_5y', 'avg_ret_30d', 'avg_ret_3m', 'avg_ret_6m',
    'avg_ret_1y', 'avg_ret_3y', 'avg_ret_5y'
]
SECTOR_COLUMNS = [
    'sector', 'sector_ret_1d', 'sector_ret_3d', 'sector_ret_7d', 'sector_ret_30d',
    'sector_ret_3m', 'sector_ret_6m', 'sector_ret_1y', 'sector_ret_3y',
    'sector_ret_5y', 'sector_avg_30d', 'sector_avg_3m', 'sector_avg_6m',
    'sector_avg_1y', 'sector_avg_3y', 'sector_avg_5y', 'sector_count'
]

# ============================================================================
# SIGNAL SCORE THRESHOLDS (locked)
# ============================================================================
SIGNAL_THRESHOLDS = {
    "STRONG_BUY": 85,
    "BUY": 75,
    "WATCH": 60,
    "NEUTRAL": 40
}

# ============================================================================
# COLOR AND UI SETTINGS (for badges and table styling)
# ============================================================================
SIGNAL_COLORS = {
    "STRONG_BUY": "#2ea043",
    "BUY": "#3fb950",
    "WATCH": "#d29922",
    "NEUTRAL": "#6e7681",
    "AVOID": "#da3633"
}
RISK_COLORS = {
    "Low": "#3fb950",
    "Medium": "#d29922",
    "High": "#da3633"
}
SECTOR_COLORS = [
    "#3fb950", "#58a6ff", "#d29922", "#da3633", "#6e7681",
    "#1a1f2e", "#8b949e", "#0e1117"
]

# ============================================================================
# SMART FACTOR FUNCTIONS — Data-aware, context-driven scoring logic
# ============================================================================

def score_momentum(df):
    """
    Smart momentum: Considers recent returns, trend smoothness, and 52w position.
    Rewards strong, consistent uptrend; penalizes reversals or mean-reversion risk.
    """
    weights = {'ret_1d': 0.1, 'ret_7d': 0.2, 'ret_30d': 0.4, 'ret_3m': 0.3}
    score = pd.Series(0.0, index=df.index)
    total_weight = sum(weights.values())
    for col, w in weights.items():
        if col in df.columns:
            ret = df[col].fillna(0)
            norm = 50 + np.clip(ret / 2, -50, 50)
            score += norm * w
    score = score / total_weight if total_weight > 0 else 50
    # Bonus: uptrend consistency
    if all(c in df.columns for c in ['ret_7d', 'ret_30d']):
        score += ((df['ret_7d'] > 0) & (df['ret_30d'] > 0)) * 10
    # Bonus: new high (near high_52w)
    if all(c in df.columns for c in ['price', 'high_52w']):
        score += (np.abs(df['price'] - df['high_52w']) < 0.01 * df['high_52w']) * 5
    # Penalty: mean reversion risk (1d drop after 30d up)
    if all(c in df.columns for c in ['ret_1d', 'ret_30d']):
        score -= ((df['ret_1d'] < -1) & (df['ret_30d'] > 10)) * 10
    return score.clip(0, 100).round(0)

def score_value(df):
    """
    Smart value: Uses PE/EPS context; avoids value traps and rewards profitable growth.
    """
    score = pd.Series(50.0, index=df.index)
    if 'pe' not in df.columns:
        return score
    pe = df['pe'].fillna(0)
    conditions = [
        (pe > 0) & (pe <= 15),          # Deep value
        (pe > 15) & (pe <= 25),         # Reasonable
        (pe > 25) & (pe <= 40),         # Getting expensive
        (pe > 40),                      # Very expensive
        (pe <= 0),                      # Loss-making
    ]
    scores = [90, 70, 50, 30, 20]
    score = np.select(conditions, scores, default=50)
    # Bonus: positive EPS growth
    if 'eps_change_pct' in df.columns:
        score += (df['eps_change_pct'] > 10) * 10
    # Penalty: negative EPS or falling EPS
    if 'eps_change_pct' in df.columns:
        score -= (df['eps_change_pct'] < -5) * 10
    return pd.Series(score, index=df.index).clip(0, 100).round(0)

def score_volume(df):
    """
    Smart volume: rvol plus context of price and volume spikes.
    Only rewards up-moves with strong volume.
    """
    score = pd.Series(50.0, index=df.index)
    if 'rvol' not in df.columns:
        return score
    rvol = df['rvol'].fillna(1.0)
    conditions = [
        rvol >= 3.0, rvol >= 2.0, rvol >= 1.5, rvol >= 0.8, rvol < 0.8
    ]
    scores = [90, 75, 65, 50, 30]
    score = np.select(conditions, scores, default=50)
    # Only reward volume spike if price is up
    if 'ret_1d' in df.columns:
        score += ((rvol > 2) & (df['ret_1d'] > 1)) * 10
        score -= ((rvol > 2) & (df['ret_1d'] < -1)) * 10
    # Bonus: strong multi-day spike
    if 'volume_spike' in df.columns:
        score += (df['volume_spike'] > 0) * 5
    return pd.Series(score, index=df.index).clip(0, 100).round(0)

def score_technical(df):
    """
    Smart technicals: Uses price vs SMA, 52w position, and breakout status.
    """
    score = pd.Series(50.0, index=df.index)
    if 'above_sma20' in df.columns:
        score += df['above_sma20'] * 20
    if 'position_52w' in df.columns:
        pos = df['position_52w'].fillna(50)
        score += (pos - 50) / 5
        # Bonus: breakout (position above 95)
        score += (pos > 95) * 5
    # Penalty: trading under 20/50/200 SMA
    if 'trading_under' in df.columns:
        score -= (df['trading_under'].notnull() & (df['trading_under'] != '')) * 5
    return score.clip(0, 100).round(0)

# ============================================================================
# FACTOR CONFIG — Dict for all factors, weights, and smart labels
# ============================================================================
FACTOR_CONFIG = {
    "balanced": {
        "momentum": {
            "weight": 0.4,
            "func": score_momentum,
            "strong_label": "Strong uptrend",
            "good_label": "Positive momentum",
            "bad_label": "Weak/negative trend"
        },
        "value": {
            "weight": 0.20,
            "func": score_value,
            "strong_label": "Value + EPS growth",
            "good_label": "Fair value",
            "bad_label": "Expensive/loss"
        },
        "volume": {
            "weight": 0.20,
            "func": score_volume,
            "strong_label": "Volume surge on gains",
            "good_label": "Healthy volume",
            "bad_label": "Low/negative volume"
        },
        "technical": {
            "weight": 0.20,
            "func": score_technical,
            "strong_label": "Breakout/highs",
            "good_label": "Above trend",
            "bad_label": "Below trend"
        }
        # Add more factors (e.g., anomaly, consistency) if needed
    }
    # Add more regimes (momentum/value/etc) as needed
}

# ============================================================================
# SYSTEM DOCSTRING FOR REFERENCE (attach to UI/docs as needed)
# ============================================================================
SCHEMA_DOC = """
1. Watchlist Sheet Columns
    - ticker: Unique stock symbol (e.g., TCS, RELIANCE)
    - exchange: NSE: or BSE: prefix
    - company_name: Full company name
    - year: Year of founding/incorporation
    - market_cap: Market capitalization (₹, 'Cr', 'Lakh', etc.)
    - category: Cap category
    - sector: Sector name
    - eps_tier: EPS bucket label
    - price: Current stock price
    - ret_1d: 1-day return (%)
    - ...etc. (see full spec for all columns)

2. Returns (Industry) Sheet Columns
    - ticker, company_name, returns_ret_1d ... avg_ret_5y: As above for returns & averages

3. Sector Sheet Columns
    - sector, sector_ret_1d ... sector_count: Sector stats for momentum/rotation
"""

# ============================================================================
# END OF CONSTANTS (Locked)
# ============================================================================
