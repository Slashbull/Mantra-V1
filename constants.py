"""
constants.py - M.A.N.T.R.A. Central Configuration & Documentation
=================================================================
Final, production-ready. All constants, settings, columns, and docs
Live here. Change here, affect everywhere.

- Google Sheets config (ID, GIDs, URL templates)
- Signal/score thresholds
- All column headers (for loader, validation, docs)
- Regime/factor weights
- UI settings (colors, emojis)
- Data requirements
- System docstring/schema for reference

"""

# ============================================================================
# GOOGLE SHEETS CONFIGURATION
# ============================================================================

GOOGLE_SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
SHEET_GIDS = {
    "watchlist": "2026492216",
    "returns": "100734077",
    "sector": "140104095"
}
SHEET_URLS = {
    name: f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={gid}"
    for name, gid in SHEET_GIDS.items()
}

# ============================================================================
# SIGNAL THRESHOLDS (Used for tagging signals)
# ============================================================================
SIGNAL_THRESHOLDS = {
    "STRONG_BUY": 85,
    "BUY": 75,
    "WATCH": 60,
    "NEUTRAL": 40
}

# ============================================================================
# FACTOR WEIGHTS (Can be made regime-switchable)
# ============================================================================
FACTOR_WEIGHTS = {
    "momentum": 0.40,
    "value": 0.25,
    "volume": 0.20,
    "technical": 0.15,
}

# ============================================================================
# COLUMN HEADERS — Normalized, Used Everywhere
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
# UI DISPLAY (Colors, Emojis, etc.)
# ============================================================================
SIGNAL_COLORS = {
    "STRONG_BUY": "#2ea043",
    "BUY": "#3fb950",
    "WATCH": "#d29922",
    "NEUTRAL": "#6e7681",
    "AVOID": "#da3633",
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
# SYSTEM DOCSTRING: ALL COLUMNS WITH MEANING (For Code, ETL, Docs)
# ============================================================================
SCHEMA_DOC = """
1. Watchlist Sheet Columns
    - ticker: Unique stock symbol (e.g., TCS, RELIANCE)
    - exchange: NSE: or BSE: prefix
    - company_name: Full company name
    - year: Year of founding/incorporation
    - market_cap: Market capitalization (₹, can include 'Cr', 'Lakh', etc.)
    - category: Cap category (Small Cap, Mid Cap, etc.)
    - sector: Sector name
    - eps_tier: EPS bucket label (custom, e.g., 5↓, 5↑, etc.)
    - price: Current stock price
    - ret_1d: 1-day return (%)
    - low_52w/high_52w: 52-week low/high
    - from_low_pct: % gain from 52W low
    - from_high_pct: % drop from 52W high
    - sma_20d/50d/200d: 20/50/200-day Simple Moving Averages
    - trading_under: Label if price under any DMA
    - ret_3d/7d/30d/3m/6m/1y/3y/5y: Returns over various periods (%)
    - volume_1d/7d/30d/3m: Volume stats (raw, 7/30/90d avg)
    - vol_ratio_*: Volume ratios
    - rvol: Relative volume indicator
    - price_tier: Price bucket
    - prev_close: Previous day's close
    - pe: Price-to-earnings ratio
    - eps_current/last_qtr/duplicate: EPS stats
    - eps_change_pct: % EPS change from last qtr
    - [score columns will be added later]

2. Returns (Industry) Sheet Columns
    - ticker, company_name, returns_ret_1d ... avg_ret_5y: As above for raw returns & averages

3. Sector Sheet Columns
    - sector, sector_ret_1d ... sector_count: Sector stats for momentum/rotation
"""

# ============================================================================
# END OF CONSTANTS
# ============================================================================

