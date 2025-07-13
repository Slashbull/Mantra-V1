"""
config.py - M.A.N.T.R.A. Configuration
======================================
Simple, clean configuration - only essentials
"""

# ============================================================================
# GOOGLE SHEETS CONFIGURATION
# ============================================================================

# Your Google Sheets ID
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"

# Sheet GIDs (tab IDs)
SHEET_GIDS = {
    "watchlist": "2026492216",
    "sector": "140104095"
}

# ============================================================================
# SIGNAL THRESHOLDS
# ============================================================================

SIGNAL_THRESHOLDS = {
    "STRONG_BUY": 85,
    "BUY": 75,
    "WATCH": 60,
    "NEUTRAL": 40
}

# ============================================================================
# SCORING WEIGHTS
# ============================================================================

WEIGHTS = {
    "momentum": 0.40,    # 40% - Price trends matter most
    "value": 0.25,       # 25% - Valuation is important
    "volume": 0.20,      # 20% - Volume confirms moves
    "technical": 0.15    # 15% - Technical position
}

# ============================================================================
# END
# ============================================================================
