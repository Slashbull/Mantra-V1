"""
constants.py - M.A.N.T.R.A. Configuration Hub
===========================================
All configuration in one place - simple and clear
"""

# Google Sheets Configuration - UPDATE THIS WITH YOUR SHEET ID!
GOOGLE_SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"

SHEET_CONFIGS = {
    "watchlist": {
        "gid": "2026492216",
        "name": "ALL STOCKS 2025 Watchlist"
    },
    "sector": {
        "gid": "140104095", 
        "name": "ALL STOCKS 2025 Sector Analysis"
    }
}

# Signal Thresholds
SIGNAL_LEVELS = {
    "BUY": 80,
    "WATCH": 65,
    "AVOID": 35
}

# Factor Weights (must sum to 1.0)
FACTOR_WEIGHTS = {
    "momentum": 0.30,
    "value": 0.25,
    "technical": 0.20,
    "volume": 0.15,
    "fundamentals": 0.10
}

# UI Colors
SIGNAL_COLORS = {
    'BUY': '#00d26a',
    'WATCH': '#ffa500',
    'AVOID': '#ff4b4b',
    'NEUTRAL': '#808080'
}

# Risk Levels
RISK_LEVELS = {
    "LOW": (0, 40),
    "MEDIUM": (40, 70),
    "HIGH": (70, 100)
}

# Cache Duration
CACHE_DURATION_MINUTES = 5
