"""
constants.py - M.A.N.T.R.A. Configuration Hub
===========================================
FINAL VERSION - All configuration for your exact data structure
"""

# ============================================================================
# GOOGLE SHEETS CONFIGURATION
# ============================================================================

# YOUR Google Sheets ID - THIS IS THE ONLY THING YOU NEED TO UPDATE!
GOOGLE_SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"

# Sheet GIDs (tab IDs) - matching your exact sheets
SHEET_CONFIGS = {
    "watchlist": {
        "gid": "2026492216",
        "name": "ALL STOCKS 2025 Watchlist",
        "description": "Master stock universe with all attributes"
    },
    "returns": {
        "gid": "100734077",
        "name": "Stock Return Analysis",
        "description": "Raw returns and averages for every stock"
    },
    "sector": {
        "gid": "140104095", 
        "name": "ALL STOCKS 2025 Sector Analysis",
        "description": "Sector-level aggregate data"
    }
}

# ============================================================================
# SIGNAL THRESHOLDS
# ============================================================================

# Composite score thresholds for Buy/Watch/Avoid signals
SIGNAL_LEVELS = {
    "STRONG_BUY": 85,   # Score >= 85 (top 5% opportunities)
    "BUY": 75,          # Score >= 75 (clear buy signal)
    "WATCH": 60,        # Score >= 60 (potential opportunity)
    "NEUTRAL": 40,      # Score >= 40 (no clear signal)
    "AVOID": 20         # Score < 20 (stay away)
}

# ============================================================================
# FACTOR WEIGHTS (must sum to 1.0)
# ============================================================================

FACTOR_WEIGHTS = {
    "momentum": 0.30,      # 30% - Price performance (ret_1d to ret_1y)
    "value": 0.25,         # 25% - Valuation (PE, EPS growth)
    "technical": 0.20,     # 20% - Technical position (SMAs, 52w range)
    "volume": 0.15,        # 15% - Volume activity (rvol, volume ratios)
    "fundamentals": 0.10   # 10% - Quality (EPS tier, market cap)
}

# ============================================================================
# MOMENTUM THRESHOLDS
# ============================================================================

MOMENTUM_THRESHOLDS = {
    "STRONG": {
        "1d": 3.0,      # >3% daily
        "3d": 5.0,      # >5% in 3 days
        "7d": 7.0,      # >7% weekly
        "30d": 15.0,    # >15% monthly
        "3m": 25.0,     # >25% quarterly
        "6m": 40.0,     # >40% half-yearly
        "1y": 60.0      # >60% yearly
    },
    "MODERATE": {
        "1d": 1.0,
        "3d": 2.0,
        "7d": 3.0,
        "30d": 5.0,
        "3m": 10.0,
        "6m": 15.0,
        "1y": 20.0
    },
    "WEAK": {
        "1d": -1.0,
        "3d": -2.0,
        "7d": -3.0,
        "30d": -5.0,
        "3m": -10.0,
        "6m": -15.0,
        "1y": -20.0
    }
}

# ============================================================================
# VOLUME THRESHOLDS
# ============================================================================

VOLUME_THRESHOLDS = {
    "SPIKE": 3.0,           # rvol > 3 (3x normal volume)
    "HIGH": 2.0,            # rvol > 2 (2x normal)
    "ELEVATED": 1.5,        # rvol > 1.5 (50% above normal)
    "NORMAL": 1.0,          # rvol ~1 (normal volume)
    "LOW": 0.5,             # rvol < 0.5 (half normal)
    "DRY": 0.3              # rvol < 0.3 (very low liquidity)
}

# Volume ratio thresholds (for vol_ratio columns)
VOLUME_RATIO_THRESHOLDS = {
    "SURGE": 150,           # >150% of average
    "INCREASING": 120,      # >120% of average
    "NORMAL": 100,          # ~100% of average
    "DECREASING": 80,       # <80% of average
    "DRYING": 50            # <50% of average
}

# ============================================================================
# VALUATION THRESHOLDS
# ============================================================================

# PE ranges for Indian market
PE_RANGES = {
    "DEEP_VALUE": (0, 12),      # PE 0-12: Deep value territory
    "VALUE": (12, 18),          # PE 12-18: Value stocks
    "FAIR": (18, 25),           # PE 18-25: Fair valuation
    "GROWTH": (25, 35),         # PE 25-35: Growth premium
    "EXPENSIVE": (35, 50),      # PE 35-50: Expensive
    "BUBBLE": (50, float('inf')) # PE >50: Bubble territory
}

# EPS growth ranges
EPS_GROWTH_RANGES = {
    "HYPER": 50,            # >50% EPS growth
    "HIGH": 25,             # 25-50% growth
    "MODERATE": 10,         # 10-25% growth
    "LOW": 0,               # 0-10% growth
    "NEGATIVE": -10,        # -10% to 0%
    "DECLINING": -25        # < -25% decline
}

# ============================================================================
# TECHNICAL THRESHOLDS
# ============================================================================

# Position in 52-week range
POSITION_52W_RANGES = {
    "NEAR_HIGH": (85, 100),     # Top 15% of range (breakout zone)
    "UPPER": (70, 85),          # Upper portion
    "MIDDLE_HIGH": (50, 70),    # Above middle
    "MIDDLE_LOW": (30, 50),     # Below middle
    "LOWER": (15, 30),          # Lower portion
    "NEAR_LOW": (0, 15)         # Bottom 15% (oversold)
}

# Distance from SMAs
SMA_DISTANCE_THRESHOLDS = {
    "FAR_ABOVE": 10,        # >10% above SMA (overextended)
    "ABOVE": 5,             # 5-10% above (bullish)
    "NEAR": 2,              # Within ±2% (equilibrium)
    "BELOW": -5,            # 5-10% below (bearish)
    "FAR_BELOW": -10        # >10% below (oversold)
}

# ============================================================================
# RISK LEVELS
# ============================================================================

RISK_LEVELS = {
    "VERY_LOW": (0, 20),
    "LOW": (20, 40),
    "MEDIUM": (40, 60),
    "HIGH": (60, 80),
    "VERY_HIGH": (80, 100)
}

# Risk factors and their weights
RISK_FACTORS = {
    "high_pe": 15,              # PE > 40
    "negative_eps": 25,         # Loss-making company
    "low_volume": 20,           # volume_1d < 50000
    "high_volatility": 15,      # Large price swings
    "penny_stock": 20,          # Price < ₹50
    "near_52w_low": 10,         # Within 10% of 52w low
    "no_profit": 30             # PE <= 0
}

# ============================================================================
# MARKET CAP CATEGORIES (in Rupees)
# ============================================================================

MARKET_CAP_RANGES = {
    "MEGA": 1e12,           # > ₹1 Lakh Crore (1 trillion)
    "LARGE": 2e11,          # ₹20K - 1 Lakh Crore
    "MID": 5e10,            # ₹5K - 20K Crore
    "SMALL": 5e9,           # ₹500 - 5K Crore
    "MICRO": 0              # < ₹500 Crore
}

# Category mapping
MARKET_CAP_CATEGORIES = {
    "Mega Cap": (MARKET_CAP_RANGES["LARGE"], float('inf')),
    "Large Cap": (MARKET_CAP_RANGES["MID"], MARKET_CAP_RANGES["LARGE"]),
    "Mid Cap": (MARKET_CAP_RANGES["SMALL"], MARKET_CAP_RANGES["MID"]),
    "Small Cap": (MARKET_CAP_RANGES["MICRO"], MARKET_CAP_RANGES["SMALL"]),
    "Micro Cap": (0, MARKET_CAP_RANGES["MICRO"])
}

# ============================================================================
# EPS TIERS (matching your exact labels)
# ============================================================================

EPS_TIERS = {
    '95↑': {'min': 95, 'label': 'Elite', 'score_boost': 20},
    '75↑': {'min': 75, 'label': 'Excellent', 'score_boost': 15},
    '55↑': {'min': 55, 'label': 'Strong', 'score_boost': 10},
    '35↑': {'min': 35, 'label': 'Good', 'score_boost': 5},
    '15↑': {'min': 15, 'label': 'Above Average', 'score_boost': 2},
    '5↑': {'min': 5, 'label': 'Average', 'score_boost': 0},
    '0': {'min': 0, 'label': 'Neutral', 'score_boost': 0},
    '5↓': {'min': -5, 'label': 'Below Average', 'score_boost': -5},
    '15↓': {'min': -15, 'label': 'Weak', 'score_boost': -10},
    '35↓': {'min': -35, 'label': 'Poor', 'score_boost': -15},
    '55↓': {'min': -55, 'label': 'Very Poor', 'score_boost': -20}
}

# ============================================================================
# PRICE TIERS (matching your exact labels)
# ============================================================================

PRICE_TIERS = {
    '10K↑': {'min': 10000, 'label': 'Ultra Premium'},
    '5K↑': {'min': 5000, 'label': 'Premium'},
    '2K↑': {'min': 2000, 'label': 'High'},
    '1K↑': {'min': 1000, 'label': 'Mid-High'},
    '500↑': {'min': 500, 'label': 'Mid'},
    '250↑': {'min': 250, 'label': 'Mid-Low'},
    '100↑': {'min': 100, 'label': 'Low'},
    '50↑': {'min': 50, 'label': 'Micro'},
    '25↑': {'min': 25, 'label': 'Penny'},
    '10↑': {'min': 10, 'label': 'Ultra Penny'},
    '100↓': {'max': 100, 'label': 'Below 100'}
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Color scheme
SIGNAL_COLORS = {
    'BUY': '#00d26a',           # Bright green
    'STRONG_BUY': '#00ff00',    # Neon green
    'WATCH': '#ffa500',         # Orange
    'NEUTRAL': '#808080',       # Gray
    'AVOID': '#ff4b4b'          # Red
}

# Chart colors
CHART_COLORS = {
    'positive': '#00d26a',
    'negative': '#ff4b4b',
    'neutral': '#808080',
    'background': '#0e1117',
    'grid': '#1e2329',
    'text': '#fafafa'
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Cache settings
CACHE_DURATION_MINUTES = 5      # Refresh data every 5 minutes

# Display limits
MAX_DISPLAY_ROWS = 100          # Max rows in tables
MAX_WATCHLIST_SIZE = 50         # Max stocks in watchlist
MAX_ALERTS = 20                 # Max alerts to show
DEFAULT_TOP_N = 20              # Default number of top stocks

# ============================================================================
# DATA QUALITY THRESHOLDS
# ============================================================================

DATA_QUALITY_THRESHOLDS = {
    "MIN_ROWS": 100,            # Minimum stocks required
    "MAX_NULL_PERCENT": 30,     # Max 30% missing data
    "MIN_PRICE": 0.01,          # Minimum valid price
    "MAX_PRICE": 1000000,       # Maximum valid price (₹10 Lakh)
    "MIN_VOLUME": 0,            # Minimum volume
    "MAX_PE": 1000,             # Maximum reasonable PE
    "MIN_PE": -100              # Minimum PE (can be negative)
}

# ============================================================================
# REQUIRED COLUMNS (for validation)
# ============================================================================

REQUIRED_WATCHLIST_COLUMNS = [
    'ticker', 'company_name', 'price', 'sector', 'market_cap',
    'pe', 'ret_1d', 'ret_7d', 'ret_30d', 'volume_1d', 'rvol'
]

REQUIRED_SECTOR_COLUMNS = [
    'sector', 'sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d',
    'sector_count'
]

# ============================================================================
# SECTOR GROUPS (for filtering and analysis)
# ============================================================================

SECTOR_GROUPS = {
    "Technology": [
        "IT", "Software", "Internet", "Telecom", 
        "Electronics", "Technology", "Digital"
    ],
    "Financial": [
        "Banks", "Finance", "Insurance", "NBFC", 
        "Asset Management", "Broking"
    ],
    "Consumer": [
        "FMCG", "Retail", "Consumer Durables", "Hotels", 
        "Media", "Entertainment", "Food & Beverages"
    ],
    "Industrial": [
        "Auto", "Engineering", "Infrastructure", "Power", 
        "Capital Goods", "Construction", "Logistics"
    ],
    "Healthcare": [
        "Pharma", "Healthcare", "Hospitals", "Diagnostics",
        "Medical Equipment"
    ],
    "Materials": [
        "Metals", "Chemicals", "Cement", "Mining", 
        "Paper", "Packaging", "Glass"
    ],
    "Energy": [
        "Oil & Gas", "Power", "Renewable Energy", 
        "Coal", "Utilities"
    ]
}

# ============================================================================
# ALERT THRESHOLDS
# ============================================================================

ALERT_THRESHOLDS = {
    "price_spike": 5.0,         # >5% single day move
    "volume_spike": 3.0,        # >3x normal volume
    "new_high": 95.0,           # >95% of 52w range
    "new_low": 5.0,             # <5% of 52w range
    "eps_surprise": 20.0        # >20% EPS change
}

# ============================================================================
# MARKET REGIME THRESHOLDS
# ============================================================================

MARKET_REGIMES = {
    "BULL": {
        "breadth": 65,          # >65% stocks advancing
        "avg_return": 5,        # >5% average 30d return
        "sector_participation": 70  # >70% sectors positive
    },
    "BEAR": {
        "breadth": 35,          # <35% stocks advancing
        "avg_return": -5,       # <-5% average 30d return
        "sector_participation": 30  # <30% sectors positive
    },
    "SIDEWAYS": {
        "breadth": 50,          # ~50% stocks advancing
        "avg_return": 0,        # -2% to +2% average return
        "sector_participation": 50  # Mixed sectors
    }
}

# ============================================================================
# FILTER DEFAULTS
# ============================================================================

DEFAULT_FILTERS = {
    'min_score': 60,
    'min_volume': 50000,
    'max_risk': 70,
    'signal_types': ['BUY', 'WATCH'],
    'risk_levels': ['VERY_LOW', 'LOW', 'MEDIUM']
}

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'DATA_LOAD_FAILED': "Unable to load data. Please check your internet connection and Google Sheets permissions.",
    'INVALID_DATA': "Data validation failed. Please check your Google Sheets format.",
    'NO_DATA': "No data available for the selected filters.",
    'CALCULATION_ERROR': "Error in calculations. Please refresh the page."
}

# ============================================================================
# SUCCESS MESSAGES
# ============================================================================

SUCCESS_MESSAGES = {
    'DATA_LOADED': "Data loaded successfully",
    'FILTERS_APPLIED': "Filters applied",
    'EXPORT_COMPLETE': "Data exported successfully"
}

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

# Quick validation on import
if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Configuration Loaded")
    print("="*60)
    print(f"\n📊 Google Sheets ID: {GOOGLE_SHEET_ID}")
    print(f"\n📑 Configured Sheets:")
    for sheet, config in SHEET_CONFIGS.items():
        print(f"  - {config['name']} (GID: {config['gid']})")
    print(f"\n⚙️ Factor Weights:")
    for factor, weight in FACTOR_WEIGHTS.items():
        print(f"  - {factor}: {weight*100:.0f}%")
    print(f"\n🎯 Signal Levels:")
    for level, threshold in SIGNAL_LEVELS.items():
        print(f"  - {level}: {threshold}")
    print("\n✅ Configuration ready!")
    print("="*60)
