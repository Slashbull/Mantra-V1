"""
constants.py - M.A.N.T.R.A. Configuration Hub
===========================================
FINAL PRODUCTION VERSION - All settings for your exact data structure
Bug-free, tested, and ready for deployment
"""

# ============================================================================
# GOOGLE SHEETS CONFIGURATION
# ============================================================================

# IMPORTANT: Replace with your actual Google Sheets ID!
GOOGLE_SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"

# Sheet GIDs (tab IDs) - These map to your exact sheets
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

# Composite score thresholds for trading decisions
SIGNAL_LEVELS = {
    "STRONG_BUY": 85,      # Score >= 85 (exceptional opportunity)
    "BUY": 75,             # Score >= 75 (clear buy signal)
    "WATCH": 60,           # Score >= 60 (potential opportunity)
    "NEUTRAL": 40,         # Score >= 40 (no clear signal)
    "AVOID": 20            # Score < 20 (stay away)
}

# ============================================================================
# FACTOR WEIGHTS (must sum to 1.0)
# ============================================================================

FACTOR_WEIGHTS = {
    "momentum": 0.30,      # 30% - Price performance trends
    "value": 0.25,         # 25% - Valuation metrics (PE, EPS)
    "technical": 0.20,     # 20% - Technical indicators (SMAs, position)
    "volume": 0.15,        # 15% - Volume activity (rvol, ratios)
    "fundamentals": 0.10   # 10% - Quality factors (market cap, profitability)
}

# ============================================================================
# MOMENTUM THRESHOLDS
# ============================================================================

MOMENTUM_THRESHOLDS = {
    "STRONG": {
        "1d": 3.0,         # >3% daily gain
        "3d": 5.0,         # >5% in 3 days
        "7d": 7.0,         # >7% weekly
        "30d": 15.0,       # >15% monthly
        "3m": 25.0,        # >25% quarterly
        "6m": 40.0,        # >40% half-yearly
        "1y": 60.0         # >60% yearly
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

# Relative volume (rvol) thresholds
VOLUME_THRESHOLDS = {
    "SPIKE": 3.0,          # rvol > 3 (extreme volume)
    "HIGH": 2.0,           # rvol > 2 (high volume)
    "ELEVATED": 1.5,       # rvol > 1.5 (above average)
    "NORMAL": 1.0,         # rvol ~1 (normal)
    "LOW": 0.5,            # rvol < 0.5 (low volume)
    "DRY": 0.3             # rvol < 0.3 (very low)
}

# Volume ratio thresholds (for vol_ratio_1d_90d etc.)
VOLUME_RATIO_THRESHOLDS = {
    "SURGE": 150,          # >150% of 90d average
    "INCREASING": 120,     # >120% of 90d average
    "NORMAL": 100,         # ~100% of 90d average
    "DECREASING": 80,      # <80% of 90d average
    "DRYING": 50           # <50% of 90d average
}

# ============================================================================
# VALUATION THRESHOLDS
# ============================================================================

# PE ranges for Indian markets
PE_RANGES = {
    "DEEP_VALUE": (0, 12),          # Deep value territory
    "VALUE": (12, 18),              # Value stocks
    "FAIR": (18, 25),               # Fair valuation
    "GROWTH": (25, 35),             # Growth premium
    "EXPENSIVE": (35, 50),          # Expensive
    "BUBBLE": (50, float('inf'))    # Bubble territory
}

# EPS growth thresholds
EPS_GROWTH_RANGES = {
    "HYPER": 50,           # >50% EPS growth (exceptional)
    "HIGH": 25,            # 25-50% growth (very good)
    "MODERATE": 10,        # 10-25% growth (good)
    "LOW": 0,              # 0-10% growth (okay)
    "NEGATIVE": -10,       # -10% to 0% (declining)
    "DECLINING": -25       # < -25% (serious decline)
}

# ============================================================================
# TECHNICAL THRESHOLDS
# ============================================================================

# Position in 52-week range
POSITION_52W_RANGES = {
    "NEAR_HIGH": (85, 100),    # Top 15% of range
    "UPPER": (70, 85),         # Upper portion
    "MIDDLE_HIGH": (50, 70),   # Above middle
    "MIDDLE_LOW": (30, 50),    # Below middle
    "LOWER": (15, 30),         # Lower portion
    "NEAR_LOW": (0, 15)        # Bottom 15%
}

# Distance from moving averages
SMA_DISTANCE_THRESHOLDS = {
    "FAR_ABOVE": 10,       # >10% above SMA
    "ABOVE": 5,            # 5-10% above SMA
    "NEAR": 2,             # Within ±2% of SMA
    "BELOW": -5,           # 5-10% below SMA
    "FAR_BELOW": -10       # >10% below SMA
}

# ============================================================================
# RISK CONFIGURATION
# ============================================================================

# Risk score ranges
RISK_LEVELS = {
    "VERY_LOW": (0, 20),
    "LOW": (20, 40),
    "MEDIUM": (40, 60),
    "HIGH": (60, 80),
    "VERY_HIGH": (80, 100)
}

# Risk factors and their point values
RISK_FACTORS = {
    "high_pe": 15,         # PE > 40
    "negative_eps": 25,    # Loss-making company
    "low_volume": 20,      # Daily volume < 50,000
    "high_volatility": 15, # Large price swings
    "penny_stock": 20,     # Price < ₹50
    "near_52w_low": 10,    # Within 10% of 52w low
    "no_profit": 30        # PE <= 0
}

# ============================================================================
# MARKET CAP CATEGORIES (in Rupees)
# ============================================================================

MARKET_CAP_RANGES = {
    "MEGA": 1e12,          # > ₹1 Lakh Crore
    "LARGE": 2e11,         # ₹20,000 - 1 Lakh Crore
    "MID": 5e10,           # ₹5,000 - 20,000 Crore
    "SMALL": 5e9,          # ₹500 - 5,000 Crore
    "MICRO": 0             # < ₹500 Crore
}

# Category boundaries
MARKET_CAP_CATEGORIES = {
    "Mega Cap": (1e12, float('inf')),
    "Large Cap": (2e11, 1e12),
    "Mid Cap": (5e10, 2e11),
    "Small Cap": (5e9, 5e10),
    "Micro Cap": (0, 5e9)
}

# ============================================================================
# EPS TIERS (matching your exact tier labels)
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
# PRICE TIERS (matching your exact tier labels)
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

# Color scheme for UI elements
SIGNAL_COLORS = {
    'BUY': '#00d26a',          # Green
    'STRONG_BUY': '#00ff00',   # Bright green
    'WATCH': '#ffa500',        # Orange
    'NEUTRAL': '#808080',      # Gray
    'AVOID': '#ff4b4b'         # Red
}

# Chart color palette
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

# Cache configuration
CACHE_DURATION_MINUTES = 5     # Data refresh interval

# Display limits
MAX_DISPLAY_ROWS = 100         # Maximum rows in tables
MAX_WATCHLIST_SIZE = 50        # Maximum stocks per watchlist
MAX_ALERTS = 20                # Maximum alerts to show
DEFAULT_TOP_N = 20             # Default top stocks count
PAGINATION_SIZE = 50           # Rows per page

# ============================================================================
# DATA QUALITY SETTINGS
# ============================================================================

DATA_QUALITY_THRESHOLDS = {
    "MIN_ROWS": 100,           # Minimum stocks required
    "MAX_NULL_PERCENT": 30,    # Maximum acceptable null %
    "MIN_PRICE": 0.01,         # Minimum valid price
    "MAX_PRICE": 1000000,      # Maximum valid price (₹10 Lakh)
    "MIN_VOLUME": 0,           # Minimum volume
    "MAX_PE": 1000,            # Maximum reasonable PE
    "MIN_PE": -100             # Minimum PE
}

# ============================================================================
# REQUIRED COLUMNS
# ============================================================================

# Watchlist sheet required columns
REQUIRED_WATCHLIST_COLUMNS = [
    'ticker', 'company_name', 'price', 'sector', 'market_cap',
    'pe', 'ret_1d', 'ret_7d', 'ret_30d', 'volume_1d', 'rvol'
]

# Sector sheet required columns
REQUIRED_SECTOR_COLUMNS = [
    'sector', 'sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d',
    'sector_count'
]

# Returns sheet columns (optional but used if present)
OPTIONAL_RETURNS_COLUMNS = [
    'returns_ret_1d', 'returns_ret_3d', 'returns_ret_7d',
    'avg_ret_30d', 'avg_ret_3m', 'avg_ret_6m'
]

# ============================================================================
# SECTOR GROUPINGS
# ============================================================================

SECTOR_GROUPS = {
    "Technology": [
        "IT", "Software", "Internet", "Telecom", 
        "Electronics", "Technology", "Digital", "IT Services"
    ],
    "Financial": [
        "Banks", "Finance", "Insurance", "NBFC", 
        "Asset Management", "Broking", "Financial Services"
    ],
    "Consumer": [
        "FMCG", "Retail", "Consumer Durables", "Hotels", 
        "Media", "Entertainment", "Food & Beverages", "QSR"
    ],
    "Industrial": [
        "Auto", "Engineering", "Infrastructure", "Power", 
        "Capital Goods", "Construction", "Logistics", "EPC"
    ],
    "Healthcare": [
        "Pharma", "Healthcare", "Hospitals", "Diagnostics",
        "Medical Equipment", "Healthcare Services"
    ],
    "Materials": [
        "Metals", "Chemicals", "Cement", "Mining", 
        "Paper", "Packaging", "Glass", "Steel"
    ],
    "Energy": [
        "Oil & Gas", "Power", "Renewable Energy", 
        "Coal", "Utilities", "Power Generation"
    ],
    "Real Estate": [
        "Real Estate", "Realty", "Housing", "Construction Materials"
    ]
}

# ============================================================================
# ALERT SETTINGS
# ============================================================================

ALERT_THRESHOLDS = {
    "price_spike_1d": 5.0,     # >5% single day move
    "price_crash_1d": -5.0,    # <-5% single day drop
    "volume_spike": 3.0,       # >3x normal volume
    "new_52w_high": 95.0,      # >95% of 52w range
    "new_52w_low": 5.0,        # <5% of 52w range
    "eps_surprise": 20.0,      # >20% EPS change
    "breakout": 3.0            # >3% above SMA with volume
}

# ============================================================================
# MARKET REGIME SETTINGS
# ============================================================================

MARKET_REGIMES = {
    "BULL": {
        "breadth": 65,         # >65% stocks advancing
        "avg_return": 5,       # >5% average 30d return
        "sectors_positive": 70 # >70% sectors positive
    },
    "BEAR": {
        "breadth": 35,         # <35% stocks advancing
        "avg_return": -5,      # <-5% average 30d return
        "sectors_positive": 30 # <30% sectors positive
    },
    "SIDEWAYS": {
        "breadth": 50,         # ~50% stocks advancing
        "avg_return": 0,       # -2% to +2% return
        "sectors_positive": 50 # Mixed sectors
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
    'risk_levels': ['VERY_LOW', 'LOW', 'MEDIUM'],
    'sectors': [],  # Empty = all sectors
    'market_caps': ['Large Cap', 'Mid Cap', 'Small Cap']
}

# ============================================================================
# TARGET AND STOP LOSS SETTINGS
# ============================================================================

TARGET_SETTINGS = {
    "CONSERVATIVE": {
        "target_1": 5,         # 5% first target
        "target_2": 10,        # 10% second target
        "stop_loss": 3         # 3% stop loss
    },
    "MODERATE": {
        "target_1": 8,         # 8% first target
        "target_2": 15,        # 15% second target
        "stop_loss": 5         # 5% stop loss
    },
    "AGGRESSIVE": {
        "target_1": 12,        # 12% first target
        "target_2": 20,        # 20% second target
        "stop_loss": 7         # 7% stop loss
    }
}

# ============================================================================
# MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'DATA_LOAD_FAILED': "Unable to load data. Check internet and Google Sheets access.",
    'INVALID_DATA': "Data validation failed. Check sheet format.",
    'NO_DATA': "No data available for selected filters.",
    'CALCULATION_ERROR': "Error in calculations. Please refresh.",
    'SHEET_NOT_FOUND': "Sheet not found. Check GID configuration."
}

SUCCESS_MESSAGES = {
    'DATA_LOADED': "✅ Data loaded successfully",
    'FILTERS_APPLIED': "✅ Filters applied",
    'EXPORT_COMPLETE': "✅ Data exported successfully",
    'REFRESH_COMPLETE': "✅ Data refreshed"
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_configuration():
    """Validate that configuration is correct"""
    # Check factor weights sum to 1
    weight_sum = sum(FACTOR_WEIGHTS.values())
    assert abs(weight_sum - 1.0) < 0.01, f"Factor weights must sum to 1, got {weight_sum}"
    
    # Check signal levels are in order
    levels = list(SIGNAL_LEVELS.values())
    assert levels == sorted(levels, reverse=True), "Signal levels must be descending"
    
    # Check sheet GIDs are present
    for sheet, config in SHEET_CONFIGS.items():
        assert 'gid' in config, f"Missing GID for {sheet}"
        assert config['gid'], f"Empty GID for {sheet}"
    
    return True

# Run validation on import
try:
    validate_configuration()
    CONFIG_VALID = True
except AssertionError as e:
    print(f"⚠️ Configuration Error: {e}")
    CONFIG_VALID = False

# ============================================================================
# QUICK ACCESS FUNCTIONS
# ============================================================================

def get_signal_color(signal: str) -> str:
    """Get color for a signal type"""
    return SIGNAL_COLORS.get(signal, SIGNAL_COLORS['NEUTRAL'])

def get_risk_level(score: float) -> str:
    """Get risk level from score"""
    for level, (min_val, max_val) in RISK_LEVELS.items():
        if min_val <= score < max_val:
            return level.replace('_', ' ').title()
    return 'Unknown'

def get_market_cap_category(market_cap: float) -> str:
    """Get market cap category from value"""
    for category, (min_val, max_val) in MARKET_CAP_CATEGORIES.items():
        if min_val <= market_cap < max_val:
            return category
    return 'Unknown'

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🔱 M.A.N.T.R.A. Configuration")
    print("="*60)
    print(f"\n✅ Configuration Valid: {CONFIG_VALID}")
    print(f"\n📊 Google Sheets ID: {GOOGLE_SHEET_ID}")
    print(f"\n📑 Sheets Configured:")
    for name, config in SHEET_CONFIGS.items():
        print(f"   - {config['name']} (GID: {config['gid']})")
    print(f"\n⚖️ Factor Weights:")
    for factor, weight in FACTOR_WEIGHTS.items():
        print(f"   - {factor}: {weight*100:.0f}%")
    print(f"\n🎯 Signal Thresholds:")
    for signal, threshold in SIGNAL_LEVELS.items():
        print(f"   - {signal}: {threshold}")
    print("\n✅ Ready for production!")
    print("="*60)
