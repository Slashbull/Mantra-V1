#!/usr/bin/env python3
# =============================================================================
# constants.py – M.A.N.T.R.A. Configuration Hub (FINAL, BUG-FREE, v1.0.0)
# =============================================================================
"""
Single source of truth for every tunable parameter used by the M.A.N.T.R.A.
stock-intelligence engine.  All modules import from here; no magic numbers
elsewhere.

This file is **immutable at runtime** thanks to MappingProxyType wrappers.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Dict, Tuple

# -----------------------------------------------------------------------------#
#  0. Logging
# -----------------------------------------------------------------------------#
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------#
#  1. Google Sheets ⇢ CSV endpoints
# -----------------------------------------------------------------------------#
GOOGLE_SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"

SHEET_CONFIGS = MappingProxyType(
    {
        "watchlist": {
            "gid": "2026492216",
            "name": "ALL STOCKS 2025 Watchlist",
            "description": "Master stock universe with all attributes",
        },
        "returns": {
            "gid": "100734077",
            "name": "Stock Return Analysis",
            "description": "Raw returns and averages for every stock",
        },
        "sector": {
            "gid": "140104095",
            "name": "ALL STOCKS 2025 Sector Analysis",
            "description": "Sector-level aggregate data",
        },
    }
)

# -----------------------------------------------------------------------------#
#  2. Signal & factor weights
# -----------------------------------------------------------------------------#
SIGNAL_LEVELS: Dict[str, int] = MappingProxyType(
    {"STRONG_BUY": 85, "BUY": 75, "WATCH": 60, "NEUTRAL": 40, "AVOID": 20}
)

FACTOR_WEIGHTS: Dict[str, float] = MappingProxyType(
    {
        "momentum": 0.30,
        "value": 0.25,
        "technical": 0.20,
        "volume": 0.15,
        "fundamentals": 0.10,
    }
)

# -----------------------------------------------------------------------------#
#  3. Momentum, volume, valuation & technical thresholds
# -----------------------------------------------------------------------------#
MOMENTUM_THRESHOLDS = MappingProxyType(
    {
        "STRONG": {"1d": 3, "3d": 5, "7d": 7, "30d": 15, "3m": 25, "6m": 40, "1y": 60},
        "MODERATE": {"1d": 1, "3d": 2, "7d": 3, "30d": 5, "3m": 10, "6m": 15, "1y": 20},
        "WEAK": {"1d": -1, "3d": -2, "7d": -3, "30d": -5, "3m": -10, "6m": -15, "1y": -20},
    }
)

VOLUME_THRESHOLDS = MappingProxyType(
    {"SPIKE": 3.0, "HIGH": 2.0, "ELEVATED": 1.5, "NORMAL": 1.0, "LOW": 0.5, "DRY": 0.3}
)

VOLUME_RATIO_THRESHOLDS = MappingProxyType(
    {"SURGE": 150, "INCREASING": 120, "NORMAL": 100, "DECREASING": 80, "DRYING": 50}
)

PE_RANGES: Dict[str, Tuple[float, float]] = MappingProxyType(
    {
        "DEEP_VALUE": (0, 12),
        "VALUE": (12, 18),
        "FAIR": (18, 25),
        "GROWTH": (25, 35),
        "EXPENSIVE": (35, 50),
        "BUBBLE": (50, float("inf")),
    }
)

EPS_GROWTH_RANGES = MappingProxyType(
    {"HYPER": 50, "HIGH": 25, "MODERATE": 10, "LOW": 0, "NEGATIVE": -10, "DECLINING": -25}
)

POSITION_52W_RANGES = MappingProxyType(
    {
        "NEAR_HIGH": (85, 100),
        "UPPER": (70, 85),
        "MIDDLE_HIGH": (50, 70),
        "MIDDLE_LOW": (30, 50),
        "LOWER": (15, 30),
        "NEAR_LOW": (0, 15),
    }
)

SMA_DISTANCE_THRESHOLDS = MappingProxyType(
    {"FAR_ABOVE": 10, "ABOVE": 5, "NEAR": 2, "BELOW": -5, "FAR_BELOW": -10}
)

# -----------------------------------------------------------------------------#
#  4. Market-cap & tiers
# -----------------------------------------------------------------------------#
MARKET_CAP_RANGES = MappingProxyType(
    {"MEGA": 1e12, "LARGE": 2e11, "MID": 5e10, "SMALL": 5e9, "MICRO": 0}
)

MARKET_CAP_CATEGORIES = MappingProxyType(
    {
        "Mega Cap": (MARKET_CAP_RANGES["MEGA"], float("inf")),
        "Large Cap": (MARKET_CAP_RANGES["LARGE"], MARKET_CAP_RANGES["MEGA"]),
        "Mid Cap": (MARKET_CAP_RANGES["MID"], MARKET_CAP_RANGES["LARGE"]),
        "Small Cap": (MARKET_CAP_RANGES["SMALL"], MARKET_CAP_RANGES["MID"]),
        "Micro Cap": (0, MARKET_CAP_RANGES["SMALL"]),
    }
)

EPS_TIERS = MappingProxyType(
    {
        "95↑": {"min": 95, "label": "Elite", "score_boost": 20},
        "75↑": {"min": 75, "label": "Excellent", "score_boost": 15},
        "55↑": {"min": 55, "label": "Strong", "score_boost": 10},
        "35↑": {"min": 35, "label": "Good", "score_boost": 5},
        "15↑": {"min": 15, "label": "Above Average", "score_boost": 2},
        "5↑": {"min": 5, "label": "Average", "score_boost": 0},
        "0": {"min": 0, "label": "Neutral", "score_boost": 0},
        "5↓": {"min": -5, "label": "Below Average", "score_boost": -5},
        "15↓": {"min": -15, "label": "Weak", "score_boost": -10},
        "35↓": {"min": -35, "label": "Poor", "score_boost": -15},
        "55↓": {"min": -55, "label": "Very Poor", "score_boost": -20},
    }
)

PRICE_TIERS = MappingProxyType(
    {
        "10K↑": {"min": 10_000, "label": "Ultra Premium"},
        "5K↑": {"min": 5_000, "label": "Premium"},
        "2K↑": {"min": 2_000, "label": "High"},
        "1K↑": {"min": 1_000, "label": "Mid-High"},
        "500↑": {"min": 500, "label": "Mid"},
        "250↑": {"min": 250, "label": "Mid-Low"},
        "100↑": {"min": 100, "label": "Low"},
        "50↑": {"min": 50, "label": "Micro"},
        "25↑": {"min": 25, "label": "Penny"},
        "10↑": {"min": 10, "label": "Ultra Penny"},
        "100↓": {"max": 100, "label": "Below 100"},
    }
)

# -----------------------------------------------------------------------------#
#  5. UI colours
# -----------------------------------------------------------------------------#
SIGNAL_COLORS = MappingProxyType(
    {"BUY": "#00d26a", "STRONG_BUY": "#00ff00", "WATCH": "#ffa500", "NEUTRAL": "#808080", "AVOID": "#ff4b4b"}
)

CHART_COLORS = MappingProxyType(
    {"positive": "#00d26a", "negative": "#ff4b4b", "neutral": "#808080", "background": "#0e1117", "grid": "#1e2329", "text": "#fafafa"}
)

# -----------------------------------------------------------------------------#
#  6. Performance & cache
# -----------------------------------------------------------------------------#
CACHE_DURATION_MINUTES = 5
MAX_DISPLAY_ROWS = 100
MAX_WATCHLIST_SIZE = 50
MAX_ALERTS = 20
DEFAULT_TOP_N = 20

# -----------------------------------------------------------------------------#
#  7. Data-quality validation
# -----------------------------------------------------------------------------#
DATA_QUALITY_THRESHOLDS = MappingProxyType(
    {
        "MIN_ROWS": 100,
        "MAX_NULL_PERCENT": 30,
        "MIN_PRICE": 0.01,
        "MAX_PRICE": 1_000_000,
        "MIN_VOLUME": 0,
        "MAX_PE": 1000,
        "MIN_PE": -100,
    }
)

REQUIRED_WATCHLIST_COLUMNS = (
    "ticker company_name price sector market_cap pe ret_1d ret_7d ret_30d volume_1d rvol".split()
)
REQUIRED_SECTOR_COLUMNS = (
    "sector sector_ret_1d sector_ret_7d sector_ret_30d sector_count".split()
)

# -----------------------------------------------------------------------------#
#  8. Risk
# -----------------------------------------------------------------------------#
RISK_LEVELS = MappingProxyType(
    {"VERY_LOW": (0, 20), "LOW": (20, 40), "MEDIUM": (40, 60), "HIGH": (60, 80), "VERY_HIGH": (80, 100)}
)

RISK_FACTORS = MappingProxyType(
    {
        "high_pe": 15,
        "negative_eps": 25,
        "low_volume": 20,
        "high_volatility": 15,
        "penny_stock": 20,
        "near_52w_low": 10,
        "no_profit": 30,
    }
)

# -----------------------------------------------------------------------------#
#  9. Sector groups
# -----------------------------------------------------------------------------#
SECTOR_GROUPS = MappingProxyType(
    {
        "Technology": ["IT", "Software", "Internet", "Telecom", "Electronics", "Technology", "Digital"],
        "Financial": ["Banks", "Finance", "Insurance", "NBFC", "Asset Management", "Broking"],
        "Consumer": ["FMCG", "Retail", "Consumer Durables", "Hotels", "Media", "Entertainment", "Food & Beverages"],
        "Industrial": ["Auto", "Engineering", "Infrastructure", "Power", "Capital Goods", "Construction", "Logistics"],
        "Healthcare": ["Pharma", "Healthcare", "Hospitals", "Diagnostics", "Medical Equipment"],
        "Materials": ["Metals", "Chemicals", "Cement", "Mining", "Paper", "Packaging", "Glass"],
        "Energy": ["Oil & Gas", "Power", "Renewable Energy", "Coal", "Utilities"],
    }
)

# -----------------------------------------------------------------------------#
# 10. Alerts & regimes
# -----------------------------------------------------------------------------#
ALERT_THRESHOLDS = MappingProxyType(
    {"price_spike": 5, "volume_spike": 3, "new_high": 95, "new_low": 5, "eps_surprise": 20}
)

MARKET_REGIMES = MappingProxyType(
    {
        "BULL": {"breadth": 65, "avg_return": 5, "sector_participation": 70},
        "BEAR": {"breadth": 35, "avg_return": -5, "sector_participation": 30},
        "SIDEWAYS": {"breadth": 50, "avg_return": 0, "sector_participation": 50},
    }
)

DEFAULT_FILTERS = MappingProxyType(
    {"min_score": 60, "min_volume": 50_000, "max_risk": 70, "signal_types": ["BUY", "WATCH"], "risk_levels": ["VERY_LOW", "LOW", "MEDIUM"]}
)

# -----------------------------------------------------------------------------#
# 11. User-facing messages
# -----------------------------------------------------------------------------#
ERROR_MESSAGES = MappingProxyType(
    {
        "DATA_LOAD_FAILED": "Unable to load data. Please check your internet connection and Google Sheets permissions.",
        "INVALID_DATA": "Data validation failed. Please check your Google Sheets format.",
        "NO_DATA": "No data available for the selected filters.",
        "CALCULATION_ERROR": "Error in calculations. Please refresh the page.",
    }
)

SUCCESS_MESSAGES = MappingProxyType(
    {"DATA_LOADED": "Data loaded successfully", "FILTERS_APPLIED": "Filters applied", "EXPORT_COMPLETE": "Data exported successfully"}
)

# -----------------------------------------------------------------------------#
# 12. Self-validation (runs on import)
# -----------------------------------------------------------------------------#
def _validate() -> None:
    """Fail-fast on mis-configuration."""
    if round(sum(FACTOR_WEIGHTS.values()), 6) != 1.0:
        raise ValueError("FACTOR_WEIGHTS must sum to 1.0")

    thresh = list(SIGNAL_LEVELS.values())
    if thresh != sorted(thresh, reverse=True):
        raise ValueError("SIGNAL_LEVELS must be strictly descending.")

    logger.info("✅ constants.py validation passed.")


_validate()

# -----------------------------------------------------------------------------#
# 13. CLI helper
# -----------------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover
    from pprint import pprint

    pprint(
        {
            "Google Sheet ID": GOOGLE_SHEET_ID,
            "Sheets": SHEET_CONFIGS,
            "Factor Weights": FACTOR_WEIGHTS,
            "Signal Levels": SIGNAL_LEVELS,
        },
        sort_dicts=False,
    )
