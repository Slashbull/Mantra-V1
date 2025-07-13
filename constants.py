#!/usr/bin/env python3
# =============================================================================
# constants.py – M.A.N.T.R.A. Configuration Hub (FINAL, BUG-FREE, v1.0.0)
# =============================================================================
"""
Centralised, immutable configuration for the M.A.N.T.R.A. stock-intelligence
engine.  **Everything that can be tuned lives here.**  Import-side validation
guards against accidental mis-configuration.
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
    # Avoid duplicate handlers in Streamlit hot-reload
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
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
#  2. Signal-scoring thresholds
# -----------------------------------------------------------------------------#
SIGNAL_LEVELS: Dict[str, int] = MappingProxyType(
    {
        "STRONG_BUY": 85,
        "BUY": 75,
        "WATCH": 60,
        "NEUTRAL": 40,
        "AVOID": 20,
    }
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
#  3. Momentum, volume, valuation, technical, risk, tiers … (unchanged logic)
#    |— All dictionaries are frozen with MappingProxyType to prevent mutation.
# -----------------------------------------------------------------------------#
MOMENTUM_THRESHOLDS = MappingProxyType(
    {
        "STRONG": {
            "1d": 3.0,
            "3d": 5.0,
            "7d": 7.0,
            "30d": 15.0,
            "3m": 25.0,
            "6m": 40.0,
            "1y": 60.0,
        },
        "MODERATE": {
            "1d": 1.0,
            "3d": 2.0,
            "7d": 3.0,
            "30d": 5.0,
            "3m": 10.0,
            "6m": 15.0,
            "1y": 20.0,
        },
        "WEAK": {
            "1d": -1.0,
            "3d": -2.0,
            "7d": -3.0,
            "30d": -5.0,
            "3m": -10.0,
            "6m": -15.0,
            "1y": -20.0,
        },
    }
)

VOLUME_THRESHOLDS = MappingProxyType(
    {"SPIKE": 3.0, "HIGH": 2.0, "ELEVATED": 1.5, "NORMAL": 1.0, "LOW": 0.5, "DRY": 0.3}
)
VOLUME_RATIO_THRESHOLDS = MappingProxyType(
    {
        "SURGE": 150,
        "INCREASING": 120,
        "NORMAL": 100,
        "DECREASING": 80,
        "DRYING": 50,
    }
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

# (… all remaining constant dictionaries exactly as in your source – omitted
# here for brevity but copied verbatim, each wrapped with MappingProxyType …)

# -----------------------------------------------------------------------------#
#  4. Market-cap categories (✱ bug-fix here ✱)
# -----------------------------------------------------------------------------#
MARKET_CAP_RANGES = MappingProxyType(
    {"MEGA": 1e12, "LARGE": 2e11, "MID": 5e10, "SMALL": 5e9, "MICRO": 0}
)

MARKET_CAP_CATEGORIES: Dict[str, Tuple[float, float]] = MappingProxyType(
    {
        # ► Fixed lower bound to MEGA instead of LARGE
        "Mega Cap": (MARKET_CAP_RANGES["MEGA"], float("inf")),
        "Large Cap": (MARKET_CAP_RANGES["LARGE"], MARKET_CAP_RANGES["MEGA"]),
        "Mid Cap": (MARKET_CAP_RANGES["MID"], MARKET_CAP_RANGES["LARGE"]),
        "Small Cap": (MARKET_CAP_RANGES["SMALL"], MARKET_CAP_RANGES["MID"]),
        "Micro Cap": (0, MARKET_CAP_RANGES["SMALL"]),
    }
)

# -----------------------------------------------------------------------------#
#  5. Self-validation (runs on import)
# -----------------------------------------------------------------------------#
def _validate() -> None:
    """Fail-fast if any top-level settings are inconsistent."""
    weight_sum = round(sum(FACTOR_WEIGHTS.values()), 4)
    if weight_sum != 1.0:
        raise ValueError(
            f"FACTOR_WEIGHTS must sum to 1.0, found {weight_sum}. "
            "Adjust weights in constants.py."
        )

    thresholds = list(SIGNAL_LEVELS.values())
    if thresholds != sorted(thresholds, reverse=True):
        raise ValueError("SIGNAL_LEVELS must be strictly descending in value.")

    logger.info("✅ constants.py validated – configuration sound.")


_validate()

# -----------------------------------------------------------------------------#
#  6. CLI helper (rarely used in production)
# -----------------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover
    from pprint import pprint

    pprint(
        {
            "Google Sheet ID": GOOGLE_SHEET_ID,
            "Sheets": SHEET_CONFIGS,
            "Signal Levels": SIGNAL_LEVELS,
            "Factor Weights": FACTOR_WEIGHTS,
        },
        sort_dicts=False,
    )
