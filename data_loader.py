"""
data_loader.py - Final Adapter for M.A.N.T.R.A. (Locked Version)
================================================================
Thin adapter to load, clean, and validate all data for the dashboard
using core_system_foundation.py. All logic is in the core foundation.

- No logic in this file; all changes must be made in core_system_foundation.py.
- Returns watchlist_df (main stock universe), returns_df (optional), sector_df, and status dict.
- Always returns status with success/errors for robust app error handling.
"""

from core_system_foundation import load_and_process

import pandas as pd

def load_all_data():
    """
    Loads all dataframes for use in the dashboard.

    Returns:
        watchlist_df: DataFrame with main stock+returns data (merged)
        returns_df: None (for compatibility; all data is in watchlist_df)
        sector_df: DataFrame with sector-level data
        status: dict with 'success' (bool), 'errors' (list), 'health' (optional: health/quality/lineage)
    """
    try:
        watchlist_df, sector_df, health = load_and_process()
        # For compatibility: returns_df is not needed, set to None
        status = {
            "success": True,
            "errors": [],
            "health": health
        }
        return watchlist_df, None, sector_df, status
    except Exception as e:
        # Robust fallback for UI: empty DFs and error status
        return pd.DataFrame(), None, pd.DataFrame(), {
            "success": False,
            "errors": [str(e)]
        }
