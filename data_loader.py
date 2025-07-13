#!/usr/bin/env python3
# =============================================================================
# data_loader.py – M.A.N.T.R.A. Data Foundation (FINAL, BUG-FREE, v1.0.0)
# =============================================================================
"""
Loads, cleans, merges and health-checks the three Google-Sheets tabs that power
M.A.N.T.R.A.  Returns two ready-for-analysis dataframes plus a health-report
dict.  Zero external dependencies beyond the five packages in requirements.txt.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from constants import (  # pylint: disable=import-error
    CACHE_DURATION_MINUTES,
    DATA_QUALITY_THRESHOLDS,
    GOOGLE_SHEET_ID,
    SHEET_CONFIGS,
)

# -----------------------------------------------------------------------------#
#  0. Logging
# -----------------------------------------------------------------------------#
logger = logging.getLogger(__name__)
if not logger.handlers:
    hnd = logging.StreamHandler()
    hnd.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(hnd)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------#
#  1. Public API
# -----------------------------------------------------------------------------#
class DataLoaderError(RuntimeError):
    """Raised when loading or validation fails catastrophically."""


class DataLoader:
    """Static helpers – no instance state ➜ inherently thread-safe."""

    # ---------------------------------------------------------------------#
    #  1.1 High-level entry point (cached by Streamlit)
    # ---------------------------------------------------------------------#
    @staticmethod
    @st.cache_data(ttl=CACHE_DURATION_MINUTES * 60, show_spinner=False)
    def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load and clean **Watchlist**, **Returns** and **Sector** tabs, merge the
        first two, add derived columns, validate quality and ship back:

        Returns
        -------
        stocks_df : pd.DataFrame
            Watchlist ⨝ Returns (one row per ticker, fully cleaned)
        sector_df : pd.DataFrame
            Cleaned sector data (one row per sector)
        health    : dict
            Lightweight health-report with status/warnings/errors/metrics
        """
        health: Dict[str, object] = {
            "status": "loading",
            "timestamp": datetime.utcnow().isoformat(),
            "errors": [],
            "warnings": [],
        }

        try:
            # ----------------------------------------------------------#
            # Build CSV export URLs
            # ----------------------------------------------------------#
            url_tpl = (
                "https://docs.google.com/spreadsheets/d/"
                f"{GOOGLE_SHEET_ID}/export?format=csv&gid={{gid}}"
            )
            watchlist_url = url_tpl.format(gid=SHEET_CONFIGS["watchlist"]["gid"])
            returns_url = url_tpl.format(gid=SHEET_CONFIGS["returns"]["gid"])
            sector_url = url_tpl.format(gid=SHEET_CONFIGS["sector"]["gid"])

            # ----------------------------------------------------------#
            # Load
            # ----------------------------------------------------------#
            watchlist_df = pd.read_csv(watchlist_url)
            returns_df = pd.read_csv(returns_url)
            sector_df = pd.read_csv(sector_url)

            # ----------------------------------------------------------#
            # Clean
            # ----------------------------------------------------------#
            watchlist_df = DataLoader._clean_watchlist(watchlist_df)
            returns_df = DataLoader._clean_returns(returns_df)
            sector_df = DataLoader._clean_sector(sector_df)

            # ----------------------------------------------------------#
            # Merge & enrich
            # ----------------------------------------------------------#
            stocks_df = DataLoader._merge_stock_data(watchlist_df, returns_df)
            stocks_df = DataLoader._add_calculated_fields(stocks_df)

            # ----------------------------------------------------------#
            # Validate
            # ----------------------------------------------------------#
            health["warnings"].extend(DataLoader._validate_data(stocks_df, sector_df))
            health.update(
                {
                    "watchlist_rows": len(watchlist_df),
                    "returns_rows": len(returns_df),
                    "sector_rows": len(sector_df),
                    "merged_rows": len(stocks_df),
                    "data_quality": DataLoader._data_quality_score(stocks_df),
                }
            )
            health["status"] = "success"
            logger.info("✅ Data loaded & validated (%s stocks)", len(stocks_df))
            return stocks_df, sector_df, health

        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("❌ Data loading failure")
            health["status"] = "error"
            health["errors"].append(str(exc))
            # Return empty frames so downstream code fails gracefully
            return pd.DataFrame(), pd.DataFrame(), health

    # =====================================================================#
    #  2. Cleaning helpers
    # =====================================================================#
    @staticmethod
    def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()
        return df.dropna(how="all").drop_duplicates()

    @classmethod
    def _clean_watchlist(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls._basic_clean(df)

        # ----------------------------------------------------------------#
        # Numeric coercion helpers
        # ----------------------------------------------------------------#
        def _to_num(series: pd.Series, strip: str = "") -> pd.Series:
            return pd.to_numeric(series.astype(str).str.replace(strip, "", regex=False).str.replace(",", ""), errors="coerce")

        price_cols = [
            "price",
            "prev_close",
            "low_52w",
            "high_52w",
            "sma_20d",
            "sma_50d",
            "sma_200d",
        ]
        pct_cols = [
            "ret_1d",
            "ret_3d",
            "ret_7d",
            "ret_30d",
            "ret_3m",
            "ret_6m",
            "ret_1y",
            "ret_3y",
            "ret_5y",
            "from_low_pct",
            "from_high_pct",
            "vol_ratio_1d_90d",
            "vol_ratio_7d_90d",
            "vol_ratio_30d_90d",
            "eps_change_pct",
        ]
        vol_cols = ["volume_1d", "volume_7d", "volume_30d", "volume_3m"]
        numeric_cols = ["pe", "eps_current", "eps_last_qtr", "eps_duplicate", "rvol", "year"]

        for col in price_cols:
            if col in df:
                df[col] = _to_num(df[col], "₹")
        for col in pct_cols:
            if col in df:
                df[col] = _to_num(df[col], "%")
        for col in vol_cols + numeric_cols:
            for col_ in [col]:
                if col_ in df:
                    df[col_] = _to_num(df[col_])

        # Market-cap parsing
        if "market_cap" in df:
            df["market_cap"] = df["market_cap"].apply(DataLoader._parse_market_cap)

        # Ticker & exchange normalisation
        if "ticker" in df:
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        if "exchange" in df:
            df["exchange"] = df["exchange"].astype(str).str.upper().str.strip()

        return df

    @classmethod
    def _clean_returns(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls._basic_clean(df)
        pct_cols = [c for c in df if c.startswith(("returns_ret_", "avg_ret_"))]

        for col in pct_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors="coerce")

        if "ticker" in df:
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        return df

    @classmethod
    def _clean_sector(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls._basic_clean(df)
        pct_cols = [c for c in df if c.startswith(("sector_ret_", "sector_avg_"))]
        for col in pct_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors="coerce")
        if "sector_count" in df:
            df["sector_count"] = pd.to_numeric(df["sector_count"], errors="coerce")
        return df

    # =====================================================================#
    #  3. Merge & enrich
    # =====================================================================#
    @staticmethod
    def _merge_stock_data(wl: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
        if "ticker" not in wl or "ticker" not in rt:
            logger.warning("Ticker column missing – merge skipped.")
            return wl

        rt_keep = [c for c in rt.columns if c not in wl.columns or c == "ticker"]
        merged = wl.merge(rt[rt_keep], on="ticker", how="left", suffixes=("", "_rt"))

        # Deduplicate (shouldn’t happen but belt-and-braces)
        dupes = merged["ticker"].duplicated(keep="first")
        if dupes.any():
            logger.warning("Dropped %s duplicated tickers.", dupes.sum())
            merged = merged[~dupes]

        return merged

    @staticmethod
    def _add_calculated_fields(df: pd.DataFrame) -> pd.DataFrame:
        if {"price", "low_52w", "high_52w"}.issubset(df.columns):
            rng = df["high_52w"] - df["low_52w"]
            df["position_52w"] = np.where(rng > 0, ((df["price"] - df["low_52w"]) / rng) * 100, np.nan)

        for sma in ("sma_20d", "sma_50d", "sma_200d"):
            if sma in df and "price" in df:
                df[f"dist_from_{sma}"] = (df["price"] - df[sma]) / df[sma] * 100

        if "rvol" in df:
            df["volume_spike"] = df["rvol"] > 2.0
        if "pe" in df:
            df["is_value_stock"] = df["pe"].between(0, 20)
        if {"ret_7d", "ret_30d"}.issubset(df.columns):
            df["has_momentum"] = (df["ret_7d"] > 0) & (df["ret_30d"] > 0)
        if "avg_ret_30d" in df:
            df["trend_strength"] = df["avg_ret_30d"]
        return df

    # =====================================================================#
    #  4. Validation & quality metrics
    # =====================================================================#
    @staticmethod
    def _validate_data(stocks: pd.DataFrame, sectors: pd.DataFrame) -> list[str]:
        warnings: list[str] = []

        # Critical columns
        crit = {"ticker", "price", "company_name", "sector", "pe"}
        missing = crit - set(stocks.columns)
        if missing:
            warnings.append(f"Missing critical cols: {', '.join(sorted(missing))}")

        # Row count
        if len(stocks) < DATA_QUALITY_THRESHOLDS["MIN_ROWS"]:
            warnings.append(f"Low stock count ({len(stocks)})")

        # Price sanity
        if "price" in stocks:
            invalid = (~stocks["price"].between(
                DATA_QUALITY_THRESHOLDS["MIN_PRICE"], DATA_QUALITY_THRESHOLDS["MAX_PRICE"]
            ) | stocks["price"].isna()).sum()
            if invalid:
                warnings.append(f"{invalid} invalid price values")

        # Sector coverage
        if not sectors.empty and {"sector"}.issubset(stocks.columns | sectors.columns):
            missing_sec = set(stocks["sector"].dropna()) - set(sectors.get("sector", []))
            if missing_sec:
                warnings.append(f"Sectors absent in sector sheet: {', '.join(list(missing_sec)[:5])}")

        return warnings

    @staticmethod
    def _data_quality_score(stocks: pd.DataFrame) -> float:
        if stocks.empty:
            return 0.0
        completeness = 1 - stocks[["ticker", "price", "pe", "sector", "company_name"]].isna().mean()
        valid_prices = stocks["price"].between(
            DATA_QUALITY_THRESHOLDS["MIN_PRICE"], DATA_QUALITY_THRESHOLDS["MAX_PRICE"]
        ).mean()
        return round(((completeness.mean() + valid_prices) / 2) * 100, 2)

    # =====================================================================#
    #  5. Utility
    # =====================================================================#
    @staticmethod
    def _parse_market_cap(val) -> float:
        """Supports '₹123.4 Cr', '456 L', plain numbers, NaN."""
        if pd.isna(val):
            return np.nan
        s = str(val).upper().replace("₹", "").replace(",", "").strip()
        try:
            if any(k in s for k in ("CR", "CRORE")):
                return float(s.split()[0].replace("CR", "").replace("CRORE", "")) * 1e7
            if any(k in s for k in ("L", "LAC", "LAKH")):
                return float(s.split()[0].replace("LAC", "").replace("LAKH", "").replace("L", "")) * 1e5
            return float(s)
        except ValueError:
            return np.nan


# -----------------------------------------------------------------------------#
#  6. Quick CLI check
# -----------------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover
    st.runtime.exists = lambda: False  # type: ignore[attr-defined] (offline run)
    df_stocks, df_sector, hlth = DataLoader.load_all_data()
    print(df_stocks.head(3))
    print("Health:", hlth)
