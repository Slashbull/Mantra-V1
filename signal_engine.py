#!/usr/bin/env python3
# =============================================================================
# signal_engine.py – M.A.N.T.R.A. Signal-Generation Engine (FINAL, BUG-FREE)
# =============================================================================
"""
Multi-factor scoring tuned to the exact columns produced by **data_loader.py**.
Pure pandas / NumPy logic, zero extra dependencies, Streamlit-cache friendly.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from constants import (  # pylint: disable=import-error
    EPS_GROWTH_RANGES,
    EPS_TIERS,
    FACTOR_WEIGHTS,
    MOMENTUM_THRESHOLDS,
    PE_RANGES,
    POSITION_52W_RANGES,
    PRICE_TIERS,  # (not directly used yet – retained for future tweaks)
    RISK_FACTORS,
    RISK_LEVELS,
    SIGNAL_LEVELS,
    SMA_DISTANCE_THRESHOLDS,  # (reserved)
    VOLUME_RATIO_THRESHOLDS,
    VOLUME_THRESHOLDS,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------#
#  1. Public API
# -----------------------------------------------------------------------------#
class SignalEngine:
    """Stateless – all helpers are `@staticmethod`s ➜ thread-safe & cacheable."""

    @staticmethod
    @st.cache_data(ttl=60, show_spinner=False)
    def calculate_all_signals(
        stocks_df: pd.DataFrame, sector_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add **~30** analytics columns (scores, decisions, targets…) to the
        already-clean `stocks_df` and return it.

        This is the only public method; everything else is private helpers.
        """
        if stocks_df.empty:
            logger.warning("Stock dataframe empty – skipping scoring.")
            return stocks_df

        df = stocks_df.copy()

        # ------------------------------------------------------------------#
        # 1.1 Factor scores
        # ------------------------------------------------------------------#
        df["momentum_score"] = SignalEngine._momentum_score(df)
        df["value_score"] = SignalEngine._value_score(df)
        df["technical_score"] = SignalEngine._technical_score(df)
        df["volume_score"] = SignalEngine._volume_score(df)
        df["fundamental_score"] = SignalEngine._fundamental_score(df)

        # ------------------------------------------------------------------#
        # 1.2 Sector bonus
        # ------------------------------------------------------------------#
        df = SignalEngine._add_sector_strength(df, sector_df)

        # ------------------------------------------------------------------#
        # 1.3 Composite score & primary decision
        # ------------------------------------------------------------------#
        df["composite_score"] = SignalEngine._composite_score(df)
        df["decision"] = df["composite_score"].map(SignalEngine._decision_from_score)

        # ------------------------------------------------------------------#
        # 1.4 Risk, opportunity, reasoning, targets
        # ------------------------------------------------------------------#
        df["risk_score"] = SignalEngine._risk_score(df)
        df["risk_level"] = df["risk_score"].map(SignalEngine._risk_level_from_score)
        df["opportunity_score"] = SignalEngine._opportunity_score(df)
        df["reasoning"] = df.apply(SignalEngine._reasoning, axis=1)
        df = SignalEngine._price_targets(df)

        # ------------------------------------------------------------------#
        # 1.5 Rank & percentile
        # ------------------------------------------------------------------#
        df["rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
        df["percentile"] = (df["composite_score"].rank(pct=True) * 100).round(1)

        return df

    # ======================================================================#
    #  2. Momentum
    # ======================================================================#
    @staticmethod
    def _momentum_score(df: pd.DataFrame) -> pd.Series:
        weights: Dict[str, float] = {
            "ret_1d": 0.05,
            "ret_3d": 0.05,
            "ret_7d": 0.10,
            "ret_30d": 0.20,
            "ret_3m": 0.30,
            "ret_6m": 0.20,
            "ret_1y": 0.10,
        }
        period_map = {
            "ret_1d": "1d",
            "ret_3d": "3d",
            "ret_7d": "7d",
            "ret_30d": "30d",
            "ret_3m": "3m",
            "ret_6m": "6m",
            "ret_1y": "1y",
        }

        base = pd.Series(0.0, index=df.index)
        total_w = 0.0

        for col, w in weights.items():
            if col not in df:
                continue
            threshold = MOMENTUM_THRESHOLDS["STRONG"][period_map[col]]
            col_score = 50 + (df[col].fillna(0) / threshold) * 25
            base += col_score.clip(0, 100) * w
            total_w += w

        if total_w == 0:
            return pd.Series(50.0, index=df.index)

        score = base / total_w

        # Consistency bonus via 30-day average return
        if "avg_ret_30d" in df:
            trend_bonus = pd.Series(0, index=df.index)
            trend_bonus[df["avg_ret_30d"] > 10] = 10
            trend_bonus[df["avg_ret_30d"] > 5] = 5
            score += trend_bonus

        return score.clip(0, 100).round(1)

    # ======================================================================#
    #  3. Value
    # ======================================================================#
    @staticmethod
    def _value_score(df: pd.DataFrame) -> pd.Series:
        comp_score = pd.Series(0.0, index=df.index)
        weight_acc = 0.0

        # ---- 3.1 PE ratio -------------------------------------------------#
        if "pe" in df:
            pe = df["pe"].fillna(df["pe"].median())
            pe_band_score = pd.Series(20.0, index=df.index)  # default worst

            for pe_range, s in [
                (PE_RANGES["DEEP_VALUE"], 95),
                (PE_RANGES["VALUE"], 85),
                (PE_RANGES["FAIR"], 70),
                (PE_RANGES["GROWTH"], 50),
                (PE_RANGES["EXPENSIVE"], 30),
                (PE_RANGES["BUBBLE"], 20),
            ]:
                lo, hi = pe_range
                pe_band_score[(pe >= lo) & (pe < hi)] = s
            pe_band_score[pe <= 0] = 20

            comp_score += pe_band_score * 0.5
            weight_acc += 0.5

        # ---- 3.2 EPS growth ----------------------------------------------#
        if "eps_change_pct" in df:
            growth = df["eps_change_pct"].fillna(0)
            growth_score = pd.Series(50.0, index=df.index)

            growth_score[growth > EPS_GROWTH_RANGES["HYPER"]] = 95
            growth_score[growth > EPS_GROWTH_RANGES["HIGH"]] = 80
            growth_score[growth > EPS_GROWTH_RANGES["MODERATE"]] = 65
            growth_score[growth > EPS_GROWTH_RANGES["LOW"]] = 50
            growth_score[growth > EPS_GROWTH_RANGES["NEGATIVE"]] = 35
            growth_score[growth <= EPS_GROWTH_RANGES["DECLINING"]] = 20

            comp_score += growth_score * 0.3
            weight_acc += 0.3

        # ---- 3.3 EPS tier bonus ------------------------------------------#
        if "eps_tier" in df:
            tier_bonus = pd.Series(0, index=df.index)
            for tier, cfg in EPS_TIERS.items():
                tier_bonus[df["eps_tier"] == tier] = cfg["score_boost"]
            comp_score += (50 + tier_bonus) * 0.2
            weight_acc += 0.2

        if weight_acc == 0:
            return pd.Series(50.0, index=df.index)

        return (comp_score / weight_acc).clip(0, 100).round(1)

    # ======================================================================#
    #  4. Technicals
    # ======================================================================#
    @staticmethod
    def _technical_score(df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        weight_acc = 0.0

        # ---- 4.1 SMA alignment -------------------------------------------#
        sma_cols = ["sma_20d", "sma_50d", "sma_200d"]
        if {"price"} <= set(df.columns):
            sma_sub = [c for c in sma_cols if c in df]
            if sma_sub:
                above_counts = sum(df["price"] > df[c] for c in sma_sub)
                sma_score = 50 + (above_counts / len(sma_sub)) * 30  # 50-80
                # Penalty if flagged "trading_under"
                if "trading_under" in df:
                    sma_score[df["trading_under"].notna() & (df["trading_under"] != "")] -= 20
                score += sma_score * 0.4
                weight_acc += 0.4

        # ---- 4.2 52-week position ----------------------------------------#
        if "position_52w" in df:
            pos_score = pd.Series(50.0, index=df.index)
            for rng, s in [
                (POSITION_52W_RANGES["NEAR_HIGH"], 75),
                (POSITION_52W_RANGES["UPPER"], 65),
                (POSITION_52W_RANGES["MIDDLE_HIGH"], 55),
                (POSITION_52W_RANGES["MIDDLE_LOW"], 50),
                (POSITION_52W_RANGES["LOWER"], 40),
                (POSITION_52W_RANGES["NEAR_LOW"], 35),
            ]:
                lo, hi = rng
                pos_score[(df["position_52w"] >= lo) & (df["position_52w"] < hi)] = s
            score += pos_score * 0.3
            weight_acc += 0.3

        # ---- 4.3 Distance from highs/lows --------------------------------#
        if {"from_high_pct", "from_low_pct"} <= set(df.columns):
            pos_adj = pd.Series(50.0, index=df.index)
            pos_adj[(df["from_high_pct"] > -10)] = 60
            pos_adj[(df["from_high_pct"] < -50)] = 40
            pos_adj[(df["from_low_pct"] < 20)] = 40
            pos_adj[(df["from_low_pct"] > 30) & (df["from_high_pct"] > -30)] = 70
            score += pos_adj * 0.3
            weight_acc += 0.3

        if weight_acc == 0:
            return pd.Series(50.0, index=df.index)

        return (score / weight_acc).clip(0, 100).round(1)

    # ======================================================================#
    #  5. Volume
    # ======================================================================#
    @staticmethod
    def _volume_score(df: pd.DataFrame) -> pd.Series:
        base = pd.Series(50.0, index=df.index)

        # ---- 5.1 RVOL -----------------------------------------------------#
        if "rvol" in df:
            rvol = df["rvol"].fillna(1.0)
            rvol_score = pd.Series(50.0, index=df.index)
            rvol_score[rvol >= VOLUME_THRESHOLDS["SPIKE"]] = 90
            rvol_score[rvol >= VOLUME_THRESHOLDS["HIGH"]] = 75
            rvol_score[rvol >= VOLUME_THRESHOLDS["ELEVATED"]] = 65
            rvol_score[rvol < VOLUME_THRESHOLDS["LOW"]] = 20
            base = rvol_score * 0.5

        # ---- 5.2 Ratio spikes --------------------------------------------#
        ratio_score = pd.Series(0.0, index=df.index)
        ratio_weights = {"vol_ratio_1d_90d": 0.4, "vol_ratio_7d_90d": 0.3, "vol_ratio_30d_90d": 0.3}
        used_weight = 0.0

        for col, w in ratio_weights.items():
            if col not in df:
                continue
            ratio = df[col].fillna(100)
            sc = pd.Series(50.0, index=df.index)
            sc[ratio >= VOLUME_RATIO_THRESHOLDS["SURGE"]] = 85
            sc[ratio >= VOLUME_RATIO_THRESHOLDS["INCREASING"]] = 70
            sc[ratio <= VOLUME_RATIO_THRESHOLDS["DECREASING"]] = 40
            sc[ratio < VOLUME_RATIO_THRESHOLDS["DRYING"]] = 30
            ratio_score += sc * w
            used_weight += w

        if used_weight:
            base += (ratio_score - 50) * 0.5

        # ---- 5.3 Spike + price action bonus ------------------------------#
        if {"rvol", "ret_1d"} <= set(df.columns):
            base += ((df["rvol"] > 2) & (df["ret_1d"] > 1)) * 10

        return base.clip(0, 100).round(1)

    # ======================================================================#
    #  6. Fundamentals (quality/profitability)
    # ======================================================================#
    @staticmethod
    def _fundamental_score(df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        used = 0.0

        # EPS trend
        if "eps_change_pct" in df:
            eps = df["eps_change_pct"].fillna(0)
            sc = pd.Series(50.0, index=df.index)
            sc[eps > 30] = 80
            sc[eps > 10] = 65
            sc[eps < -20] = 30
            score += sc * 0.25
            used += 0.25

        # Market-cap quality
        if "market_cap" in df:
            mc = df["market_cap"].fillna(0)
            sc = pd.Series(50.0, index=df.index)
            sc[mc > 1e12] = 75
            sc[mc > 2e11] = 70
            sc[mc > 5e10] = 60
            sc[mc > 5e9] = 50
            sc[mc <= 5e9] = 40
            score += sc * 0.25
            used += 0.25

        # Category quality
        if "category" in df:
            sc = pd.Series(50.0, index=df.index)
            sc[df["category"] == "Large Cap"] = 70
            sc[df["category"] == "Mid Cap"] = 60
            sc[df["category"] == "Small Cap"] = 50
            sc[df["category"] == "Micro Cap"] = 40
            score += sc * 0.25
            used += 0.25

        # Profitability via PE
        if "pe" in df:
            pe = df["pe"].fillna(0)
            sc = pd.Series(50.0, index=df.index)
            sc[(pe > 0) & (pe < 30)] = 75
            sc[(pe >= 30) & (pe < 50)] = 60
            sc[pe <= 0] = 30
            score += sc * 0.25
            used += 0.25

        if used == 0:
            return pd.Series(50.0, index=df.index)

        return (score / used).clip(0, 100).round(1)

    # ======================================================================#
    #  7. Sector integration
    # ======================================================================#
    @staticmethod
    def _add_sector_strength(df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        if sector_df.empty or "sector" not in df:
            df["sector_score"] = 50.0
            return df

        if "sector_ret_30d" not in sector_df:
            df["sector_score"] = 50.0
            return df

        sector_map = sector_df.set_index("sector")["sector_ret_30d"].to_dict()
        perf = df["sector"].map(sector_map).fillna(0)

        if perf.std() > 0:
            df["sector_score"] = (50 + (perf - perf.mean()) / perf.std() * 20).clip(0, 100)
        else:
            df["sector_score"] = 50.0

        return df

    # ======================================================================#
    #  8. Composite score
    # ======================================================================#
    @staticmethod
    def _composite_score(df: pd.DataFrame) -> pd.Series:
        factors = {
            "momentum": "momentum_score",
            "value": "value_score",
            "technical": "technical_score",
            "volume": "volume_score",
            "fundamentals": "fundamental_score",
        }

        comp = pd.Series(0.0, index=df.index)
        w_sum = 0.0

        for fac, col in factors.items():
            if col not in df or fac not in FACTOR_WEIGHTS:
                continue
            w = FACTOR_WEIGHTS[fac]
            comp += df[col].fillna(50) * w
            w_sum += w

        if w_sum == 0:
            comp[:] = 50.0
        else:
            comp /= w_sum

        # Sector bonus ±5
        if "sector_score" in df:
            comp += (df["sector_score"] - 50) * 0.1

        return comp.clip(0, 100).round(1)

    # ======================================================================#
    #  9. Decision helpers
    # ======================================================================#
    @staticmethod
    def _decision_from_score(score: float) -> str:
        if score >= SIGNAL_LEVELS["STRONG_BUY"]:
            return "STRONG_BUY"
        if score >= SIGNAL_LEVELS["BUY"]:
            return "BUY"
        if score >= SIGNAL_LEVELS["WATCH"]:
            return "WATCH"
        if score >= SIGNAL_LEVELS["NEUTRAL"]:
            return "NEUTRAL"
        return "AVOID"

    # ======================================================================#
    # 10. Risk
    # ======================================================================#
    @staticmethod
    def _risk_score(df: pd.DataFrame) -> pd.Series:
        risk = pd.Series(0.0, index=df.index)

        # Volatility
        vol_cols = {"ret_1d", "ret_7d", "ret_30d"}
        if vol_cols <= set(df.columns):
            std = df[list(vol_cols)].std(axis=1)
            risk += (std / 10 * 20).clip(0, 20)

        # Factor-based risks
        if "pe" in df:
            risk += (df["pe"] > 40) * RISK_FACTORS["high_pe"]
            risk += (df["pe"] <= 0) * RISK_FACTORS["no_profit"]
        if "volume_1d" in df:
            risk += (df["volume_1d"] < 50_000) * RISK_FACTORS["low_volume"]
        if "price" in df:
            risk += (df["price"] < 50) * RISK_FACTORS["penny_stock"]
        if "position_52w" in df:
            risk += (df["position_52w"] < 10) * RISK_FACTORS["near_52w_low"]
        if "eps_current" in df:
            risk += (df["eps_current"] < 0) * RISK_FACTORS["negative_eps"]

        return risk.clip(0, 100).round(1)

    @staticmethod
    def _risk_level_from_score(score: float) -> str:
        for lvl, (lo, hi) in RISK_LEVELS.items():
            if lo <= score < hi:
                return lvl.replace("_", " ").title()
        return "Unknown"

    # ======================================================================#
    # 11. Opportunity
    # ======================================================================#
    @staticmethod
    def _opportunity_score(df: pd.DataFrame) -> pd.Series:
        s = df.get("composite_score", 50)
        r = df.get("risk_score", 50)
        m = df.get("momentum_score", 50)
        opp = s * 0.5 + (100 - r) * 0.3 + m * 0.2
        if "value_score" in df:
            opp += (df["value_score"] > 80) * 5
        return opp.clip(0, 100).round(1)

    # ======================================================================#
    # 12. Targets / stop-loss
    # ======================================================================#
    @staticmethod
    def _price_targets(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[["target_1", "target_2", "stop_loss"]] = np.nan

        for idx, row in df.iterrows():
            price = row.get("price", np.nan)
            if price <= 0 or np.isnan(price):
                continue

            score = row.get("composite_score", 50)
            risk_lvl = row.get("risk_level", "Medium")

            t1, t2 = (10, 20) if score >= 80 else (8, 15) if score >= 70 else (5, 10)
            if abs(row.get("ret_30d", 0)) > 20:
                t1 *= 1.5
                t2 *= 1.5

            stop_pct = 5 if risk_lvl in {"Very Low", "Low"} else 7 if risk_lvl == "Medium" else 10

            df.at[idx, "target_1"] = round(price * (1 + t1 / 100), 2)
            df.at[idx, "target_2"] = round(price * (1 + t2 / 100), 2)
            df.at[idx, "stop_loss"] = round(price * (1 - stop_pct / 100), 2)

        return df

    # ======================================================================#
    # 13. Human-readable reasoning
    # ======================================================================#
    @staticmethod
    def _reasoning(row: pd.Series) -> str:
        reasons = []

        # Composite score
        cs = row.get("composite_score", 50)
        if cs >= 85:
            reasons.append("Excellent overall score")
        elif cs >= 75:
            reasons.append("Strong fundamentals")
        elif cs < 40:
            reasons.append("Weak indicators")

        # Momentum
        ms = row.get("momentum_score", 50)
        if ms > 80:
            reasons.append("Strong momentum")
        elif ms < 30:
            reasons.append("Poor momentum")

        # Valuation
        if row.get("value_score", 50) > 80:
            reasons.append("Attractive valuation")

        # Volume
        if row.get("volume_score", 50) > 80:
            reasons.append("High volume activity")

        # Special flags
        if row.get("rvol", 1) > 3:
            reasons.append("Volume spike")
        pos52 = row.get("position_52w", 50)
        if pos52 > 85:
            reasons.append("Near 52-week high")
        elif pos52 < 15:
            reasons.append("Near 52-week low")

        # Risk
        rl = row.get("risk_level", "")
        if rl in {"High", "Very High"}:
            reasons.append(f"{rl} risk")

        return " | ".join(reasons[:3]) if reasons else "Mixed signals"
