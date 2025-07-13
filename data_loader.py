"""
data_loader.py - M.A.N.T.R.A. Data Foundation
============================================
Perfectly aligned with your Google Sheets structure
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Optional
import logging
from datetime import datetime
from constants import (
    GOOGLE_SHEET_ID, SHEET_CONFIGS, CACHE_DURATION_MINUTES,
    DATA_QUALITY_THRESHOLDS
)

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader perfectly matched to your sheets structure"""
    
    @staticmethod
    @st.cache_data(ttl=CACHE_DURATION_MINUTES*60, show_spinner=False)
    def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load all three sheets: Watchlist, Returns, and Sector
        Returns: (watchlist_df, returns_df, sector_df, health_report)
        """
        health = {
            'status': 'loading',
            'timestamp': datetime.now(),
            'errors': [],
            'warnings': []
        }
        
        try:
            # Build URLs for all three sheets
            watchlist_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['watchlist']['gid']}"
            sector_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['sector']['gid']}"
            
            # Add returns sheet config if not in constants
            returns_gid = SHEET_CONFIGS.get('returns', {}).get('gid', '100734077')
            returns_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={returns_gid}"
            
            # Load all dataframes
            watchlist_df = pd.read_csv(watchlist_url)
            returns_df = pd.read_csv(returns_url)
            sector_df = pd.read_csv(sector_url)
            
            # Basic cleaning
            watchlist_df = DataLoader._basic_clean(watchlist_df)
            returns_df = DataLoader._basic_clean(returns_df)
            sector_df = DataLoader._basic_clean(sector_df)
            
            # Clean each dataset according to its structure
            watchlist_df = DataLoader._clean_watchlist_data(watchlist_df)
            returns_df = DataLoader._clean_returns_data(returns_df)
            sector_df = DataLoader._clean_sector_data(sector_df)
            
            # Merge watchlist with returns data
            stocks_df = DataLoader._merge_stock_data(watchlist_df, returns_df)
            
            # Add calculated fields
            stocks_df = DataLoader._add_calculated_fields(stocks_df)
            
            # Validate data
            validation_issues = DataLoader._validate_data(stocks_df, sector_df)
            health['warnings'].extend(validation_issues)
            
            # Update health status
            health['status'] = 'success'
            health['watchlist_count'] = len(watchlist_df)
            health['returns_count'] = len(returns_df)
            health['sectors_count'] = len(sector_df)
            health['merged_stocks_count'] = len(stocks_df)
            health['data_quality'] = DataLoader._calculate_data_quality(stocks_df)
            
            return stocks_df, sector_df, health
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            health['status'] = 'error'
            health['errors'].append(str(e))
            return pd.DataFrame(), pd.DataFrame(), health
    
    @staticmethod
    def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
        """Basic dataframe cleaning"""
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        return df
    
    @staticmethod
    def _clean_watchlist_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean watchlist data matching your exact structure"""
        
        # Price columns - these are your actual price columns
        price_cols = ['price', 'prev_close', 'low_52w', 'high_52w', 
                      'sma_20d', 'sma_50d', 'sma_200d']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('₹', '').str.replace(',', ''), 
                    errors='coerce'
                )
        
        # Percentage columns - all your return and percentage columns
        pct_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 
                    'ret_1y', 'ret_3y', 'ret_5y', 'from_low_pct', 'from_high_pct',
                    'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                    'eps_change_pct']
        for col in pct_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', ''), 
                    errors='coerce'
                )
        
        # Volume columns
        vol_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_3m']
        for col in vol_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                )
        
        # Market cap - handle Cr/Lakh notations
        if 'market_cap' in df.columns:
            df['market_cap'] = df['market_cap'].apply(DataLoader._parse_market_cap)
        
        # Numeric columns
        numeric_cols = ['pe', 'eps_current', 'eps_last_qtr', 'eps_duplicate', 
                        'rvol', 'year']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure ticker is uppercase
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        
        # Clean exchange column if present
        if 'exchange' in df.columns:
            df['exchange'] = df['exchange'].astype(str).str.upper().str.strip()
        
        return df
    
    @staticmethod
    def _clean_returns_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean returns data matching your structure"""
        
        # All return columns in the returns sheet
        return_cols = ['returns_ret_1d', 'returns_ret_3d', 'returns_ret_7d', 
                       'returns_ret_30d', 'returns_ret_3m', 'returns_ret_6m',
                       'returns_ret_1y', 'returns_ret_3y', 'returns_ret_5y']
        
        # Average return columns
        avg_cols = ['avg_ret_30d', 'avg_ret_3m', 'avg_ret_6m', 
                    'avg_ret_1y', 'avg_ret_3y', 'avg_ret_5y']
        
        # Clean all percentage columns
        for col in return_cols + avg_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', ''), 
                    errors='coerce'
                )
        
        # Ensure ticker is uppercase for merging
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        
        return df
    
    @staticmethod
    def _clean_sector_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean sector data matching your structure"""
        
        # Sector return columns
        sector_return_cols = ['sector_ret_1d', 'sector_ret_3d', 'sector_ret_7d',
                              'sector_ret_30d', 'sector_ret_3m', 'sector_ret_6m',
                              'sector_ret_1y', 'sector_ret_3y', 'sector_ret_5y']
        
        # Sector average columns
        sector_avg_cols = ['sector_avg_30d', 'sector_avg_3m', 'sector_avg_6m',
                           'sector_avg_1y', 'sector_avg_3y', 'sector_avg_5y']
        
        # Clean all percentage columns
        for col in sector_return_cols + sector_avg_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', ''), 
                    errors='coerce'
                )
        
        # Ensure sector_count is numeric
        if 'sector_count' in df.columns:
            df['sector_count'] = pd.to_numeric(df['sector_count'], errors='coerce')
        
        return df
    
    @staticmethod
    def _parse_market_cap(val) -> float:
        """Parse market cap with Cr/Lakh conversion"""
        if pd.isna(val):
            return np.nan
        
        val = str(val).upper().replace('₹', '').replace(',', '').strip()
        
        # Handle Crores (1 Cr = 10 million)
        if 'CR' in val:
            number = val.replace('CR', '').replace('CRORE', '').replace('CRORES', '').strip()
            try:
                return float(number) * 1e7
            except:
                return np.nan
        
        # Handle Lakhs (1 Lakh = 100,000)
        if 'L' in val or 'LAC' in val or 'LAKH' in val:
            number = val.replace('L', '').replace('LAC', '').replace('LAKH', '').replace('LAKHS', '').strip()
            try:
                return float(number) * 1e5
            except:
                return np.nan
        
        # Try direct conversion
        try:
            return float(val)
        except:
            return np.nan
    
    @staticmethod
    def _merge_stock_data(watchlist_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Merge watchlist and returns data"""
        
        # The returns data has prefixed column names, so we'll handle that
        # First, let's see what columns we have
        if 'ticker' in watchlist_df.columns and 'ticker' in returns_df.columns:
            # Don't include company_name from returns if it's already in watchlist
            returns_cols_to_keep = [col for col in returns_df.columns 
                                   if col not in watchlist_df.columns or col == 'ticker']
            
            # Merge on ticker
            merged = pd.merge(
                watchlist_df, 
                returns_df[returns_cols_to_keep], 
                on='ticker', 
                how='left',
                suffixes=('', '_returns')
            )
            
            # Remove any duplicate tickers
            if merged['ticker'].duplicated().any():
                logger.warning(f"Found {merged['ticker'].duplicated().sum()} duplicate tickers")
                merged = merged.drop_duplicates(subset=['ticker'], keep='first')
            
            return merged
        else:
            logger.warning("Could not merge - ticker column missing")
            return watchlist_df
    
    @staticmethod
    def _add_calculated_fields(df: pd.DataFrame) -> pd.DataFrame:
        """Add useful calculated fields"""
        
        # Calculate position_52w if not present
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            range_size = df['high_52w'] - df['low_52w']
            df['position_52w'] = np.where(
                range_size > 0,
                ((df['price'] - df['low_52w']) / range_size * 100).round(2),
                50.0
            )
        
        # Distance from SMAs
        if 'price' in df.columns:
            for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
                if sma in df.columns:
                    df[f'distance_from_{sma}'] = (
                        (df['price'] - df[sma]) / df[sma] * 100
                    ).round(2)
        
        # Volume spike indicator
        if 'rvol' in df.columns:
            df['volume_spike'] = df['rvol'] > 2.0
        
        # Value indicator based on PE
        if 'pe' in df.columns:
            df['is_value_stock'] = (df['pe'] > 0) & (df['pe'] < 20)
        
        # Momentum indicator
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            df['has_momentum'] = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
        
        # Trend strength (if we have averages from returns sheet)
        if 'avg_ret_30d' in df.columns:
            df['trend_strength'] = df['avg_ret_30d']
        
        return df
    
    @staticmethod
    def _validate_data(stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> list:
        """Validate data quality and return warnings"""
        warnings = []
        
        # Check for critical columns in your structure
        critical_columns = ['ticker', 'price', 'company_name', 'sector', 'pe']
        missing_cols = set(critical_columns) - set(stocks_df.columns)
        if missing_cols:
            warnings.append(f"Missing columns: {', '.join(missing_cols)}")
        
        # Check data size
        if len(stocks_df) < DATA_QUALITY_THRESHOLDS['MIN_ROWS']:
            warnings.append(f"Low data count: {len(stocks_df)} stocks")
        
        # Check price validity
        if 'price' in stocks_df.columns:
            invalid_prices = (
                (stocks_df['price'] < DATA_QUALITY_THRESHOLDS['MIN_PRICE']) |
                (stocks_df['price'] > DATA_QUALITY_THRESHOLDS['MAX_PRICE']) |
                stocks_df['price'].isna()
            ).sum()
            if invalid_prices > 0:
                warnings.append(f"{invalid_prices} stocks have invalid prices")
        
        # Check sector data
        if sector_df.empty:
            warnings.append("No sector data available")
        elif 'sector' in stocks_df.columns:
            # Check if all sectors in stocks are in sector data
            stock_sectors = set(stocks_df['sector'].dropna().unique())
            sector_sectors = set(sector_df['sector'].unique()) if 'sector' in sector_df.columns else set()
            missing_sectors = stock_sectors - sector_sectors
            if missing_sectors:
                warnings.append(f"Sectors missing from sector sheet: {', '.join(list(missing_sectors)[:5])}")
        
        return warnings
    
    @staticmethod
    def _calculate_data_quality(df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)"""
        if df.empty:
            return 0.0
        
        scores = []
        
        # Completeness score for critical columns
        critical_cols = ['ticker', 'price', 'pe', 'sector', 'company_name']
        for col in critical_cols:
            if col in df.columns:
                completeness = (1 - df[col].isna().sum() / len(df)) * 100
                scores.append(completeness)
        
        # Price validity
        if 'price' in df.columns:
            valid_prices = (
                (df['price'] >= DATA_QUALITY_THRESHOLDS['MIN_PRICE']) &
                (df['price'] <= DATA_QUALITY_THRESHOLDS['MAX_PRICE']) &
                df['price'].notna()
            ).sum()
            validity_score = (valid_prices / len(df)) * 100
            scores.append(validity_score)
        
        return np.mean(scores) if scores else 0.0
