"""
data_loader.py - M.A.N.T.R.A. Data Foundation
============================================
FINAL PRODUCTION VERSION - Fast, reliable, bug-free
Optimized for Streamlit Community Cloud
"""

import pandas as pd
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Optional
import logging
from datetime import datetime
import warnings

from constants import (
    GOOGLE_SHEET_ID, SHEET_CONFIGS, CACHE_DURATION_MINUTES,
    DATA_QUALITY_THRESHOLDS, REQUIRED_WATCHLIST_COLUMNS
)

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """High-performance data loader for Google Sheets"""
    
    @staticmethod
    @st.cache_data(ttl=CACHE_DURATION_MINUTES*60, show_spinner=False)
    def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load all data with parallel processing and optimized cleaning
        Returns: (stocks_df, sector_df, health_report)
        """
        start_time = datetime.now()
        health = {
            'status': 'loading',
            'timestamp': start_time,
            'errors': [],
            'warnings': [],
            'load_time_seconds': 0
        }
        
        try:
            # Build URLs for all sheets
            urls = {
                'watchlist': f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['watchlist']['gid']}",
                'returns': f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS.get('returns', {}).get('gid', '100734077')}",
                'sector': f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['sector']['gid']}"
            }
            
            # PARALLEL LOADING - Much faster!
            dataframes = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_sheet = {
                    executor.submit(DataLoader._load_sheet, sheet_name, url): sheet_name 
                    for sheet_name, url in urls.items()
                }
                
                for future in as_completed(future_to_sheet):
                    sheet_name = future_to_sheet[future]
                    try:
                        dataframes[sheet_name] = future.result()
                        logger.info(f"Loaded {sheet_name}: {len(dataframes[sheet_name])} rows")
                    except Exception as e:
                        logger.error(f"Failed to load {sheet_name}: {e}")
                        health['errors'].append(f"{sheet_name}: {str(e)}")
            
            # Check if we have minimum required data
            if 'watchlist' not in dataframes or dataframes['watchlist'].empty:
                raise ValueError("Failed to load watchlist data")
            
            if 'sector' not in dataframes:
                health['warnings'].append("No sector data available")
                dataframes['sector'] = pd.DataFrame()
            
            # FAST CLEANING
            watchlist_df = DataLoader._fast_clean_watchlist(dataframes['watchlist'])
            sector_df = DataLoader._fast_clean_sectors(dataframes.get('sector', pd.DataFrame()))
            
            # Merge with returns data if available
            if 'returns' in dataframes and not dataframes['returns'].empty:
                returns_df = DataLoader._fast_clean_returns(dataframes['returns'])
                stocks_df = DataLoader._merge_data(watchlist_df, returns_df)
            else:
                stocks_df = watchlist_df
                health['warnings'].append("No returns data - using watchlist only")
            
            # Add calculated fields
            stocks_df = DataLoader._add_calculated_fields(stocks_df)
            
            # Validate data quality
            validation_warnings = DataLoader._validate_data(stocks_df, sector_df)
            health['warnings'].extend(validation_warnings)
            
            # Calculate final metrics
            health['status'] = 'success'
            health['stocks_count'] = len(stocks_df)
            health['sectors_count'] = len(sector_df)
            health['data_quality'] = DataLoader._calculate_data_quality(stocks_df)
            health['load_time_seconds'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Data loaded successfully in {health['load_time_seconds']:.1f}s")
            
            return stocks_df, sector_df, health
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            health['status'] = 'error'
            health['errors'].append(str(e))
            return pd.DataFrame(), pd.DataFrame(), health
    
    @staticmethod
    def _load_sheet(sheet_name: str, url: str) -> pd.DataFrame:
        """Load a single sheet with error handling"""
        try:
            df = pd.read_csv(url, low_memory=False)
            # Basic cleaning
            df = df.dropna(how='all')
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            logger.error(f"Error loading {sheet_name}: {e}")
            raise
    
    @staticmethod
    def _fast_clean_watchlist(df: pd.DataFrame) -> pd.DataFrame:
        """Fast vectorized cleaning for watchlist data"""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Ensure ticker is clean
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
            df = df[df['ticker'] != 'NAN']  # Remove invalid tickers
        
        # FAST NUMERIC CONVERSION - All at once
        # Price columns
        price_cols = ['price', 'prev_close', 'low_52w', 'high_52w', 
                      'sma_20d', 'sma_50d', 'sma_200d']
        
        # Percentage columns  
        pct_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 
                    'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                    'from_low_pct', 'from_high_pct', 'eps_change_pct',
                    'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
        
        # Volume columns
        vol_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_3m']
        
        # Other numeric columns
        other_cols = ['pe', 'eps_current', 'eps_last_qtr', 'rvol', 'year']
        
        # Convert all numeric columns in one pass
        all_numeric = price_cols + pct_cols + vol_cols + other_cols
        
        for col in all_numeric:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Remove symbols and convert
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace('[₹,%,]', '', regex=True),
                        errors='coerce'
                    )
        
        # FAST MARKET CAP PARSING
        if 'market_cap' in df.columns:
            df['market_cap'] = DataLoader._vectorized_parse_market_cap(df['market_cap'])
        
        # Handle categorical columns
        if 'category' in df.columns:
            df['category'] = df['category'].astype(str).str.strip()
        
        if 'sector' in df.columns:
            df['sector'] = df['sector'].astype(str).str.strip()
        
        if 'eps_tier' in df.columns:
            df['eps_tier'] = df['eps_tier'].astype(str).str.strip()
        
        if 'price_tier' in df.columns:
            df['price_tier'] = df['price_tier'].astype(str).str.strip()
        
        return df
    
    @staticmethod
    def _fast_clean_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Fast cleaning for returns data"""
        # Clean ticker
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        
        # Clean all return columns at once
        return_cols = [col for col in df.columns if 'ret_' in col or 'avg_' in col]
        
        for col in return_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(
                    df[col].str.replace('%', ''),
                    errors='coerce'
                )
        
        return df
    
    @staticmethod
    def _fast_clean_sectors(df: pd.DataFrame) -> pd.DataFrame:
        """Fast cleaning for sector data"""
        if df.empty:
            return df
        
        # Clean all percentage columns
        for col in df.columns:
            if ('ret_' in col or 'avg_' in col) and df[col].dtype == 'object':
                df[col] = pd.to_numeric(
                    df[col].str.replace('%', ''),
                    errors='coerce'
                )
        
        # Ensure sector_count is numeric
        if 'sector_count' in df.columns:
            df['sector_count'] = pd.to_numeric(df['sector_count'], errors='coerce')
        
        return df
    
    @staticmethod
    def _vectorized_parse_market_cap(series: pd.Series) -> pd.Series:
        """Vectorized market cap parsing for speed"""
        if series.dtype != 'object':
            return series
        
        # Convert to string and uppercase
        s = series.astype(str).str.upper().str.strip()
        
        # Initialize result
        result = pd.Series(np.nan, index=series.index)
        
        # Handle Crores (most common)
        cr_mask = s.str.contains('CR|CRORE', na=False)
        if cr_mask.any():
            cr_values = s[cr_mask].str.extract(r'([\d.]+)', expand=False)
            result[cr_mask] = pd.to_numeric(cr_values, errors='coerce') * 1e7
        
        # Handle Lakhs
        lakh_mask = s.str.contains('L|LAC|LAKH', na=False) & ~cr_mask
        if lakh_mask.any():
            lakh_values = s[lakh_mask].str.extract(r'([\d.]+)', expand=False)
            result[lakh_mask] = pd.to_numeric(lakh_values, errors='coerce') * 1e5
        
        # Handle direct numbers
        direct_mask = ~cr_mask & ~lakh_mask & s.str.match(r'^[\d.]+$', na=False)
        if direct_mask.any():
            result[direct_mask] = pd.to_numeric(s[direct_mask], errors='coerce')
        
        return result
    
    @staticmethod
    def _merge_data(watchlist_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Merge watchlist and returns data efficiently"""
        # Only keep non-duplicate columns from returns
        returns_cols = ['ticker'] + [col for col in returns_df.columns 
                                    if col not in watchlist_df.columns or col == 'ticker']
        
        # Merge on ticker
        merged = pd.merge(
            watchlist_df,
            returns_df[returns_cols],
            on='ticker',
            how='left',
            suffixes=('', '_returns')
        )
        
        # Remove any duplicates
        merged = merged.drop_duplicates(subset=['ticker'], keep='first')
        
        return merged
    
    @staticmethod
    def _add_calculated_fields(df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields using vectorized operations"""
        # Position in 52-week range
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            range_52w = df['high_52w'] - df['low_52w']
            df['position_52w'] = np.where(
                range_52w > 0,
                ((df['price'] - df['low_52w']) / range_52w * 100).round(2),
                50.0
            )
        
        # Distance from SMAs (vectorized)
        if 'price' in df.columns:
            for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
                if sma in df.columns:
                    df[f'distance_from_{sma}'] = (
                        (df['price'] - df[sma]) / df[sma] * 100
                    ).round(2)
        
        # Volume spike indicator
        if 'rvol' in df.columns:
            df['volume_spike'] = df['rvol'] > 2.0
        
        # Value stock indicator
        if 'pe' in df.columns:
            df['is_value_stock'] = (df['pe'] > 0) & (df['pe'] < 20)
        
        # Momentum indicator
        if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
            df['has_momentum'] = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
        
        # Trend strength from averages
        if 'avg_ret_30d' in df.columns:
            df['trend_strength'] = df['avg_ret_30d']
        elif 'ret_30d' in df.columns:
            df['trend_strength'] = df['ret_30d']
        
        # Market cap in billions for display
        if 'market_cap' in df.columns:
            df['market_cap_b'] = (df['market_cap'] / 1e9).round(2)
        
        return df
    
    @staticmethod
    def _validate_data(stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> list:
        """Validate data quality and return warnings"""
        warnings = []
        
        # Check for required columns
        missing_cols = set(REQUIRED_WATCHLIST_COLUMNS) - set(stocks_df.columns)
        if missing_cols:
            warnings.append(f"Missing columns: {', '.join(missing_cols)}")
        
        # Check data size
        if len(stocks_df) < DATA_QUALITY_THRESHOLDS['MIN_ROWS']:
            warnings.append(f"Low stock count: {len(stocks_df)}")
        
        # Check for invalid prices
        if 'price' in stocks_df.columns:
            invalid_prices = (
                (stocks_df['price'] < DATA_QUALITY_THRESHOLDS['MIN_PRICE']) |
                (stocks_df['price'] > DATA_QUALITY_THRESHOLDS['MAX_PRICE']) |
                stocks_df['price'].isna()
            ).sum()
            
            if invalid_prices > 0:
                warnings.append(f"{invalid_prices} stocks have invalid prices")
        
        # Check null percentage
        null_pct = stocks_df.isna().sum().sum() / (len(stocks_df) * len(stocks_df.columns)) * 100
        if null_pct > DATA_QUALITY_THRESHOLDS['MAX_NULL_PERCENT']:
            warnings.append(f"High null percentage: {null_pct:.1f}%")
        
        # Check sector coverage
        if not sector_df.empty and 'sector' in stocks_df.columns:
            stock_sectors = set(stocks_df['sector'].dropna().unique())
            sector_list = set(sector_df['sector'].unique()) if 'sector' in sector_df.columns else set()
            missing_sectors = stock_sectors - sector_list
            
            if missing_sectors:
                warnings.append(f"Sectors without data: {len(missing_sectors)}")
        
        return warnings
    
    @staticmethod
    def _calculate_data_quality(df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        if df.empty:
            return 0.0
        
        scores = []
        
        # Critical columns completeness
        critical_cols = ['ticker', 'price', 'pe', 'sector', 'ret_30d']
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
        
        # Overall completeness
        overall_completeness = (1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        scores.append(overall_completeness)
        
        return round(np.mean(scores), 1) if scores else 0.0

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Main function to load all data"""
    return DataLoader.load_all_data()

def get_data_summary(stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> Dict:
    """Get summary statistics of loaded data"""
    if stocks_df.empty:
        return {'error': 'No data loaded'}
    
    summary = {
        'total_stocks': len(stocks_df),
        'total_sectors': len(sector_df),
        'unique_sectors': stocks_df['sector'].nunique() if 'sector' in stocks_df else 0,
        'avg_pe': stocks_df['pe'].mean() if 'pe' in stocks_df else None,
        'avg_market_cap_cr': (stocks_df['market_cap'].mean() / 1e7) if 'market_cap' in stocks_df else None,
        'data_coverage': {
            'with_price': stocks_df['price'].notna().sum() if 'price' in stocks_df else 0,
            'with_pe': stocks_df['pe'].notna().sum() if 'pe' in stocks_df else 0,
            'with_returns': stocks_df['ret_30d'].notna().sum() if 'ret_30d' in stocks_df else 0,
        }
    }
    
    return summary

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Data Loader - Testing")
    print("="*60)
    
    print("\nLoading data...")
    stocks_df, sector_df, health = load_data()
    
    if health['status'] == 'success':
        print(f"\n✅ Success! Loaded in {health['load_time_seconds']:.1f} seconds")
        print(f"Stocks: {health['stocks_count']}")
        print(f"Sectors: {health['sectors_count']}")
        print(f"Data Quality: {health['data_quality']}%")
        
        if health['warnings']:
            print("\n⚠️ Warnings:")
            for warning in health['warnings']:
                print(f"  - {warning}")
        
        # Show sample data
        print("\nSample data:")
        print(stocks_df[['ticker', 'company_name', 'price', 'pe', 'ret_30d']].head())
        
        # Show summary
        print("\nData Summary:")
        summary = get_data_summary(stocks_df, sector_df)
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    else:
        print(f"\n❌ Failed: {health['errors']}")
    
    print("="*60)
