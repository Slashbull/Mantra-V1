"""
data_loader.py - M.A.N.T.R.A. Data Loading Engine (Final Version)
================================================================
Handles all data loading, cleaning, validation, and preprocessing
from Google Sheets. Optimized for Streamlit Cloud deployment.
"""

import pandas as pd
import numpy as np
import requests
import io
import re
import warnings
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
import time

from constants import SHEET_URLS, WATCHLIST_COLUMNS, RETURNS_COLUMNS, SECTOR_COLUMNS

warnings.filterwarnings('ignore')

class DataLoader:
    """Enhanced data loader for M.A.N.T.R.A. system"""
    
    def __init__(self):
        self.session = self._create_session()
        
    def _create_session(self):
        """Create HTTP session with retry logic"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session
    
    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Load a single sheet from Google Sheets with error handling"""
        try:
            url = SHEET_URLS[sheet_name]
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(response.text))
            
            # Clean and validate
            df = self._clean_dataframe(df)
            df = self._validate_columns(df, sheet_name)
            
            return df
            
        except Exception as e:
            print(f"Error loading {sheet_name}: {str(e)}")
            return pd.DataFrame()
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize dataframe"""
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
        
        # Clean column names - preserve exact structure
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        
        # Remove hidden characters
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.replace('\u00A0', ' ', regex=False)
            df[col] = df[col].str.strip()
        
        return df
    
    def _validate_columns(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Validate and ensure required columns exist"""
        if sheet_name == 'watchlist':
            required = WATCHLIST_COLUMNS
        elif sheet_name == 'returns':
            required = RETURNS_COLUMNS
        elif sheet_name == 'sector':
            required = SECTOR_COLUMNS
        else:
            return df
        
        # Add missing columns as NaN
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
        
        return df
    
    def _clean_numeric_column(self, series: pd.Series, col_name: str = "") -> pd.Series:
        """Advanced numeric cleaning for Indian market data"""
        if series.dtype in ['int64', 'float64']:
            return series
        
        # Convert to string for cleaning
        s = series.astype(str)
        
        # Remove currency symbols and units common in Indian data
        for symbol in ['₹', '$', '€', '£', 'Cr', 'L', 'K', 'M', 'B', '%', ',', '↑', '↓']:
            s = s.str.replace(symbol, '', regex=False)
        
        # Remove extra spaces and non-ASCII
        s = s.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()
        
        # Handle empty strings
        s = s.replace(['', 'nan', 'NaN', 'null', 'NULL'], np.nan)
        
        # Convert to numeric
        numeric_series = pd.to_numeric(s, errors='coerce')
        
        # Handle market cap scaling (Cr to actual values)
        if 'market_cap' in col_name and series.astype(str).str.contains('Cr').any():
            # Multiply by 10 million (1 Crore = 10^7)
            numeric_series = numeric_series * 1e7
        
        return numeric_series
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated features for analysis"""
        # Position in 52-week range
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            df['position_52w'] = (
                (df['price'] - df['low_52w']) / 
                (df['high_52w'] - df['low_52w']) * 100
            ).fillna(50)
        
        # Price above SMAs
        for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
            if sma in df.columns and 'price' in df.columns:
                df[f'above_{sma.replace("d", "")}'] = df['price'] > df[sma]
        
        # Volume spikes
        if 'rvol' in df.columns:
            df['volume_spike'] = (df['rvol'] > 2).astype(int)
        
        # Value indicators
        if all(col in df.columns for col in ['pe', 'eps_change_pct']):
            df['value_score'] = np.where(
                (df['pe'] < 20) & (df['eps_change_pct'] > 10), 1, 0
            )
        
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean all numeric columns in dataframe"""
        # Identify numeric columns by pattern
        numeric_patterns = [
            r'.*ret_.*', r'.*price.*', r'.*volume.*', r'.*pe.*', r'.*eps.*',
            r'.*sma_.*', r'.*ratio.*', r'.*pct.*', r'.*market_cap.*', r'.*rvol.*',
            r'.*low_.*', r'.*high_.*', r'.*from_.*'
        ]
        
        for col in df.columns:
            for pattern in numeric_patterns:
                if re.match(pattern, col, re.IGNORECASE):
                    df[col] = self._clean_numeric_column(df[col], col)
                    break
        
        return df
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        # Downcast numeric types
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert low-cardinality strings to category
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        return df
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and generate metrics"""
        if df.empty:
            return {'score': 0, 'status': 'No Data', 'issues': ['Empty dataset']}
        
        issues = []
        
        # Critical columns completeness
        critical_cols = ['ticker', 'price', 'sector']
        missing_critical = []
        for col in critical_cols:
            if col in df.columns:
                null_pct = df[col].isnull().sum() / len(df) * 100
                if null_pct > 20:
                    missing_critical.append(f"{col}: {null_pct:.1f}% missing")
            else:
                missing_critical.append(f"{col}: Column missing")
        
        if missing_critical:
            issues.extend(missing_critical)
        
        # Price validation
        if 'price' in df.columns:
            invalid_prices = (df['price'] <= 0) | (df['price'] > 1000000)
            if invalid_prices.any():
                issues.append(f"Invalid prices: {invalid_prices.sum()} stocks")
        
        # Calculate overall score
        score = 100
        score -= min(50, len(issues) * 10)
        score -= min(30, df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        
        if score >= 90:
            status = 'Excellent'
        elif score >= 75:
            status = 'Good'
        elif score >= 60:
            status = 'Fair'
        else:
            status = 'Poor'
        
        return {
            'score': max(0, score),
            'status': status,
            'issues': issues,
            'rows': len(df),
            'columns': len(df.columns)
        }

def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load all data sheets and return processed dataframes
    
    Returns:
        watchlist_df: Main stock data
        returns_df: Returns data (for merging)  
        sector_df: Sector analysis data
        status: Loading status and quality metrics
    """
    loader = DataLoader()
    status = {'success': False, 'errors': [], 'quality': {}}
    
    try:
        # Load all sheets
        print("Loading M.A.N.T.R.A. data...")
        watchlist_df = loader.load_sheet('watchlist')
        returns_df = loader.load_sheet('returns')
        sector_df = loader.load_sheet('sector')
        
        if watchlist_df.empty:
            status['errors'].append("Failed to load watchlist data")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), status
        
        # Clean numeric columns
        watchlist_df = loader._clean_numeric_columns(watchlist_df)
        if not returns_df.empty:
            returns_df = loader._clean_numeric_columns(returns_df)
        if not sector_df.empty:
            sector_df = loader._clean_numeric_columns(sector_df)
        
        # Merge watchlist with returns data
        if not returns_df.empty and 'ticker' in returns_df.columns:
            # Remove duplicate columns before merging
            returns_merge = returns_df.drop(columns=['company_name'], errors='ignore')
            watchlist_df = watchlist_df.merge(
                returns_merge, on='ticker', how='left', suffixes=('', '_returns')
            )
        
        # Add derived features
        watchlist_df = loader._add_derived_features(watchlist_df)
        
        # Optimize memory
        watchlist_df = loader._optimize_memory(watchlist_df)
        if not sector_df.empty:
            sector_df = loader._optimize_memory(sector_df)
        
        # Quality assessment
        quality = loader.assess_data_quality(watchlist_df)
        status['quality'] = quality
        
        # Clean ticker column
        if 'ticker' in watchlist_df.columns:
            watchlist_df['ticker'] = watchlist_df['ticker'].astype(str).str.upper().str.strip()
            # Remove invalid tickers
            watchlist_df = watchlist_df[
                (watchlist_df['ticker'] != 'NAN') & 
                (watchlist_df['ticker'].str.len() > 0)
            ]
        
        status['success'] = True
        print(f"✅ Loaded {len(watchlist_df)} stocks, quality: {quality['status']}")
        
        return watchlist_df, returns_df, sector_df, status
        
    except Exception as e:
        status['errors'].append(str(e))
        print(f"❌ Data loading failed: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), status
