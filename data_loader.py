"""
data_loader.py - M.A.N.T.R.A. Data Loading Module
=================================================
Production-ready data loading with comprehensive error handling.
Optimized for Streamlit Cloud deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import re
import warnings
from typing import Tuple, Dict, List, Any, Optional
from datetime import datetime

from constants import (
    SHEET_URLS, CACHE_TTL, REQUEST_TIMEOUT, MAX_RETRIES,
    ERROR_MESSAGES, SUCCESS_MESSAGES, VALIDATION_RULES
)

warnings.filterwarnings('ignore')

# ============================================================================
# CORE DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_sheet_data(sheet_name: str) -> pd.DataFrame:
    """
    Load data from Google Sheets with comprehensive error handling.
    
    Args:
        sheet_name: Name of the sheet to load ('watchlist', 'sector', 'returns')
        
    Returns:
        pd.DataFrame: Cleaned dataframe or empty dataframe on error
    """
    try:
        if sheet_name not in SHEET_URLS:
            st.error(f"Unknown sheet: {sheet_name}")
            return pd.DataFrame()
        
        url = SHEET_URLS[sheet_name]
        
        # Create session with proper headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Retry logic
        for attempt in range(MAX_RETRIES):
            try:
                response = session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                continue
        
        # Parse CSV with error handling
        try:
            df = pd.read_csv(io.StringIO(response.text))
        except pd.errors.EmptyDataError:
            st.warning(f"Empty data received for {sheet_name}")
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            st.error(f"Failed to parse {sheet_name}: {str(e)}")
            return pd.DataFrame()
        
        # Clean the dataframe
        df = clean_dataframe(df)
        
        # Validate data
        if df.empty:
            st.warning(f"No data found in {sheet_name}")
            return pd.DataFrame()
        
        return df
        
    except requests.exceptions.Timeout:
        st.error(ERROR_MESSAGES['network_timeout'])
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"{ERROR_MESSAGES['sheet_not_found']} Error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error loading {sheet_name}: {str(e)}")
        return pd.DataFrame()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize dataframe with comprehensive error handling.
    
    Args:
        df: Raw dataframe from CSV
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if df.empty:
        return df
    
    try:
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
        
        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Clean string data
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = clean_string_column(df[col])
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.warning(f"Error cleaning dataframe: {str(e)}")
        return df

def clean_column_name(col_name: str) -> str:
    """Clean individual column name."""
    try:
        # Convert to string and strip
        name = str(col_name).strip()
        
        # Replace spaces and special characters
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'[^\w]', '', name)
        
        # Convert to lowercase
        name = name.lower()
        
        # Handle empty names
        if not name:
            name = 'unnamed_column'
        
        return name
    except:
        return 'unnamed_column'

def clean_string_column(series: pd.Series) -> pd.Series:
    """Clean string column data."""
    try:
        # Convert to string
        cleaned = series.astype(str)
        
        # Remove extra whitespace
        cleaned = cleaned.str.strip()
        
        # Replace various null representations
        null_values = ['nan', 'NaN', 'None', 'NULL', 'null', '', ' ']
        cleaned = cleaned.replace(null_values, np.nan)
        
        return cleaned
    except:
        return series

def clean_numeric_column(series: pd.Series, column_name: str = "") -> pd.Series:
    """
    Clean numeric column with Indian market specific handling.
    
    Args:
        series: Pandas series to clean
        column_name: Name of column for context-specific cleaning
        
    Returns:
        pd.Series: Cleaned numeric series
    """
    try:
        # If already numeric, return as-is
        if pd.api.types.is_numeric_dtype(series):
            return series
        
        # Convert to string for cleaning
        cleaned = series.astype(str)
        
        # Remove Indian market specific symbols
        symbols_to_remove = ['₹', 'Rs', 'Cr', 'Lakh', 'L', 'K', 'M', 'B', '%', ',', '↑', '↓', '+']
        for symbol in symbols_to_remove:
            cleaned = cleaned.str.replace(symbol, '', regex=False)
        
        # Remove extra spaces
        cleaned = cleaned.str.strip()
        
        # Handle empty strings and null values
        cleaned = cleaned.replace(['', 'nan', 'NaN', 'None', 'null'], np.nan)
        
        # Convert to numeric
        numeric_series = pd.to_numeric(cleaned, errors='coerce')
        
        # Handle specific column types
        if 'market_cap' in column_name.lower():
            # Handle crore scaling
            if series.astype(str).str.contains('Cr', case=False, na=False).any():
                numeric_series = numeric_series * 1e7
        
        elif any(term in column_name.lower() for term in ['ret_', 'change', 'pct']):
            # Ensure percentages are in proper range
            numeric_series = numeric_series.clip(-100, 1000)
        
        elif 'price' in column_name.lower():
            # Validate price range
            numeric_series = numeric_series.clip(VALIDATION_RULES['min_price'], 
                                                VALIDATION_RULES['max_price'])
        
        elif 'pe' in column_name.lower():
            # Validate PE ratio
            numeric_series = numeric_series.clip(0, VALIDATION_RULES['max_pe'])
        
        return numeric_series
        
    except Exception as e:
        st.warning(f"Error cleaning numeric column {column_name}: {str(e)}")
        return pd.to_numeric(series, errors='coerce')

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Process all numeric columns in dataframe."""
    if df.empty:
        return df
    
    try:
        # Identify numeric columns by pattern
        numeric_patterns = [
            r'.*price.*', r'.*ret_.*', r'.*volume.*', r'.*pe.*', r'.*eps.*',
            r'.*sma_.*', r'.*ratio.*', r'.*pct.*', r'.*market_cap.*', 
            r'.*rvol.*', r'.*low_.*', r'.*high_.*', r'.*from_.*'
        ]
        
        for col in df.columns:
            # Check if column matches numeric patterns
            is_numeric_pattern = any(re.match(pattern, col, re.IGNORECASE) 
                                   for pattern in numeric_patterns)
            
            if is_numeric_pattern:
                df[col] = clean_numeric_column(df[col], col)
        
        return df
        
    except Exception as e:
        st.warning(f"Error processing numeric columns: {str(e)}")
        return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features with error handling."""
    if df.empty:
        return df
    
    try:
        # Calculate 52-week position
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            with np.errstate(divide='ignore', invalid='ignore'):
                df['position_52w'] = (
                    (df['price'] - df['low_52w']) / 
                    (df['high_52w'] - df['low_52w']) * 100
                ).fillna(50).clip(0, 100)
        
        # Price above moving averages
        sma_columns = ['sma_20d', 'sma_50d', 'sma_200d']
        for sma_col in sma_columns:
            if sma_col in df.columns and 'price' in df.columns:
                indicator_name = f"above_{sma_col.replace('d', '')}"
                df[indicator_name] = (df['price'] > df[sma_col]).fillna(False)
        
        # Volume indicators
        if 'rvol' in df.columns:
            df['volume_spike'] = (df['rvol'] > 2.0).fillna(False)
            df['high_volume'] = (df['rvol'] > 1.5).fillna(False)
        
        # Value indicators
        if all(col in df.columns for col in ['pe', 'eps_change_pct']):
            df['undervalued'] = ((df['pe'] < 20) & (df['eps_change_pct'] > 10)).fillna(False)
        
        # Momentum indicators
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            df['uptrend'] = ((df['ret_7d'] > 0) & (df['ret_30d'] > 0)).fillna(False)
        
        # Technical indicators
        if 'position_52w' in df.columns:
            df['near_high'] = (df['position_52w'] > 90).fillna(False)
            df['near_low'] = (df['position_52w'] < 10).fillna(False)
        
        return df
        
    except Exception as e:
        st.warning(f"Error adding derived features: {str(e)}")
        return df

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data quality assessment."""
    if df.empty:
        return {
            'score': 0,
            'status': 'No Data',
            'issues': ['Empty dataset'],
            'rows': 0,
            'columns': 0
        }
    
    try:
        issues = []
        
        # Basic metrics
        total_rows = len(df)
        total_columns = len(df.columns)
        total_cells = total_rows * total_columns
        
        # Null analysis
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        null_percentage = (total_nulls / total_cells) * 100 if total_cells > 0 else 100
        
        # Critical columns check
        critical_columns = ['ticker', 'price', 'sector']
        missing_critical = []
        
        for col in critical_columns:
            if col not in df.columns:
                missing_critical.append(f"Missing {col}")
            elif df[col].isnull().all():
                missing_critical.append(f"Empty {col}")
            elif df[col].isnull().sum() / len(df) > 0.5:
                missing_critical.append(f"Sparse {col}")
        
        if missing_critical:
            issues.extend(missing_critical)
        
        # Data validation checks
        if 'price' in df.columns:
            invalid_prices = ((df['price'] <= 0) | (df['price'] > VALIDATION_RULES['max_price'])).sum()
            if invalid_prices > 0:
                issues.append(f"Invalid prices: {invalid_prices}")
        
        if 'volume_1d' in df.columns:
            zero_volume = (df['volume_1d'] == 0).sum()
            if zero_volume > total_rows * 0.3:
                issues.append(f"High zero volume: {zero_volume}")
        
        # Calculate overall quality score
        score = 100
        score -= min(50, null_percentage)  # Penalize missing data
        score -= min(30, len(issues) * 5)  # Penalize issues
        score -= min(20, len(missing_critical) * 10)  # Penalize critical issues
        
        score = max(0, score)
        
        # Determine status
        if score >= 90:
            status = 'Excellent'
        elif score >= 75:
            status = 'Good'
        elif score >= 60:
            status = 'Fair'
        else:
            status = 'Poor'
        
        return {
            'score': round(score, 1),
            'status': status,
            'issues': issues,
            'rows': total_rows,
            'columns': total_columns,
            'null_percentage': round(null_percentage, 1),
            'total_nulls': int(total_nulls)
        }
        
    except Exception as e:
        return {
            'score': 0,
            'status': 'Error',
            'issues': [f"Quality check failed: {str(e)}"],
            'rows': len(df) if not df.empty else 0,
            'columns': len(df.columns) if not df.empty else 0
        }

# ============================================================================
# MAIN LOADING FUNCTION
# ============================================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load and process all data sheets.
    
    Returns:
        Tuple containing:
        - watchlist_df: Main stock data
        - sector_df: Sector performance data
        - status: Loading status and quality metrics
    """
    status = {
        'success': False,
        'errors': [],
        'warnings': [],
        'quality': {},
        'load_time': None
    }
    
    start_time = datetime.now()
    
    try:
        # Load all sheets
        watchlist_df = load_sheet_data('watchlist')
        sector_df = load_sheet_data('sector')
        
        # Check if main data loaded
        if watchlist_df.empty:
            status['errors'].append("Failed to load watchlist data")
            return pd.DataFrame(), pd.DataFrame(), status
        
        # Process numeric columns
        watchlist_df = process_numeric_columns(watchlist_df)
        if not sector_df.empty:
            sector_df = process_numeric_columns(sector_df)
        
        # Add derived features
        watchlist_df = add_derived_features(watchlist_df)
        
        # Clean ticker column
        if 'ticker' in watchlist_df.columns:
            watchlist_df['ticker'] = watchlist_df['ticker'].astype(str).str.upper().str.strip()
            # Remove invalid tickers
            valid_tickers = (watchlist_df['ticker'] != 'NAN') & (watchlist_df['ticker'].str.len() > 0)
            watchlist_df = watchlist_df[valid_tickers].reset_index(drop=True)
        
        # Quality assessment
        quality = validate_data_quality(watchlist_df)
        status['quality'] = quality
        
        # Check if quality meets minimum standards
        if quality['score'] < VALIDATION_RULES['min_data_quality']:
            status['warnings'].append(f"Data quality below threshold: {quality['score']}")
        
        # Calculate load time
        load_time = (datetime.now() - start_time).total_seconds()
        status['load_time'] = round(load_time, 2)
        
        # Success
        status['success'] = True
        
        return watchlist_df, sector_df, status
        
    except Exception as e:
        status['errors'].append(f"Critical error: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), status

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for loaded data."""
    if df.empty:
        return {}
    
    try:
        summary = {
            'total_stocks': len(df),
            'columns': len(df.columns),
            'sectors': df['sector'].nunique() if 'sector' in df.columns else 0,
            'price_range': {
                'min': df['price'].min() if 'price' in df.columns else 0,
                'max': df['price'].max() if 'price' in df.columns else 0
            },
            'data_types': df.dtypes.value_counts().to_dict()
        }
        return summary
    except:
        return {}

def format_load_status(status: Dict[str, Any]) -> str:
    """Format loading status for display."""
    if not status['success']:
        return f"❌ Loading failed: {'; '.join(status['errors'])}"
    
    quality = status.get('quality', {})
    load_time = status.get('load_time', 0)
    
    return f"✅ Loaded successfully in {load_time}s | Quality: {quality.get('status', 'Unknown')}"
