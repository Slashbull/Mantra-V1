"""
data.py - Optimized Data Handler for M.A.N.T.R.A.
=================================================
Fast, reliable data loading from Google Sheets
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
from config import SHEET_ID, SHEET_GIDS

logger = logging.getLogger(__name__)

class DataHandler:
    """Streamlined data handler for Google Sheets"""
    
    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load all data with parallel processing
        Returns: (stocks_df, sector_df, status)
        """
        status = {'success': False, 'errors': []}
        
        try:
            # Build URLs
            urls = {
                'stocks': f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GIDS['watchlist']}",
                'sector': f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GIDS['sector']}"
            }
            
            # Parallel loading for speed
            dataframes = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(pd.read_csv, url): name 
                          for name, url in urls.items()}
                
                for future in futures:
                    name = futures[future]
                    try:
                        dataframes[name] = future.result()
                    except Exception as e:
                        status['errors'].append(f"{name}: {str(e)}")
                        logger.error(f"Failed to load {name}: {e}")
            
            # Process data
            if 'stocks' not in dataframes:
                return pd.DataFrame(), pd.DataFrame(), status
            
            stocks_df = DataHandler._clean_stocks(dataframes['stocks'])
            sector_df = DataHandler._clean_sectors(dataframes.get('sector', pd.DataFrame()))
            
            # Add calculated fields
            stocks_df = DataHandler._add_calculations(stocks_df)
            
            status['success'] = True
            return stocks_df, sector_df, status
            
        except Exception as e:
            status['errors'].append(str(e))
            logger.error(f"Data loading failed: {e}")
            return pd.DataFrame(), pd.DataFrame(), status
    
    @staticmethod
    def _clean_stocks(df: pd.DataFrame) -> pd.DataFrame:
        """Fast cleaning for stock data"""
        # Essential columns only - remove clutter
        essential_cols = {
            'ticker': 'ticker',
            'company_name': 'name',
            'price': 'price',
            'pe': 'pe',
            'eps_current': 'eps',
            'market_cap': 'mcap',
            'sector': 'sector',
            'volume_1d': 'volume',
            'rvol': 'rvol',
            'ret_1d': 'ret_1d',
            'ret_7d': 'ret_7d', 
            'ret_30d': 'ret_30d',
            'ret_3m': 'ret_3m',
            'position_52w': 'pos_52w',
            'sma_20d': 'sma20',
            'sma_50d': 'sma50',
            'sma_200d': 'sma200'
        }
        
        # Rename to shorter names for efficiency
        df = df.rename(columns={k: v for k, v in essential_cols.items() if k in df.columns})
        
        # Keep only essential columns that exist
        keep_cols = [v for v in essential_cols.values() if v in df.columns]
        df = df[keep_cols]
        
        # Clean ticker
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
            df = df[df['ticker'] != 'NAN']
        
        # Vectorized numeric conversion
        numeric_cols = ['price', 'pe', 'eps', 'volume', 'rvol', 
                       'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m',
                       'pos_52w', 'sma20', 'sma50', 'sma200']
        
        for col in numeric_cols:
            if col in df.columns:
                # Fast conversion without regex
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Parse market cap efficiently
        if 'mcap' in df.columns:
            df['mcap'] = DataHandler._parse_mcap(df['mcap'])
        
        # Fill critical missing values
        df['price'] = df['price'].fillna(0)
        df['pe'] = df['pe'].fillna(0)
        df['volume'] = df['volume'].fillna(0)
        df['rvol'] = df['rvol'].fillna(1)
        
        # Remove invalid rows
        df = df[df['price'] > 0]
        
        return df
    
    @staticmethod
    def _clean_sectors(df: pd.DataFrame) -> pd.DataFrame:
        """Fast cleaning for sector data"""
        if df.empty:
            return df
        
        # Essential columns only
        essential_cols = ['sector', 'sector_ret_1d', 'sector_ret_7d', 
                         'sector_ret_30d', 'sector_count']
        
        keep_cols = [col for col in essential_cols if col in df.columns]
        df = df[keep_cols]
        
        # Convert percentages
        for col in df.columns:
            if 'ret_' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def _parse_mcap(series: pd.Series) -> pd.Series:
        """Vectorized market cap parsing"""
        if series.dtype != 'object':
            return series
        
        # Convert to string
        s = series.astype(str).str.upper()
        
        # Initialize result
        result = pd.Series(0.0, index=series.index)
        
        # Crores (most common in India)
        cr_mask = s.str.contains('CR', na=False)
        if cr_mask.any():
            nums = s[cr_mask].str.extract(r'([\d.]+)', expand=False)
            result[cr_mask] = pd.to_numeric(nums, errors='coerce') * 1e7
        
        # Direct numbers
        num_mask = ~cr_mask & s.str.match(r'^[\d.]+$', na=False)
        if num_mask.any():
            result[num_mask] = pd.to_numeric(s[num_mask], errors='coerce')
        
        return result
    
    @staticmethod
    def _add_calculations(df: pd.DataFrame) -> pd.DataFrame:
        """Add only essential calculated fields"""
        # Price above SMA20 (simple trend indicator)
        if 'price' in df.columns and 'sma20' in df.columns:
            df['above_sma20'] = df['price'] > df['sma20']
        
        # Simple momentum indicator
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            df['momentum_trend'] = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
        
        # Volume spike indicator
        if 'rvol' in df.columns:
            df['volume_spike'] = df['rvol'] > 2.0
        
        # Market cap category (simplified)
        if 'mcap' in df.columns:
            df['mcap_cat'] = pd.cut(
                df['mcap'],
                bins=[0, 5e9, 5e10, 2e11, float('inf')],
                labels=['Small', 'Mid', 'Large', 'Mega']
            )
        
        return df
    
    @staticmethod
    def assess_quality(df: pd.DataFrame) -> Dict:
        """Quick data quality assessment"""
        if df.empty:
            return {'score': 0, 'status': 'No Data'}
        
        # Calculate completeness for critical columns
        critical = ['ticker', 'price', 'pe', 'ret_30d']
        completeness = []
        
        for col in critical:
            if col in df.columns:
                pct = (df[col].notna().sum() / len(df)) * 100
                completeness.append(pct)
        
        avg_completeness = np.mean(completeness) if completeness else 0
        
        # Determine status
        if avg_completeness >= 90:
            status = 'Excellent'
        elif avg_completeness >= 75:
            status = 'Good'
        elif avg_completeness >= 60:
            status = 'Fair'
        else:
            status = 'Poor'
        
        return {
            'score': round(avg_completeness, 1),
            'status': status,
            'rows': len(df)
        }
