"""
data_loader.py - M.A.N.T.R.A. Data Foundation
============================================
Clean, simple, reliable data loading
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Optional
import logging
from constants import GOOGLE_SHEET_ID, SHEET_CONFIGS, CACHE_DURATION_MINUTES

logger = logging.getLogger(__name__)

class DataLoader:
    """Simple and reliable data loader"""
    
    @staticmethod
    @st.cache_data(ttl=CACHE_DURATION_MINUTES*60, show_spinner=False)
    def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load and clean all data from Google Sheets"""
        try:
            # Build URLs
            watchlist_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['watchlist']['gid']}"
            sector_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['sector']['gid']}"
            
            # Load data
            stocks_df = pd.read_csv(watchlist_url)
            sector_df = pd.read_csv(sector_url)
            
            # Clean column names
            stocks_df.columns = stocks_df.columns.str.strip()
            sector_df.columns = sector_df.columns.str.strip()
            
            # Remove empty rows
            stocks_df = stocks_df.dropna(how='all')
            sector_df = sector_df.dropna(how='all')
            
            # Clean data types
            stocks_df = DataLoader._clean_stocks_data(stocks_df)
            sector_df = DataLoader._clean_sector_data(sector_df)
            
            # Health check
            health = {
                'status': 'success',
                'stocks_count': len(stocks_df),
                'sectors_count': len(sector_df),
                'data_quality': DataLoader._calculate_data_quality(stocks_df)
            }
            
            return stocks_df, sector_df, health
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return pd.DataFrame(), pd.DataFrame(), {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def _clean_stocks_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean stock data with proper type conversion"""
        # Price columns
        price_cols = ['price', 'prev_close', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce')
