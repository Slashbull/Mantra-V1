"""
constants.py - M.A.N.T.R.A. Configuration Module
===============================================
All configuration, colors, thresholds, and styling constants.
Production-ready, bug-free version.
"""

import pandas as pd
import numpy as np

# ============================================================================
# GOOGLE SHEETS CONFIGURATION
# ============================================================================

GOOGLE_SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"

SHEET_GIDS = {
    "watchlist": "2026492216",
    "returns": "100734077",
    "sector": "140104095"
}

SHEET_URLS = {
    name: f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={gid}"
    for name, gid in SHEET_GIDS.items()
}

# ============================================================================
# SIGNAL THRESHOLDS
# ============================================================================

SIGNAL_THRESHOLDS = {
    "STRONG_BUY": 85,
    "BUY": 70,
    "WATCH": 55,
    "NEUTRAL": 40
}

# ============================================================================
# FACTOR WEIGHTS
# ============================================================================

FACTOR_WEIGHTS = {
    "momentum": 0.40,
    "value": 0.25,
    "volume": 0.20,
    "technical": 0.15
}

# ============================================================================
# UI COLORS AND STYLING
# ============================================================================

SIGNAL_COLORS = {
    "STRONG_BUY": "#00ff88",
    "BUY": "#00cc66",
    "WATCH": "#ffaa00", 
    "NEUTRAL": "#888888",
    "AVOID": "#ff4444"
}

RISK_COLORS = {
    "Low": "#00cc66",
    "Medium": "#ffaa00",
    "High": "#ff4444"
}

SECTOR_COLORS = [
    "#00ff88", "#00cc66", "#60a5fa", "#a78bfa", 
    "#f472b6", "#fb7185", "#fbbf24", "#34d399"
]

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styling */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
    color: #e2e8f0;
}

/* Hide Streamlit Elements */
#MainMenu, footer, .stDeployButton, .stDecoration {
    visibility: hidden !important;
    display: none !important;
}

.stHeader {
    background: transparent !important;
}

/* Header Styling */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid #4a5568;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}

.main-title {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    text-shadow: 0 0 30px rgba(96, 165, 250, 0.3);
}

.main-subtitle {
    font-size: 1.2rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #475569;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #60a5fa, #34d399);
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(96, 165, 250, 0.2);
    border-color: #60a5fa;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0.5rem 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.metric-label {
    font-size: 0.875rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-delta {
    font-size: 0.875rem;
    font-weight: 600;
    margin-top: 0.25rem;
}

/* Stock Cards */
.stock-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #475569;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stock-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #60a5fa, #34d399);
}

.stock-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(96, 165, 250, 0.15);
    border-color: #60a5fa;
}

.stock-ticker {
    font-size: 1.25rem;
    font-weight: 700;
    color: #60a5fa;
    margin-bottom: 0.25rem;
}

.stock-name {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-bottom: 1rem;
    line-height: 1.4;
}

.stock-price {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0.75rem 0;
}

.stock-details {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #475569;
}

/* Signal Badges */
.signal-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.375rem 0.875rem;
    border-radius: 25px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Data Quality Indicator */
.quality-indicator {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-radius: 15px;
    padding: 1rem;
    border: 1px solid #475569;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.quality-excellent { border-color: #00cc66; }
.quality-good { border-color: #60a5fa; }
.quality-fair { border-color: #ffaa00; }
.quality-poor { border-color: #ff4444; }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    border: none !important;
    border-radius: 15px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
}

/* Sidebar Styling */
.css-1d391kg {
    background: linear-gradient(180deg, #1e293b 0%, #334155 100%) !important;
}

.css-1lcbmhc {
    background: linear-gradient(180deg, #1e293b 0%, #334155 100%) !important;
}

/* Tables */
.stDataFrame {
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
}

.stDataFrame > div {
    border-radius: 15px !important;
}

/* Selectbox and Input Styling */
.stSelectbox > div > div {
    background: #2d3748 !important;
    border: 1px solid #4a5568 !important;
    border-radius: 10px !important;
}

.stSlider > div > div > div {
    background: #2d3748 !important;
}

/* Charts */
.js-plotly-plot {
    border-radius: 15px !important;
    overflow: hidden !important;
}

/* Loading Spinner */
.stSpinner > div {
    border-color: #60a5fa transparent transparent transparent !important;
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .main-title { 
        font-size: 2.5rem !important; 
    }
    .metric-card { 
        height: 120px !important; 
        padding: 1rem !important; 
    }
    .metric-value { 
        font-size: 2rem !important; 
    }
    .stock-card { 
        padding: 1rem !important; 
    }
    .stock-price {
        font-size: 1.5rem !important;
    }
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
    animation: fadeIn 0.5s ease-out;
}

/* Status Colors */
.status-excellent { background: #059669 !important; color: white !important; }
.status-good { background: #0284c7 !important; color: white !important; }
.status-fair { background: #d97706 !important; color: white !important; }
.status-poor { background: #dc2626 !important; color: white !important; }

/* Section Spacing */
.section-spacer {
    height: 2rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 2rem;
    border-top: 1px solid #334155;
    margin-top: 3rem;
    background: linear-gradient(135deg, #1a202c, #2d3748);
    border-radius: 15px;
}
</style>
"""

# ============================================================================
# DATA SCHEMA DOCUMENTATION
# ============================================================================

SCHEMA_DOC = """
🔱 M.A.N.T.R.A. Data Schema

📊 Watchlist Sheet Columns:
• ticker: Stock symbol (TCS, RELIANCE)
• company_name: Full company name
• price: Current stock price (₹)
• ret_1d, ret_7d, ret_30d: Returns (%)
• pe: Price-to-Earnings ratio
• eps_current: Current EPS
• volume_1d: Daily volume
• rvol: Relative volume
• sector: Industry sector
• low_52w, high_52w: 52-week range
• sma_20d, sma_50d: Moving averages

🏭 Sector Sheet Columns:
• sector: Sector name
• sector_ret_1d/7d/30d: Sector returns (%)
• sector_count: Number of stocks

📈 Returns Sheet Columns:
• ticker: Stock symbol
• avg_ret_30d/3m/1y: Average returns
"""

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'data_load_failed': "❌ Failed to load data. Please check your internet connection.",
    'sheet_not_found': "❌ Google Sheet not accessible. Please check permissions.",
    'invalid_data': "⚠️ Data quality issues detected. Some features may be limited.",
    'calculation_error': "⚠️ Error in signal calculation. Using default values.",
    'network_timeout': "🌐 Network timeout. Please try refreshing the page."
}

# ============================================================================
# SUCCESS MESSAGES
# ============================================================================

SUCCESS_MESSAGES = {
    'data_loaded': "✅ Data loaded successfully!",
    'signals_calculated': "🧠 Signals calculated successfully!",
    'export_ready': "📥 Export file ready for download!"
}

# ============================================================================
# CACHE SETTINGS
# ============================================================================

CACHE_TTL = 300  # 5 minutes
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# ============================================================================
# VALIDATION RULES
# ============================================================================

VALIDATION_RULES = {
    'min_price': 1,
    'max_price': 100000,
    'min_volume': 1,
    'max_pe': 1000,
    'min_data_quality': 50
}
