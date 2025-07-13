# 🔱 M.A.N.T.R.A. - Market Analysis Neural Trading Research Assistant

**Final Production Version 1.0.0 - Locked & Complete**

> *"All signal, no noise. Decisions, not guesses."*

## 🚀 Quick Deployment to Streamlit Cloud

### 1. File Structure
Make sure you have these 6 files in your repository:

```
your-repo/
├── app.py                 # Main Streamlit application
├── constants.py           # All configuration and business logic
├── data_loader.py         # Data loading and processing
├── signals.py             # Signal generation engine
├── ui_components.py       # UI components and styling
├── requirements.txt       # Dependencies
└── README.md             # This file
```

### 2. Deploy to Streamlit Cloud

1. **Push to GitHub**: Upload all files to your GitHub repository

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **That's it!** Your M.A.N.T.R.A. system will be live in 2-3 minutes.

### 3. Update Your Google Sheets ID

In `constants.py`, update the `GOOGLE_SHEET_ID` with your own:

```python
GOOGLE_SHEET_ID = "your-google-sheets-id-here"
```

Make sure your Google Sheets are **publicly viewable** (anyone with link can view).

## 📊 Features

✅ **Real-time Indian Stock Market Analysis**
- Live data from Google Sheets
- 40+ technical and fundamental indicators
- Advanced multi-factor signal generation

✅ **Professional Dashboard**
- Dark theme optimized for traders
- Mobile-responsive design
- Interactive charts and filters
- Export functionality

✅ **Intelligent Signals**
- STRONG_BUY, BUY, WATCH, NEUTRAL, AVOID ratings
- Risk assessment (Low, Medium, High)
- Detailed explanations for each signal
- Sector momentum analysis

✅ **Production Ready**
- Error handling and fallbacks
- Data quality monitoring
- Memory optimized
- Fast loading with caching

## 🛠️ System Requirements

- **Python 3.8+**
- **Streamlit Community Cloud** (free tier supported)
- **Google Sheets** with public view access
- **Internet connection** for real-time data

## 📈 Data Requirements

Your Google Sheets should have 3 tabs:

1. **Watchlist Sheet** - Main stock universe with all indicators
2. **Returns Sheet** - Multi-timeframe return analysis  
3. **Sector Sheet** - Sector-level performance data

See `constants.py` for complete column specifications.

## 🔧 Configuration

All business logic is in `constants.py`:

- **Scoring Functions**: Modify factor scoring algorithms
- **Weights**: Adjust factor importance (momentum, value, volume, technical)
- **Thresholds**: Change signal cut-offs
- **Colors**: Customize UI colors
- **URLs**: Update Google Sheets links

## 📱 Usage

1. **Overview**: Market metrics and top opportunities
2. **Filters**: Narrow down by signal, sector, risk, price, volume
3. **Analysis**: Detailed stock table with all factors
4. **Charts**: Sector heatmaps and distributions
5. **Export**: Download filtered results as CSV

## 🚨 Important Notes

- **Educational Use Only**: All signals are for learning purposes
- **Do Your Research**: Always validate before making investment decisions
- **Data Dependency**: Requires reliable Google Sheets data
- **No Real-time Prices**: Uses data from your sheets, not live feeds

## 🔒 Version Lock

This is the **FINAL, LOCKED VERSION**. No future updates or changes needed.

- All logic is data/config driven
- Change behavior by updating `constants.py` or your Google Sheets
- System is designed to be maintenance-free

## 🆘 Troubleshooting

**Data Not Loading?**
- Check Google Sheets are publicly accessible
- Verify GOOGLE_SHEET_ID in constants.py
- Ensure sheet tabs have correct names

**Slow Performance?**
- Data is cached for 5 minutes
- Use filters to reduce dataset size
- Check internet connection

**Display Issues?**
- Try refreshing the page
- Clear browser cache
- Check mobile vs desktop view

## 🏆 Success Metrics

- **Loading Time**: < 10 seconds for full data refresh
- **Data Quality**: 90%+ completeness score
- **Responsiveness**: Works on mobile and desktop
- **Reliability**: 99%+ uptime on Streamlit Cloud

---

**Built with ❤️ for the Indian stock market community**

*Philosophy: "Precision over noise. Decisions, not guesses."*
