import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime
import pytz
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# निफ्टी ५० मधील सर्व प्रमुख शेअर्स
stocks = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

buckets = {
    "100%": [], "90%": [], "80%": [], "70%": [], "60%": [],
    "50%": [], "40%": [], "30%": [], "20%": [], "10%": [], "5%": []
}

def get_bucket(prob):
    if prob >= 95: return "100%"
    elif prob >= 85: return "90%"
    elif prob >= 75: return "80%"
    elif prob >= 65: return "70%"
    elif prob >= 55: return "60%"
    elif prob >= 45: return "50%"
    elif prob >= 35: return "40%"
    elif prob >= 25: return "30%"
    elif prob >= 15: return "20%"
    elif prob >= 7.5: return "10%"
    else: return "5%"

def calculate_holding_days(prob, rsi, adx, price, ema20, ema50):
    # ट्रेंडची ताकद (ADX) आणि प्रोबॅबिलिटीनुसार होल्डिंग दिवस ठरवणे (1 दिवस ते 365 दिवस)
    if prob >= 85 and price > ema20 > ema50 and adx > 25:
        return "90 to 180 Days (Medium Term)"
    elif prob >= 75 and price > ema50:
        return "30 to 90 Days (Positional)"
    elif prob >= 65 and rsi < 65:
        return "7 to 21 Days (Swing Trade)"
    elif prob >= 55:
        return "1 to 5 Days (Short Term)"
    else:
        return "1 Day (Intraday Watch Only)"

for ticker in stocks:
    try:
        # १. मागील २ वर्षांचा डेटा + आजचा लाइव्ह डेटा (१ मिनिटाचा ताजा भाव)
        df = yf.download(ticker, period="2y", progress=False)
        live_df = yf.download(ticker, period="1d", interval="1m", progress=False)
        
        if df.empty or len(df) < 100:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not live_df.empty and isinstance(live_df.columns, pd.MultiIndex):
            live_df.columns = live_df.columns.get_level_values(0)

        # आजचा ताजा भाव (Live Price) शेवटच्या ओळीत अपडेट करणे
        current_price = float(live_df['Close'].iloc[-1]) if not live_df.empty else float(df['Close'].iloc[-1])
        df.iloc[-1, df.columns.get_loc('Close')] = current_price

        # २. सर्व अनिवार्य टेक्निकल इंडिकेटर्स (Mandatory Indicators)
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        df['Vol_Change'] = df['Volume'].pct_change()
        df['Return'] = df['Close'].pct_change()
        
        # टार्गेट: पुढील ५ दिवसांत शेअर वर जाण्याची शक्यता
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        
        features = ['RSI', 'MACD', 'MACD_Signal', 'EMA_20', 'EMA_50', 'ATR', 'ADX', 'Vol_Change', 'Return']
        
        latest_data = df[features].iloc[-1:]
        train_df = df.dropna()
        
        X = train_df[features]
        y = train_df['Target']
        
        # ३. प्रगत Gradient Boosting AI मॉडेल
        model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42)
        model.fit(X[:-5], y[:-5])
        
        prob = float(model.predict_proba(latest_data)[0][1]) * 100
        exact_prob = round(prob, 1)
        
        # ४. Min Low (सपोर्ट/स्टॉप-लॉस) आणि Min High (टार्गेट) काढणे
        atr_val = float(df['ATR'].iloc[-1])
        rsi_val = float(df['RSI'].iloc[-1])
        adx_val = float(df['ADX'].iloc[-1])
        ema20_val = float(df['EMA_20'].iloc[-1])
        ema50_val = float(df['EMA_50'].iloc[-1])
        
        # होल्डिंग कालावधीनुसार लो आणि हाय रेंज मोजणे
        holding_text = calculate_holding_days(exact_prob, rsi_val, adx_val, current_price, ema20_val, ema50_val)
        
        multiplier = 3.5 if "90 to 180" in holding_text else (2.5 if "30 to 90" in holding_text else (1.8 if "7 to 21" in holding_text else 1.2))
        min_low = round(current_price - (atr_val * 1.2), 2)
        min_high = round(current_price + (atr_val * multiplier), 2)
        
        share_name = ticker.replace('.NS', '')
        bucket_key = get_bucket(prob)
        
        row_str = f"| **{share_name}** | ₹{round(current_price, 2)} | **{exact_prob}%** | ₹{min_low} | ₹{min_high} | {holding_text} |"
        buckets[bucket_key].append((exact_prob, row_str))
        
    except Exception as e:
        pass

# भारतीय वेळेनुसार तारीख आणि वेळ
ist = pytz.timezone('Asia/Kolkata')
now = datetime.now(ist).strftime('%d-%b-%Y %I:%M %p')

md_content = f"# 📈 Live AI Stock Scoring & Holding Guide (100% to 5%)\n"
md_content += f"**शेवटचे लाइव्ह अपडेट (IST):** {now}\n\n"

for b_key in ["100%", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%", "5%"]:
    md_content += f"### 🎯 {b_key} Probability Group\n"
    items = sorted(buckets[b_key], key=lambda x: x[0], reverse=True)
    if items:
        md_content += "| Share Name | Current Price | Percentage | Min Low | Min High | Holding Days (1D - 1Y) |\n"
        md_content += "| :--- | :---: | :---: | :---: | :---: | :--- |\n"
        for _, row in items:
            md_content += row + "\n"
    else:
        md_content += "*सध्या या गटात कोणताही शेअर नाही.*\n"
    md_content += "\n---\n"

with open("README.md", "w", encoding="utf-8") as f:
    f.write(md_content)

print("Live Prediction & Holding Report Updated!")
