import yfinance as yf
import pandas as pd
import ta
import requests
import os
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# टेलिग्राम माहिती (ही आपण पुढे Settings मध्ये सुरक्षित ठेवू)
BOT_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "TATAMOTORS.NS"
]

results = []

for ticker in stocks:
    try:
        df = yf.download(ticker, period="2y", progress=False)
        if df.empty or len(df) < 100:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        latest_data = df[['RSI', 'EMA_20', 'EMA_50', 'Return']].iloc[-1:]
        train_df = df.dropna()
        
        X = train_df[['RSI', 'EMA_20', 'EMA_50', 'Return']]
        y = train_df['Target']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[:-1], y[:-1])
        
        prob = model.predict_proba(latest_data)[0][1] * 100
        results.append({
            'Share': ticker.replace('.NS', ''),
            'Price': round(df['Close'].iloc[-1], 2),
            'Prob': round(prob, 1)
        })
    except Exception as e:
        pass

final_df = pd.DataFrame(results).sort_values(by='Prob', ascending=False)

# मेसेज तयार करणे
message = "📊 *आजचे AI मार्केट स्कोअरिंग (वर जाण्याची शक्यता)*:\n\n"
for index, row in final_df.iterrows():
    message += f"🔹 *{row['Share']}* (₹{row['Price']}) : *{row['Prob']}%*\n"

print(message)

# टेलिग्रामवर मेसेज पाठवणे
if BOT_TOKEN and CHAT_ID:
    url = f"https://telegram.org{BOT_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'})
