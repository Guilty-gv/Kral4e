# -*- coding: utf-8 -*-
"""
Long-Term Crypto Analyzer + ML + Candlestick & Harmonic Patterns + Telegram Notifier
Optimized for KuCoin + GitHub Actions
"""

import pandas as pd, ta, asyncio, numpy as np, os
from telegram import Bot
from datetime import datetime
import xgboost as xgb
from kucoin.client import Market

# ================= CONFIG =================
KUCOIN_PAIRS = ["BTCUSDT","ETHUSDT","ADAUSDT","SOLUSDT","XRPUSDT"]
TIMEFRAMES = ["1d","1w"]
EMA_PERIODS = [50,200]
RSI_PERIOD = 14
STOCH_PERIOD = 14
ATR_PERIOD = 14
PRICE_CHANGE_THRESHOLD = 0.01
MAX_OHLCV = 200

market_client = Market()

# ================= TELEGRAM =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)
last_price_sent = {}

def now_str(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def send_telegram(msg: str):
    try:
        bot.send_message(CHAT_ID, msg)
        print("Telegram message sent!")
    except Exception as e:
        print("Telegram send error:", e)

# ================= HELPERS =================
def kucoin_interval_map(interval):
    return {"1d":"1day","1w":"1week"}.get(interval, "1day")

async def fetch_kucoin(symbol, interval="1d"):
    for attempt in range(3):
        try:
            kline_interval = kucoin_interval_map(interval)
            symbol_str = symbol.replace("USDT","-USDT")
            data = market_client.get_kline(symbol_str, kline_interval, limit=MAX_OHLCV)
            if data:
                df = pd.DataFrame(data, columns=["timestamp","open","close","high","low","volume","turnover"])
                df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="s")
                for c in ["open","close","high","low","volume","turnover"]:
                    df[c] = df[c].astype(float)
                return df
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {symbol}: {e}")
            await asyncio.sleep(2)
    return pd.DataFrame()

async def fetch_data(symbol, interval="1d"):
    return await fetch_kucoin(symbol, interval)

# ================= INDICATORS =================
def add_indicators(df):
    if df.empty: return df
    for p in EMA_PERIODS: df[f"EMA{p}"] = df['close'].ewm(span=p, adjust=False).mean()
    df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min())/2
    df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min())/2
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], RSI_PERIOD).rsi()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], STOCH_PERIOD)
    df['%K'] = stoch.stoch(); df['%D'] = stoch.stoch_signal()
    macd = ta.trend.MACD(df['close']); df['MACD'] = macd.macd(); df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close']); df['BB_upper'] = bb.bollinger_hband(); df['BB_lower'] = bb.bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ATR_PERIOD).average_true_range()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    high = df['close'].max(); low = df['close'].min(); diff = high-low
    df['Fib_0.236']=high-0.236*diff; df['Fib_0.382']=high-0.382*diff; df['Fib_0.5']=high-0.5*diff; df['Fib_0.618']=high-0.618*diff
    vol_ema = df['volume'].ewm(span=20,adjust=False).mean()
    df['VolumeSpike'] = df['volume'] >= 2*vol_ema
    return df

# ================= CANDLE & HARMONIC =================
def candlestick_patterns(df):
    row = df.iloc[-1]; patterns=[]
    o,h,l,c = row['open'], row['high'], row['low'], row['close']
    body = abs(c-o); upper_shadow=h-max(c,o); lower_shadow=min(c,o)-l
    if lower_shadow>2*body and upper_shadow<0.5*body: patterns.append("BUY")   # Hammer
    if upper_shadow>2*body and lower_shadow<0.5*body: patterns.append("SELL")  # Inverted Hammer
    if body<0.1*(h-l): patterns.append("HOLD")  # Doji
    if len(df)>=2:
        prev=df.iloc[-2]
        if prev['close']<prev['open'] and c>o and c>prev['open'] and o<prev['close']: patterns.append("BUY") # Bullish Engulfing
        if prev['close']>prev['open'] and c<o and o>prev['close'] and c<prev['open']: patterns.append("SELL") # Bearish Engulfing
    return patterns

def harmonic_patterns(df):
    row = df.iloc[-1]; high, low = df['close'].max(), df['close'].min()
    fib_786 = high-0.786*(high-low)
    if row['close'] < fib_786: return ["BUY"]
    elif row['close'] > fib_786: return ["SELL"]
    else: return []

def generate_indicator_signals(df):
    row = df.iloc[-1]; p=row['close']; signals={}
    signals['EMA_cross'] = ["BUY" if row['EMA50']>row['EMA200'] else "SELL"]
    signals['Golden_Death_Cross'] = ["BUY" if row['EMA50']>row['EMA200'] else "SELL"]
    signals['RSI'] = ["BUY" if row['RSI']<30 else "SELL" if row['RSI']>70 else ""]
    signals['Stochastic'] = ["BUY" if row['%K']>row['%D'] else "SELL"]
    signals['MACD'] = ["BUY" if row['MACD']>row['MACD_signal'] else "SELL"]
    signals['Bollinger'] = ["BUY" if p<row['BB_lower'] else "SELL" if p>row['BB_upper'] else ""]
    fib_signals=[]
    for f in ['Fib_0.236','Fib_0.382','Fib_0.5','Fib_0.618']:
        if abs(p-row[f])/p<0.01: fib_signals.append("BUY" if p<row[f] else "SELL")
    signals['Fibonacci']=fib_signals
    signals['VolumeSpike'] = ["BUY"] if row['VolumeSpike'] else []
    signals['Candlestick'] = candlestick_patterns(df)
    signals['Harmonic'] = harmonic_patterns(df)
    return signals

def combine_signals(signals_dict):
    votes=[]; [votes.extend(v) for v in signals_dict.values()]
    buy = votes.count("BUY"); sell = votes.count("SELL")
    if buy>sell: return "BUY"
    elif sell>buy: return "SELL"
    else: return "HOLD"

# ================= LOG =================
CSV_FILE = "crypto_signals_log.csv"
def log_to_csv(symbol, interval, price, final_signal, indicator_signals):
    entry={"timestamp": now_str(), "symbol": symbol, "interval": interval,
           "price": round(float(price),8), "signal": final_signal,
           "indicator_signals": ",".join([s for sl in indicator_signals.values() for s in sl])}
    df_row=pd.DataFrame([entry])
    df_row.to_csv(CSV_FILE, mode="a", index=False, header=not os.path.exists(CSV_FILE))

# ================= ANALYSIS =================
async def analyze_coin(symbol):
    global last_price_sent
    interval_msgs={}
    for tf in TIMEFRAMES:
        df = await fetch_data(symbol, tf)
        if df.empty: continue
        df = add_indicators(df)
        signals_dict = generate_indicator_signals(df)
        final_signal = combine_signals(signals_dict)
        price = df['close'].iloc[-1]
        key=(symbol, tf)
        if key in last_price_sent and abs(price-last_price_sent[key])/last_price_sent[key]<PRICE_CHANGE_THRESHOLD:
            continue
        last_price_sent[key]=price
        interval_msgs[tf]=final_signal
        log_to_csv(symbol, tf, price, final_signal, signals_dict)
    if interval_msgs:
        msg_lines=[f"⏰ {now_str()}", f"📊 {symbol} Signals:"]
        for k,v in interval_msgs.items(): msg_lines.append(f"{k} → {v}")
        msg = "\n".join(msg_lines)
        send_telegram(msg)

# ================= MAIN =================
async def main():
    tasks=[analyze_coin(sym) for sym in KUCOIN_PAIRS]
    await asyncio.gather(*tasks)

if __name__=="__main__":
    print(f"{now_str()} ▶ Starting Long-Term Crypto Bot (KuCoin Only)")
    asyncio.run(main())
