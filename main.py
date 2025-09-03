# -*- coding: utf-8 -*-
"""
Crypto Swing Trading + XGBoost Analyzer + Telegram Notifier
Optimized for GitHub Actions with dynamic EMA
"""

import aiohttp, pandas as pd, ta, asyncio, numpy as np, os
from telegram import Bot
from datetime import datetime
import xgboost as xgb

# ================= CONFIG =================
BINANCE_PAIRS = ["BTCUSDT","XRPUSDT","LINKUSDT","ONDOUSDT"]
COINGECKO_PAIRS = {"KASUSDT":"kas-network"}
TIMEFRAMES = ["15m","30m","1h","4h","1d"]
PRICE_CHANGE_THRESHOLD = 0.05
MAX_OHLCV = 200

BINANCE_URL = "https://api.binance.com/api/v3/klines"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

# ================= TELEGRAM =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# ================= EMA COMBINATIONS =================
EMA_COMBINATIONS = {
    "15m": [9, 21],
    "30m": [12, 26],
    "1h": [20, 50],
    "4h": [20, 50],
    "1d": [50, 200]
}

# ================= LOGGING =================
CSV_FILE = "crypto_signals_log.csv"
last_price_sent = {}

def now_str(): 
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def send_telegram(msg: str):
    try:
        bot.send_message(CHAT_ID, msg)
        print("Telegram message sent!")
        print(msg)
    except Exception as e:
        print("Telegram send error:", e)
        print(msg)

# ================= HELPERS =================
async def fetch_binance(symbol, interval="1h"):
    try:
        params = {"symbol":symbol,"interval":interval,"limit":MAX_OHLCV}
        async with aiohttp.ClientSession() as session:
            async with session.get(BINANCE_URL, params=params, timeout=20) as resp:
                data = await resp.json()
                if not isinstance(data,list) or len(data)==0: return pd.DataFrame()
                df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume",
                                                 "close_time","quote_asset_volume","num_trades",
                                                 "taker_buy_base","taker_buy_quote","ignore"])
                for c in ["open","high","low","close","volume"]: df[c]=df[c].astype(float)
                df["open_time"]=pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"]=pd.to_datetime(df["close_time"], unit="ms")
                return df
    except Exception as e:
        print(f"Error in fetch_binance({symbol}): {e}")
        return pd.DataFrame()

async def fetch_coingecko(symbol_id, interval="hourly"):
    try:
        url = COINGECKO_URL.format(id=symbol_id)
        params = {"vs_currency":"usd","days":30,"interval":interval}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=20) as resp:
                data = await resp.json()
                prices = data.get("prices", [])
                if len(prices)==0: return pd.DataFrame()
                df = pd.DataFrame(prices, columns=["timestamp","close"])
                df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms")
                df["open"]=df["close"]; df["high"]=df["close"]; df["low"]=df["close"]; df["volume"]=0
                return df
    except Exception as e:
        print(f"Error in fetch_coingecko({symbol_id}): {e}")
        return pd.DataFrame()

async def fetch_data(symbol, interval="1h"):
    if symbol in BINANCE_PAIRS: return await fetch_binance(symbol, interval)
    elif symbol in COINGECKO_PAIRS: return await fetch_coingecko(COINGECKO_PAIRS[symbol], interval)
    return pd.DataFrame()

# ================= INDICATORS =================
def add_indicators(df, timeframe):
    if df.empty: return df
    ema_periods = EMA_COMBINATIONS.get(timeframe, [20,50])
    for p in ema_periods:
        df[f"EMA{p}"] = df['close'].ewm(span=p, adjust=False).mean()
    # други индикатори (RSI, MACD, Stochastic)
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 14)
    df['%K'] = stoch.stoch(); df['%D'] = stoch.stoch_signal()
    macd = ta.trend.MACD(df['close']); df['MACD'] = macd.macd(); df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close']); df['BB_upper'] = bb.bollinger_hband(); df['BB_lower'] = bb.bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    high = df['close'].max(); low = df['close'].min(); diff = high-low
    df['Fib_0.236']=high-0.236*diff; df['Fib_0.382']=high-0.382*diff; df['Fib_0.5']=high-0.5*diff; df['Fib_0.618']=high-0.618*diff
    vol_ema = df['volume'].ewm(span=20,adjust=False).mean()
    df['VolumeSpike'] = df['volume'] >= 2*vol_ema
    return df

# ================= SIGNAL =================
def generate_signal(row, timeframe):
    if row.empty: return ["HOLD"]
    signals = []

    # EMA crossover сигнал
    ema_periods = EMA_COMBINATIONS.get(timeframe, [20,50])
    short_ema, long_ema = ema_periods[0], ema_periods[1]
    if f"EMA{short_ema}" in row and f"EMA{long_ema}" in row:
        if row[f"EMA{short_ema}"] > row[f"EMA{long_ema}"]:
            signals.append("BUY")
        elif row[f"EMA{short_ema}"] < row[f"EMA{long_ema}"]:
            signals.append("SELL")

    # останати индикатори (RSI, MACD, Stochastic, Bollinger, Fib, VolumeSpike)
    p = row['close']
    if row['RSI'] < 30: signals.append("BUY")
    elif row['RSI'] > 70: signals.append("SELL")
    if row['%K'] > row['%D']: signals.append("BUY")
    else: signals.append("SELL")
    if row['MACD'] > row['MACD_signal']: signals.append("BUY")
    else: signals.append("SELL")
    if p < row['BB_lower']: signals.append("BUY")
    elif p > row['BB_upper']: signals.append("SELL")
    for f in ['Fib_0.236','Fib_0.382','Fib_0.5','Fib_0.618']:
        if abs(p - row[f])/p < 0.01: signals.append("BUY" if p < row[f] else "SELL")
    if 'VolumeSpike' in row and row['VolumeSpike']: signals.append("BUY")

    return [s for s in signals if s]

# ================= XGBoost ML =================
async def run_ml(df):
    signals=[]
    try:
        feats=pd.DataFrame(index=df.index)
        for p in [20,50]: feats[f"EMA{p}"]=df['close'].ewm(span=p,adjust=False).mean()
        feats["RSI"]=ta.momentum.RSIIndicator(df['close'],14).rsi()
        feats.dropna(inplace=True)
        if len(feats)>=50:
            y=(df['close'].shift(-1).loc[feats.index]>df['close'].loc[feats.index]).astype(int)
            X=feats
            model=xgb.XGBClassifier(n_estimators=25,max_depth=3,use_label_encoder=False,eval_metric='logloss')
            model.fit(X,y)
            pred=int(model.predict(X.iloc[-1:].values)[0])
            signals.append("BUY" if pred==1 else "SELL")
    except Exception as e:
        print(f"Error in run_ml: {e}")
    return signals

# ================= MAJORITY VOTE =================
def combine_signals(indicator_signals, ml_signals):
    votes = indicator_signals.copy(); votes.extend(ml_signals)
    buy = votes.count("BUY"); sell = votes.count("SELL")
    if buy>sell: return "BUY"
    elif sell>buy: return "SELL"
    else: return "HOLD"

# ================= LOG TO CSV =================
def log_to_csv(symbol, interval, price, final_signal, indicator_signals):
    entry = {
        "timestamp": now_str(),
        "symbol": symbol,
        "interval": interval,
        "price": round(float(price),8),
        "signal": final_signal,
        "indicator_signals": ",".join(indicator_signals)
    }
    df_row = pd.DataFrame([entry])
    df_row.to_csv(CSV_FILE, mode="a", index=False, header=not os.path.exists(CSV_FILE))

# ================= ANALYSIS =================
async def analyze_coin(symbol):
    global last_price_sent
    for tf in TIMEFRAMES:
        df = await fetch_data(symbol, tf)
        if df.empty: 
            send_telegram(f"DEBUG: No data for {symbol} {tf}")
            continue
        df = add_indicators(df, tf)
        indicator_signals = generate_signal(df.iloc[-1], tf)
        ml_signal = await run_ml(df)
        final_signal = combine_signals(indicator_signals, ml_signal)
        price = df['close'].iloc[-1]
        key = (symbol, tf)
        if key in last_price_sent and abs(price-last_price_sent[key])/last_price_sent[key]<PRICE_CHANGE_THRESHOLD:
            continue
        last_price_sent[key]=price
        log_to_csv(symbol, tf, price, final_signal, indicator_signals)
        send_telegram(f"⏰ {now_str()} | {symbol} {tf} → {final_signal}")

# ================= MAIN =================
async def main():
    tasks = [analyze_coin(sym) for sym in BINANCE_PAIRS + list(COINGECKO_PAIRS.keys())]
    await asyncio.gather(*tasks)

if __name__=="__main__":
    print(f"{now_str()} ▶ Starting Crypto Signal Bot")
    asyncio.run(main())
