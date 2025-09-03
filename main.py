# -*- coding: utf-8 -*-
"""
Crypto Swing Trading + XGBoost Analyzer + Telegram Notifier
Optimized for GitHub Actions
"""

import aiohttp, pandas as pd, ta, asyncio, numpy as np, os
from telegram import Bot
from datetime import datetime
import xgboost as xgb

# ================= CONFIG =================
BINANCE_PAIRS = [BINANCE_PAIRS = [
    "BTCUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "ONDOUSDT",
    "ACHUSDT",
    "ALGOUSDT",
    "AVAXUSDT",
    "FETUSDT",
    "IOTAUSDT",
    "AXLUSDT",
    "HBARUSDT",
    "WUSDT"  # Wormhole (W) на Binance
]
COINGECKO_PAIRS = {"KASUSDT":"kas-network"}
TIMEFRAMES = ["1d","4h","1h","15m"]

# EMA комбинации по timeframe
EMA_MAP = {
    "15m": (9, 20),
    "1h": (20, 50),
    "4h": (50, 200),
    "1d": (100, 200)
}

RSI_PERIOD = 14
STOCH_PERIOD = 14
ATR_PERIOD = 14
PRICE_CHANGE_THRESHOLD = 0.00  # Испраќа само ако промена ≥ 5%
MAX_OHLCV = 200

BINANCE_URL = "https://api.binance.com/api/v3/klines"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

# ================= TELEGRAM =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)
last_price_sent = {}

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def send_telegram(msg: str):
    try:
        bot.send_message(CHAT_ID, msg)
        print("Telegram message sent!")
    except Exception as e:
        print("Telegram send error:", e)

# ================= HELPERS =================
async def fetch_binance(symbol, interval="1h"):
    params = {"symbol":symbol,"interval":interval,"limit":MAX_OHLCV}
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.get(BINANCE_URL, params=params, timeout=30) as resp:
                    data = await resp.json()
                    if isinstance(data,list) and len(data)>0:
                        df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume",
                                                        "close_time","quote_asset_volume","num_trades",
                                                        "taker_buy_base","taker_buy_quote","ignore"])
                        for c in ["open","high","low","close","volume"]: df[c]=df[c].astype(float)
                        df["open_time"]=pd.to_datetime(df["open_time"], unit="ms")
                        df["close_time"]=pd.to_datetime(df["close_time"], unit="ms")
                        return df
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {symbol}: {e}")
                await asyncio.sleep(2)
    print(f"DEBUG: {symbol} {interval} - нема податоци")
    return pd.DataFrame()

async def fetch_coingecko(symbol_id, interval="hourly"):
    url = COINGECKO_URL.format(id=symbol_id)
    params = {"vs_currency":"usd","days":30,"interval":interval}
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    data = await resp.json()
                    prices = data.get("prices", [])
                    if len(prices) > 0:
                        df = pd.DataFrame(prices, columns=["timestamp","close"])
                        df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms")
                        df["open"]=df["close"]; df["high"]=df["close"]; df["low"]=df["close"]; df["volume"]=0
                        return df
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {symbol_id}: {e}")
                await asyncio.sleep(2)
    print(f"DEBUG: {symbol_id} {interval} - нема податоци")
    return pd.DataFrame()

async def fetch_data(symbol, interval="1h"):
    if symbol in BINANCE_PAIRS: return await fetch_binance(symbol, interval)
    elif symbol in COINGECKO_PAIRS: return await fetch_coingecko(COINGECKO_PAIRS[symbol], interval)
    return pd.DataFrame()

# ================= INDICATORS =================
def add_indicators(df, interval):
    if df.empty: return df
    # додади сите EMA од EMA_MAP
    for ema_set in EMA_MAP.values():
        for p in ema_set:
            if f"EMA{p}" not in df.columns:
                df[f"EMA{p}"] = df['close'].ewm(span=p, adjust=False).mean()

    try:
        df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min())/2
        df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min())/2
    except: pass
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], RSI_PERIOD).rsi()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], STOCH_PERIOD)
    df['%K'] = stoch.stoch(); df['%D'] = stoch.stoch_signal()
    macd = ta.trend.MACD(df['close']); df['MACD'] = macd.macd(); df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close']); df['BB_upper'] = bb.bollinger_hband(); df['BB_lower'] = bb.bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ATR_PERIOD).average_true_range()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    return df

# ================= SIGNAL & ML =================
def generate_signal(row, interval):
    if row.empty: return ["HOLD"]
    signals = []
    p = row['close']

    # EMA логика според timeframe
    short_ema, long_ema = EMA_MAP.get(interval, (20, 50))
    if f"EMA{short_ema}" in row and f"EMA{long_ema}" in row:
        if row[f"EMA{short_ema}"] > row[f"EMA{long_ema}"]:
            signals.append("BUY")
        elif row[f"EMA{short_ema}"] < row[f"EMA{long_ema}"]:
            signals.append("SELL")

    # Ichimoku
    if 'Tenkan' in row and 'Kijun' in row:
        signals.append("BUY" if p > max(row['Tenkan'], row['Kijun']) else "SELL")

    # RSI
    signals.append("BUY" if row['RSI'] < 30 else "SELL" if row['RSI'] > 70 else "")

    # Stochastic
    signals.append("BUY" if row['%K'] > row['%D'] else "SELL")

    # MACD
    signals.append("BUY" if row['MACD'] > row['MACD_signal'] else "SELL")

    # Bollinger
    if p < row['BB_lower']: signals.append("BUY")
    elif p > row['BB_upper']: signals.append("SELL")

    return [s for s in signals if s]

# ================= LOG =================
CSV_FILE = "crypto_signals_log.csv"
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
    interval_msgs = {}
    global last_price_sent
    for tf in TIMEFRAMES:
        df = await fetch_data(symbol, tf)
        if df.empty:
            continue
        df = add_indicators(df, tf)
        indicator_signals = generate_signal(df.iloc[-1], tf)
        price = df['close'].iloc[-1]
        key = (symbol, tf)
        if key in last_price_sent and abs(price-last_price_sent[key])/last_price_sent[key]<PRICE_CHANGE_THRESHOLD:
            continue
        last_price_sent[key]=price
        final_signal = " / ".join(indicator_signals) if indicator_signals else "HOLD"
        interval_msgs[tf] = final_signal
        log_to_csv(symbol, tf, price, final_signal, indicator_signals)
    if interval_msgs:
        msg_lines=[f"⏰ {now_str()}", f"📊 {symbol} Signals:"]
        for k,v in interval_msgs.items(): msg_lines.append(f"{k} → {v}")
        msg = "\n".join(msg_lines)
        send_telegram(msg)

# ================= MAIN =================
async def main():
    tasks = [analyze_coin(sym) for sym in BINANCE_PAIRS + list(COINGECKO_PAIRS.keys())]
    await asyncio.gather(*tasks)

if __name__=="__main__":
    print(f"{now_str()} ▶ Starting Crypto Signal Bot")
    asyncio.run(main())
