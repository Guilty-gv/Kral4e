# -*- coding: utf-8 -*-
"""
Crypto Swing Trading + XGBoost Analyzer + Telegram Notifier
Optimized for GitHub Actions with KuCoin only
"""

import pandas as pd, ta, asyncio, os
from telegram import Bot
from datetime import datetime
import xgboost as xgb
from kucoin.client import Client as KuClient

# ================= CONFIG =================
KUCOIN_PAIRS = ["BTC-USDT","ETH-USDT","LINK-USDT","XRP-USDT"]  # Твој листа на KuCoin парови
TIMEFRAMES = ["15m","1h","4h","1d"]
EMA_PERIODS = [20,50]
RSI_PERIOD = 14
STOCH_PERIOD = 14
ATR_PERIOD = 14
PRICE_CHANGE_THRESHOLD = 0.01
MAX_OHLCV = 200

# ================= KUCOIN API =================
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")
ku_client = KuClient(KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE)

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
async def fetch_kucoin(symbol, interval="1h"):
    interval_map = {"15m":"15min","1h":"1hour","4h":"4hour","1d":"1day"}
    kline_interval = interval_map.get(interval,"1hour")
    for attempt in range(3):
        try:
            klines = ku_client.get_kline(symbol, kline_interval, limit=MAX_OHLCV)
            if klines:
                df = pd.DataFrame(klines, columns=["time","open","close","high","low","volume","turnover"])
                df["open"]=df["open"].astype(float); df["high"]=df["high"].astype(float)
                df["low"]=df["low"].astype(float); df["close"]=df["close"].astype(float)
                df["volume"]=df["volume"].astype(float)
                df["open_time"]=pd.to_datetime(df["time"], unit="s")
                return df
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {symbol}: {e}")
            await asyncio.sleep(2)
    print(f"DEBUG: {symbol} {interval} - нема податоци")
    return pd.DataFrame()

async def fetch_data(symbol, interval="1h"):
    return await fetch_kucoin(symbol, interval)

# ================= INDICATORS =================
def add_indicators(df):
    if df.empty: return df
    for p in EMA_PERIODS: df[f"EMA{p}"] = df['close'].ewm(span=p, adjust=False).mean()
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
    high = df['close'].max(); low = df['close'].min(); diff = high-low
    df['Fib_0.236']=high-0.236*diff; df['Fib_0.382']=high-0.382*diff; df['Fib_0.5']=high-0.5*diff; df['Fib_0.618']=high-0.618*diff
    vol_ema = df['volume'].ewm(span=20,adjust=False).mean()
    df['VolumeSpike'] = df['volume'] >= 2*vol_ema
    return df

# ================= SIGNAL & ML =================
# (generate_signal, run_ml, combine_signals - исти како предходно)

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
        df = add_indicators(df)
        indicator_signals = generate_signal(df.iloc[-1])
        ml_signal = await run_ml(df)
        final_signal = combine_signals(indicator_signals, ml_signal)
        price = df['close'].iloc[-1]
        key = (symbol, tf)
        if key in last_price_sent and abs(price-last_price_sent[key])/last_price_sent[key]<PRICE_CHANGE_THRESHOLD:
            continue
        last_price_sent[key]=price
        interval_msgs[tf] = final_signal
        log_to_csv(symbol, tf, price, final_signal, indicator_signals)
    if interval_msgs:
        msg_lines=[f"⏰ {now_str()}", f"📊 {symbol} Signals:"]
        for k,v in interval_msgs.items(): msg_lines.append(f"{k} → {v}")
        msg = "\n".join(msg_lines)
        send_telegram(msg)

# ================= MAIN =================
async def main():
    tasks = [analyze_coin(sym) for sym in KUCOIN_PAIRS]
    await asyncio.gather(*tasks)

if __name__=="__main__":
    print(f"{now_str()} ▶ Starting Crypto Signal Bot (KuCoin Only)")
    asyncio.run(main())
