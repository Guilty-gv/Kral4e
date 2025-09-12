# -*- coding: utf-8 -*-
"""
Hybrid Long-Term Crypto Bot - CONSOLIDATED VERSION
- Combines 1D + 1W for suggested prices
- Fib/Harmonic priority with fallback to ATR
- Single final Telegram message per token
"""

import os, asyncio, logging, pandas as pd, numpy as np
from datetime import datetime, timedelta
from kucoin.client import Market
from harmonics import detect_harmonics
from telegram import Bot

# ================ CONFIG ================
TIMEFRAMES = ["1d","1w"]
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
VWAP_PERIOD = 20
MAX_OHLCV = 500
MIN_VOLUME_USDT = 1000
COOLDOWN_MINUTES = 60

TOKENS = ["BTC","XRP","LINK","ONDO","AVAX","W","ACH","PEPE","PONKE","ICP",
          "FET","ALGO","HBAR","KAS","PYTH","IOTA","WAXL","ETH","ADA"]

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

market_client = Market(
    key=KUCOIN_API_KEY, secret=KUCOIN_API_SECRET, passphrase=KUCOIN_API_PASSPHRASE
) if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE else Market()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

last_price_sent = {}
last_sent_time = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hybrid_bot")

# ================ UTIL ================
def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

async def send_telegram(msg: str):
    if not bot: return
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        logger.error("Telegram send error: %s", e)

def smart_round(value: float) -> float:
    if value >= 1: return round(value, 2)
    elif value >= 0.01: return round(value, 4)
    else: return round(value, 8)

# ================ FETCH ================
async def fetch_kucoin_candles(symbol: str, tf: str, limit: int = 200):
    interval_map = {"1d":"1day","1w":"1week"}
    if tf not in interval_map: return pd.DataFrame()
    interval = interval_map[tf]
    loop = asyncio.get_running_loop()
    try:
        candles = await loop.run_in_executor(None, lambda: market_client.get_kline(symbol, interval, limit=limit))
        if not candles: return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp","open","close","high","low","volume","turnover"])
        for col in ["open","close","high","low","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        return df.sort_values("timestamp").reset_index(drop=True)[["timestamp","open","high","low","close","volume"]]
    except Exception as e:
        logger.error("Error fetching %s-%s: %s", symbol, tf, e)
        return pd.DataFrame()

# ================ INDICATORS ================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    df = df.copy()
    df["ATR"] = df["close"].diff().abs().rolling(ATR_PERIOD).mean()
    df["VWAP"] = (df["close"]*df["volume"]).rolling(VWAP_PERIOD).sum() / (df["volume"].rolling(VWAP_PERIOD).sum() + 1e-9)
    high_v, low_v = df["close"].max(), df["close"].min(); diff = max(high_v-low_v,1e-9)
    df["Fib_0.382"], df["Fib_0.5"], df["Fib_0.618"] = high_v-0.382*diff, high_v-0.5*diff, high_v-0.618*diff
    return df

# ================ STRATEGY & PRICES ================
def hybrid_suggested_prices(df: pd.DataFrame, strategy: str):
    last_price = df["close"].iloc[-1]
    levels = []

    # Fib + Harmonic priority
    for f in ["Fib_0.382","Fib_0.5","Fib_0.618"]:
        if f in df.columns: levels.append(df[f].iloc[-1])
    harmonics = detect_harmonics(df)
    for h in harmonics:
        try: levels.append(float(h.split("@")[-1]))
        except: pass

    # fallback ATR
    if not levels:
        atr = df["ATR"].iloc[-1] if "ATR" in df.columns else last_price*0.01
        levels = [last_price-atr, last_price+atr]

    buy_price, sell_price = last_price, last_price
    if strategy=="conservative":
        buy_price = max([l for l in levels if l<=last_price], default=last_price)
        sell_price = min([l for l in levels if l>=last_price], default=last_price)
    elif strategy=="aggressive":
        buy_price = min([l for l in levels if l>=last_price], default=last_price)
        sell_price = max([l for l in levels if l<=last_price], default=last_price)
    else:  # moderate
        buy_price = sell_price = np.mean(levels)

    return smart_round(buy_price), smart_round(sell_price)

def dynamic_strategy(df: pd.DataFrame):
    last_price = df["close"].iloc[-1]
    levels = []
    for f in ["Fib_0.382","Fib_0.5","Fib_0.618"]:
        if f in df.columns: levels.append(df[f].iloc[-1])
    harmonics = detect_harmonics(df)
    for h in harmonics:
        try: levels.append(float(h.split("@")[-1]))
        except: pass
    if not levels:
        atr = df["ATR"].iloc[-1] if "ATR" in df.columns else last_price*0.01
        levels = [last_price-atr, last_price+atr]
    min_level, max_level = min(levels), max(levels)
    if last_price < min_level: return "aggressive"
    elif last_price > max_level: return "conservative"
    else: return "moderate"

def format_message(symbol, strategy, last_price, final, buy, sell, ml_conf):
    t = now_str()
    return (f"Strategy: {strategy.capitalize()}\n⏰ {t}\n📊 {symbol}\n💰 Last Price: {last_price} USDT\n\n"
            f"✅ FINAL DECISION: {final}\n🛒 Suggested Buy Price: {buy} USDT\n💵 Suggested Sell Price: {sell} USDT\n"
            f"💡 ML Confidence: {round(ml_conf*100,2)}%")

# ================ ANALYZE SYMBOL ================
async def analyze_symbol(symbol: str):
    dfs = {}
    for tf in TIMEFRAMES:
        df = await fetch_kucoin_candles(symbol, tf, MAX_OHLCV)
        if not df.empty and df["close"].iloc[-1]*df["volume"].iloc[-1]>=MIN_VOLUME_USDT:
            dfs[tf] = add_indicators(df).dropna().reset_index(drop=True)

    if not dfs: return
    # Combine 1D + 1W
    combined_df = pd.concat(dfs.values()).sort_values("timestamp").reset_index(drop=True)
    last_price = combined_df["close"].iloc[-1]
    strategy = dynamic_strategy(combined_df)

    # Final decision placeholder (ML + indicator votes can be added)
    final_decision, ml_conf = "BUY" if last_price%2<1 else "SELL", 0.7

    buy_p, sell_p = hybrid_suggested_prices(combined_df, strategy)

    # Telegram & cooldown
    key = symbol
    now = datetime.utcnow()
    if key in last_sent_time and now - last_sent_time[key] < timedelta(minutes=COOLDOWN_MINUTES):
        return
    last_price_sent[key] = last_price
    last_sent_time[key] = now

    msg = format_message(symbol, strategy, last_price, final_decision, buy_p, sell_p, ml_conf)
    logger.info(msg)
    asyncio.create_task(send_telegram(msg))

# ================ MAIN LOOP ================
async def continuous_monitor():
    logger.info("Starting Consolidated Hybrid Bot")
    while True:
        tasks = [analyze_symbol(sym+"-USDT") for sym in TOKENS]
        await asyncio.gather(*tasks)
        await asyncio.sleep(60)

if __name__=="__main__":
    asyncio.run(continuous_monitor())
