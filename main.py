# -*- coding: utf-8 -*-
"""
Hybrid Long-Term Crypto Bot - Weighted Voting + Fib Priority
- Weighted Voting System
- Fib levels (last 100 candles)
- Harmonics + ATR fallback
- 3% threshold for alerts
- 1h cooldown
"""

import os, asyncio, logging, pandas as pd, numpy as np
from datetime import datetime, timedelta
from kucoin.client import Market
from harmonics import detect_harmonics
from telegram import Bot

try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    import ta
    TALIB_AVAILABLE = False

# ================ CONFIG ================
TOKENS = ["BTC","XRP","LINK","ONDO","AVAX","W","ACH","PEPE","PONKE","ICP",
          "FET","ALGO","HBAR","KAS","PYTH","IOTA","WAXL","ETH","ADA"]

MAX_OHLCV = 500
PRICE_ALERT_THRESHOLD = 0.03  # 3% промена
COOLDOWN_MINUTES = 60

EMA_FAST, EMA_SLOW, RSI_PERIOD, ATR_PERIOD = 50, 200, 14, 14
VWAP_PERIOD = 20

CATEGORY_WEIGHTS = {
    "structure": 0.35,
    "momentum": 0.25,
    "volume": 0.15,
    "candles": 0.15,
    "exotic": 0.10
}

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

# ================ UTILS ================
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
async def fetch_kucoin_candles(symbol: str, tf: str = "1d", limit: int = 200):
    interval_map = {"1d":"1day","1w":"1week"}
    interval = interval_map.get(tf, "1day")
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
        logger.error("Error fetching %s: %s", symbol, e)
        return pd.DataFrame()

# ================ INDICATORS ================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    close, high, low, vol = df["close"].values, df["high"].values, df["low"].values, df["volume"].values
    if TALIB_AVAILABLE:
        df["EMA_fast"], df["EMA_slow"] = talib.EMA(close, EMA_FAST), talib.EMA(close, EMA_SLOW)
        df["RSI"] = talib.RSI(close, RSI_PERIOD)
        df["OBV"] = talib.OBV(close, vol)
        df["ATR"] = talib.ATR(high, low, close, ATR_PERIOD)
    else:
        df["EMA_fast"], df["EMA_slow"] = df["close"].ewm(span=EMA_FAST).mean(), df["close"].ewm(span=EMA_SLOW).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi()
        df["ATR"] = df["close"].diff().abs().rolling(ATR_PERIOD).mean()
    df["VWAP"] = (df["close"]*df["volume"]).rolling(VWAP_PERIOD).sum() / (df["volume"].rolling(VWAP_PERIOD).sum() + 1e-9)
    return df

# ================ FIB LEVELS ================
def fib_levels(df: pd.DataFrame, lookback: int = 100):
    recent = df.tail(lookback)
    high, low = recent["high"].max(), recent["low"].min()
    diff = high - low
    return {
        "0.382": high - diff * 0.382,
        "0.5": high - diff * 0.5,
        "0.618": high - diff * 0.618
    }

# ================ VOTING ====================
def weighted_voting_signals(df: pd.DataFrame) -> tuple[str, float]:
    votes = []

    # Structure
    structure_vote = "BUY" if df["close"].iloc[-1] > df["EMA_fast"].iloc[-1] else "SELL"
    votes.append(("structure", structure_vote))

    # Momentum
    rsi = df["RSI"].iloc[-1]
    momentum_vote = "BUY" if rsi < 30 else "SELL" if rsi > 70 else "HOLD"
    votes.append(("momentum", momentum_vote))

    # Volume
    volume_vote = "BUY" if df["OBV"].iloc[-1] > df["OBV"].iloc[-20] else "SELL"
    votes.append(("volume", volume_vote))

    # Candles (placeholder)
    candles_vote = "BUY" if df["close"].iloc[-1] > df["EMA_slow"].iloc[-1] else "SELL"
    votes.append(("candles", candles_vote))

    # Exotic placeholder
    exotic_vote = "HOLD"
    votes.append(("exotic", exotic_vote))

    # Weighted score
    score = 0
    for cat, vote in votes:
        w = CATEGORY_WEIGHTS.get(cat, 0)
        if vote == "BUY": score += w
        elif vote == "SELL": score -= w

    decision = "BUY" if score > 0.2 else "SELL" if score < -0.2 else "HOLD"
    return decision, score

# ================ PRICE TARGETS ================
def hybrid_price_targets(df: pd.DataFrame, last_price: float):
    fibs = fib_levels(df, 100)
    levels = list(fibs.values())

    # Harmonic fallback
    harmonics = detect_harmonics(df)
    for h in harmonics:
        try: levels.append(float(h.split("@")[-1]))
        except: pass

    # ATR fallback
    if not levels:
        atr = df["ATR"].iloc[-1] if "ATR" in df.columns else last_price * 0.01
        levels = [last_price - atr, last_price + atr]

    # Conservative
    buy = max([l for l in levels if l <= last_price], default=last_price)
    sell = min([l for l in levels if l >= last_price], default=last_price)

    return smart_round(buy), smart_round(sell), fibs

# ================ ANALYZE SYMBOL ================
async def analyze_symbol(symbol: str):
    df = await fetch_kucoin_candles(symbol, "1d", MAX_OHLCV)
    if df.empty: return
    df = add_indicators(df).dropna()

    last_price = df["close"].iloc[-1]
    decision, score = weighted_voting_signals(df)
    buy, sell, fibs = hybrid_price_targets(df, last_price)

    # === 3% RULE ===
    key = symbol
    now = datetime.utcnow()

    if key in last_price_sent:
        change = abs(last_price - last_price_sent[key]) / last_price_sent[key]
        if change < PRICE_ALERT_THRESHOLD:
            return  # skip if no 3% change

    if key in last_sent_time and now - last_sent_time[key] < timedelta(minutes=COOLDOWN_MINUTES):
        return  # respect cooldown

    last_price_sent[key] = last_price
    last_sent_time[key] = now

    msg = (f"Strategy: Weighted Voting\n⏰ {now_str()}\n📊 {symbol}\n"
           f"💰 Last Price: {last_price}\n"
           f"✅ Final Decision: {decision} (Score: {round(score,2)})\n"
           f"🛒 Suggested Buy: {buy}\n💵 Suggested Sell: {sell}\n"
           f"📊 Fib Levels: {fibs}\n")
    logger.info(msg)
    asyncio.create_task(send_telegram(msg))

# ================ MAIN LOOP ====================
async def continuous_monitor():
    logger.info("Starting Weighted Hybrid Bot")
    while True:
        tasks = [analyze_symbol(sym+"-USDT") for sym in TOKENS]
        await asyncio.gather(*tasks)
        await asyncio.sleep(60)

if __name__=="__main__":
    asyncio.run(continuous_monitor())
