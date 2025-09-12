# -*- coding: utf-8 -*-
"""
Hybrid Bot — TEST (15m / 1h)
Purpose: verify end-to-end pipeline (fetch -> analyze -> Telegram)
Switch back to 1d/1w after verification.
"""

import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import dump, load
from filelock import FileLock

# KuCoin SDK
from kucoin.client import Market

# harmonics detector (must exist in project)
from harmonics import detect_harmonics

# TA / fallback
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    import ta
    TALIB_AVAILABLE = False

# Telegram
from telegram import Bot

# ================ CONFIG ================
TIMEFRAMES = ["15m", "1h"]                # TEST intervals
MAX_OHLCV = 500
MIN_VOLUME_USDT = 10     # lower for tests
PRICE_CHANGE_THRESHOLD = 0.0  # disable price threshold for tests
COOLDOWN_MINUTES = 0          # disable cooldown for tests

# KuCoin API
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE:
    market_client = Market(
        key=KUCOIN_API_KEY,
        secret=KUCOIN_API_SECRET,
        passphrase=KUCOIN_API_PASSPHRASE,
        is_sandbox=False
    )
else:
    market_client = Market()

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

# State + logging
last_price_sent = {}
last_sent_time = {}
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hybrid_bot_test")

def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def send_telegram(msg: str):
    if not bot:
        logger.info("Telegram not configured, skipping send.")
        return
    try:
        bot.send_message(CHAT_ID, msg)
        logger.info("Telegram sent.")
    except Exception as e:
        logger.error("Telegram send error: %s", e)

# ================ FETCH (SDK via run_in_executor) ================
async def fetch_kucoin_candles(symbol: str, tf: str, limit: int = 200):
    """
    Uses market_client.get_kline(symbol, interval, limit=...)
    symbol must be in 'BTC-USDT' format (as returned by get_symbol_list()).
    Supports: 15m -> 15min, 1h -> 1hour
    Returns DataFrame with timestamp, open, high, low, close, volume
    """
    interval_map = {"15m": "15min", "1h": "1hour"}
    if tf not in interval_map:
        logger.warning("Unsupported timeframe %s for %s", tf, symbol)
        return pd.DataFrame()

    interval = interval_map[tf]
    loop = asyncio.get_running_loop()

    try:
        # run sync client in executor
        candles = await loop.run_in_executor(None, lambda: market_client.get_kline(symbol, interval, limit=limit))
        if not candles:
            logger.info("SKIP %s %s: empty response", symbol, tf)
            return pd.DataFrame()

        # KuCoin returns: [time, open, close, high, low, volume, turnover]
        df = pd.DataFrame(candles, columns=["timestamp","open","close","high","low","volume","turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp","open","high","low","close","volume"]]
    except Exception as e:
        # log full exception for debugging
        logger.error("Error fetching %s %s: %s", symbol, tf, e)
        return pd.DataFrame()

# ================ LIGHTWEIGHT INDICATORS (test-friendly) ================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["ret1"] = df["close"].pct_change()
    df["vol20"] = df["ret1"].rolling(20).std()
    # simple EMA50/200 fallback for quick test (don't require talib)
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()
    # RSI quick
    if TALIB_AVAILABLE:
        df["RSI"] = talib.RSI(df["close"].values, timeperiod=14)
    else:
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    return df

def simple_signal_from_df(df: pd.DataFrame):
    """
    Very simple voting for tests: EMA cross + RSI + last candle
    Returns final decision string and brief detail.
    """
    if df.empty or len(df) < 5:
        return "HOLD", "insufficient data"
    last = df.iloc[-1]
    votes = []
    votes.append("BUY" if last["EMA50"] > last["EMA200"] else "SELL")
    if last["RSI"] < 30:
        votes.append("BUY")
    elif last["RSI"] > 70:
        votes.append("SELL")
    # candle body
    o, c = last["open"], last["close"]
    if c > o:
        votes.append("BUY")
    else:
        votes.append("SELL")
    buy = votes.count("BUY"); sell = votes.count("SELL")
    if buy > sell:
        return "BUY", f"buy:{buy} sell:{sell}"
    if sell > buy:
        return "SELL", f"buy:{buy} sell:{sell}"
    return "HOLD", f"buy:{buy} sell:{sell}"

def suggested_price_for_test(df, decision):
    last = df["close"].iloc[-1]
    if decision == "BUY":
        return round(last * 0.995, 6), round(last * 1.02, 6)
    if decision == "SELL":
        return round(last * 0.98, 6), round(last * 1.005, 6)
    return round(last * 0.99, 6), round(last * 1.01, 6)

# ================ ANALYZE TASK ================
async def analyze_symbol(symbol: str, tf: str):
    key = (symbol, tf)
    logger.info("Analyzing %s %s", symbol, tf)

    df = await fetch_kucoin_candles(symbol, tf, limit=MAX_OHLCV)
    if df.empty:
        logger.info("SKIP %s %s: DataFrame empty", symbol, tf)
        return

    # low-liquidity filter (relaxed for test)
    try:
        if df["close"].iloc[-1] * df["volume"].iloc[-1] < MIN_VOLUME_USDT:
            logger.info("SKIP %s %s: low liquidity (%.2f USDT)", symbol, tf, df["close"].iloc[-1] * df["volume"].iloc[-1])
            return
    except Exception as e:
        logger.warning("Volume check failed for %s %s: %s", symbol, tf, e)

    df = add_indicators(df).dropna().reset_index(drop=True)
    decision, detail = simple_signal_from_df(df)
    buy_p, sell_p = suggested_price_for_test(df, decision)
    last_price = df["close"].iloc[-1]

    # no thresholds in test mode (but keep logs)
    logger.info("DECISION %s %s -> %s (%s) price: %.6f buy:%s sell:%s", symbol, tf, decision, detail, last_price, buy_p, sell_p)

    # send Telegram message (test format)
    msg = (
        f"⏰ {now_str()}\n"
        f"📊 {symbol} | {tf}\n"
        f"💰 Last Price: {round(float(last_price),6)} USDT\n\n"
        f"✅ FINAL DECISION: {decision}\n"
        f"🛒 Suggested Buy Price: {buy_p} USDT\n"
        f"💵 Suggested Sell Price: {sell_p} USDT\n\n"
        f"(debug detail: {detail})"
    )
    send_telegram(msg)

# ================ MAIN ================
async def main():
    logger.info("Starting Hybrid Bot TEST (15m/1h)")

    # get all symbols from KuCoin and filter to enableTrading + USDT quote
    try:
        all_sym = market_client.get_symbol_list()  # returns e.g. ["BTC-USDT", ...]
    except Exception as e:
        logger.error("Failed to fetch symbol list from KuCoin: %s", e)
        return

    supported_symbols = []
    for s in all_sym:
        try:
            info = market_client.get_symbol(s)
            if info.get("enableTrading") and info.get("quoteCurrency") == "USDT":
                supported_symbols.append(s)  # keep form "BTC-USDT"
        except Exception as e:
            logger.debug("get_symbol failed for %s: %s", s, e)

    if not supported_symbols:
        logger.error("No supported USDT symbols found. Exiting.")
        return

    logger.info("Using %d supported symbols (sample): %s", len(supported_symbols), supported_symbols[:10])

    # build tasks
    tasks = []
    for sym in supported_symbols:
        for tf in TIMEFRAMES:
            tasks.append(analyze_symbol(sym, tf))

    # run in bounded parallelism to avoid blowing up the API
    semaphore = asyncio.Semaphore(10)

    async def sem_task(t):
        async with semaphore:
            await t

    await asyncio.gather(*[sem_task(t) for t in tasks])
    logger.info("Test run finished.")

if __name__ == "__main__":
    asyncio.run(main())
