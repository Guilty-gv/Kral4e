# -*- coding: utf-8 -*-
"""
Hybrid Long-Term Crypto Bot - FINAL VERSION
- KuCoin public API (async)
- Indicators: TA-Lib (preferred), fallback to 'ta'
- ML ensemble: LogisticRegression + RandomForest + XGBoost
- ADX, VWAP, Harmonics integrated
- Telegram alerts: FINAL DECISION + dynamic Suggested Buy/Sell Price
- CSV logging, atomic model save/load
"""

import os, asyncio, aiohttp, logging, pandas as pd, numpy as np
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from joblib import dump, load
from filelock import FileLock
from harmonics import detect_harmonics
from kucoin.client import Market

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Try to use TA-Lib for speed/standard; fallback to 'ta' if not available
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    import ta
    TALIB_AVAILABLE = False

# Telegram
from telegram import Bot

# ================ CONFIG ================
TIMEFRAMES = ["1d", "1w"]
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
STOCH_FASTK = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
VWAP_PERIOD = 20
MAX_OHLCV = 500
PRICE_CHANGE_THRESHOLD = 0.01
MIN_VOLUME_USDT = 1000
COOLDOWN_MINUTES = 60

# KuCoin API config
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

MODEL_DIR = ".models"; os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "ensemble.joblib")
MODEL_LOCKPATH = MODEL_PATH + ".lock"
CSV_FILE = "hybrid_bot_log.csv"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

# State
last_price_sent = {}
last_sent_time = {}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hybrid_bot")

# ================ UTIL ================
def now_str(): return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def send_telegram(msg: str):
    if not bot: return
    try: bot.send_message(CHAT_ID, msg)
    except Exception as e: logger.error("Telegram send error: %s", e)

# ================ FETCH ================
async def fetch_kucoin_candles(symbol: str, tf: str, limit: int = 200):
    """
    Асинхрон fetch на KuCoin kline преку официјалниот Market client.
    Поддржани timeframe: 1d, 1w
    Враќа pandas DataFrame со колони: timestamp, open, high, low, close, volume
    """
    interval_map = {"1d":"1day", "1w":"1week"}
    if tf not in interval_map:
        logger.warning("Unsupported timeframe %s for %s. Supported: %s", tf, symbol, list(interval_map.keys()))
        return pd.DataFrame()

    interval = interval_map[tf]
    loop = asyncio.get_running_loop()
    try:
        candles = await loop.run_in_executor(
            None, lambda: market_client.get_kline(symbol, interval, limit=limit)
        )
        if not candles:
            logger.info("SKIP %s-%s: KuCoin returned empty response", symbol, tf)
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp","open","close","high","low","volume","turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp","open","high","low","close","volume"]]
    except Exception as e:
        logger.error("Error fetching %s-%s candles: %s", symbol, tf, e)
        return pd.DataFrame()

# ================ INDICATORS ================
# (Ставете го истиот add_indicators, build_features, ML, signals и suggested_prices код од претходниот main.py)

# ================ ANALYZE TASK ================
async def analyze_symbol(symbol: str, tf: str):
    key = (symbol, tf)
    now = datetime.utcnow()

    # cooldown check
    if key in last_sent_time and now - last_sent_time[key] < timedelta(minutes=COOLDOWN_MINUTES):
        return

    df = await fetch_kucoin_candles(symbol, tf, MAX_OHLCV)
    if df.empty:
        logger.info("SKIP %s-%s: DataFrame empty", symbol, tf)
        return

    # liquidity check
    if df["close"].iloc[-1] * df["volume"].iloc[-1] < MIN_VOLUME_USDT:
        return

    df = add_indicators(df).dropna().reset_index(drop=True)
    votes = indicator_votes(df)
    final, buy, sell = combine_votes(votes)
    buy_p, sell_p = suggested_prices(df, final)
    last_price = df["close"].iloc[-1]

    # price change threshold
    if key in last_price_sent and abs(last_price - last_price_sent[key]) / max(last_price_sent[key], 1e-9) < PRICE_CHANGE_THRESHOLD:
        return

    # update state
    last_price_sent[key] = last_price
    last_sent_time[key] = now

    # log
    log_to_csv(symbol, tf, last_price, final, votes, buy, sell)
    msg = format_message(symbol, tf, last_price, final, buy, sell, buy_p, sell_p)

    logger.info(f"DEBUG: Trying to send Telegram for {symbol} {tf} at price {last_price}")
    send_telegram(msg)
    logger.info("Analyzed %s %s -> %s (buy:%d sell:%d)", symbol, tf, final, buy, sell)

# ================ MAIN ================
async def main():
    logger.info("Starting Hybrid Bot FINAL")
    # Filter supported KuCoin USDT symbols
    supported_symbols = [s['symbol'] for s in market_client.get_symbols() if s['enableTrading'] and s['quoteCurrency']=='USDT']
    logger.info("Supported symbols: %s", supported_symbols)
    async with aiohttp.ClientSession() as session:
        tasks = [analyze_symbol(sym, tf) for sym in supported_symbols for tf in TIMEFRAMES]
        await asyncio.gather(*tasks)
    logger.info("Run finished.")

if __name__=="__main__":
    asyncio.run(main())
