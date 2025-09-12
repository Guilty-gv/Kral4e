# -*- coding: utf-8 -*-
"""
Hybrid Long-Term Crypto Bot - CONTINUOUS VERSION
- Only selected tokens
- KuCoin public API (async)
- Indicators + ML ensemble
- Telegram alerts with confidence
- Continuous monitoring of new candles
"""

import os, asyncio, logging, pandas as pd, numpy as np
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from joblib import dump, load
from filelock import FileLock
from harmonics import detect_harmonics
from kucoin.client import Market
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split

# TA
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    import ta
    TALIB_AVAILABLE = False

# Telegram
from telegram import Bot

# ================ CONFIG ================
TIMEFRAMES = ["1d","1w"]
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

# Selected tokens
TOKENS = ["BTC","XRP","LINK","ONDO","AVAX","W","ACH","PEPE","PONKE","ICP",
          "FET","ALGO","HBAR","KAS","PYTH","IOTA","WAXL","ETH","ADA"]

# KuCoin API config
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

market_client = Market(
    key=KUCOIN_API_KEY,
    secret=KUCOIN_API_SECRET,
    passphrase=KUCOIN_API_PASSPHRASE,
    is_sandbox=False
) if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE else Market()

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
    interval_map = {"1d":"1day","1w":"1week"}
    if tf not in interval_map:
        logger.warning("Unsupported timeframe %s for %s", tf, symbol)
        return pd.DataFrame()
    interval = interval_map[tf]
    loop = asyncio.get_running_loop()
    try:
        candles = await loop.run_in_executor(
            None, lambda: market_client.get_kline(symbol, interval, limit=limit)
        )
        if not candles:
            logger.info("SKIP %s-%s: Empty response", symbol, tf)
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp","open","close","high","low","volume","turnover"])
        
        # Конвертирање на колоните во бројки за да се избегне TypeError
        for col in ["open","close","high","low","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Отстрани редови каде што конверзијата не успеала
        df = df.dropna(subset=["close","volume"])
        
        # timestamp колоната со errors='coerce' за да се избегне FutureWarning
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp","open","high","low","close","volume"]]
    except Exception as e:
        logger.error("Error fetching %s-%s candles: %s", symbol, tf, e)
        return pd.DataFrame()

# ================ INDICATORS ================
# (Се остава истиот код од оригиналниот пример)

# ================ ML FUNCTIONS ================
# (Се остава истиот код од оригиналниот пример)

# ================ SIGNALS ================
# (Се остава истиот код од оригиналниот пример)

# ================ ANALYZE SYMBOL ================
async def analyze_symbol(symbol: str, tf: str):
    key = (symbol, tf)
    now = datetime.utcnow()
    if key in last_sent_time and now - last_sent_time[key] < timedelta(minutes=COOLDOWN_MINUTES):
        return
    df = await fetch_kucoin_candles(symbol, tf, MAX_OHLCV)
    if df.empty: return
    
    # Проверка на минимален волумен и цената
    if df.empty or df["close"].iloc[-1] * df["volume"].iloc[-1] < MIN_VOLUME_USDT:
        return
    
    df = add_indicators(df).dropna().reset_index(drop=True)
    votes, ml_conf = indicator_votes(df)
    final, buy, sell = combine_votes(votes, ml_conf)
    buy_p, sell_p = suggested_prices(df, final)
    last_price = df["close"].iloc[-1]
    if key in last_price_sent and abs(last_price - last_price_sent[key]) / max(last_price_sent[key], 1e-9) < PRICE_CHANGE_THRESHOLD:
        return
    last_price_sent[key] = last_price
    last_sent_time[key] = now
    log_to_csv(symbol, tf, last_price, final, votes, buy, sell)
    msg = format_message(symbol, tf, last_price, final, buy, sell, buy_p, sell_p, ml_conf)
    logger.info(f"DEBUG: Sending Telegram for {symbol} {tf} at price {last_price}")
    send_telegram(msg)
    logger.info("Analyzed %s %s -> %s (buy:%d sell:%d)", symbol, tf, final, buy, sell)

# ================ MAIN LOOP ================
async def continuous_monitor():
    logger.info("Starting Continuous Hybrid Bot")
    while True:
        tasks = [analyze_symbol(sym+"-USDT", tf) for sym in TOKENS for tf in TIMEFRAMES]
        await asyncio.gather(*tasks)
        await asyncio.sleep(60)  # check every 60 seconds

if __name__=="__main__":
    asyncio.run(continuous_monitor())
