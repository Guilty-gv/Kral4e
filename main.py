# -*- coding: utf-8 -*-
"""
Hybrid Long-Term Crypto Bot - TEST VERSION
- KuCoin public API (async)
- Indicators: TA-Lib (preferred), fallback to 'ta'
- ML ensemble: LogisticRegression + RandomForest + XGBoost
- ADX, VWAP, Harmonics integrated
- Telegram alerts: FINAL DECISION + dynamic Suggested Buy/Sell Price
- CSV logging, atomic model save/load
"""

import os, asyncio, logging, pandas as pd, numpy as np
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

# TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    import ta
    TALIB_AVAILABLE = False

# Telegram
from telegram import Bot

# ================ CONFIG ================
KUCOIN_SYMBOLS = [
    "BTCUSDT","XRPUSDT","LINKUSDT","ALGOUSDT","AVAXUSDT",
    "FETUSDT","IOTAUSDT","HBARUSDT","ACHUSDT","WAXLUSDT","WUSDT",
    "KASUSDT","ONDOUSDT","PEPEUSDT","PONKEUSDT"
]
TIMEFRAMES = ["15m","1h"]  # тест
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

MODEL_DIR = ".models"
os.makedirs(MODEL_DIR, exist_ok=True)
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
def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def send_telegram(msg: str):
    if not bot:
        logger.info("Telegram bot not configured, skipping message")
        return
    try:
        bot.send_message(CHAT_ID, msg)
    except Exception as e:
        logger.error("Telegram send error: %s", e)

# ================ FETCH ================
async def fetch_kucoin_candles(symbol: str, tf: str, limit: int = 200):
    interval_map = {
        "15m": "15min",
        "1h": "1hour"
    }

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

# ================ ANALYZE TASK ================
async def analyze_symbol(symbol: str, tf: str, session: asyncio.AbstractEventLoop):
    key = (symbol, tf)
    now = datetime.utcnow()

    if key in last_sent_time and now - last_sent_time[key] < timedelta(minutes=COOLDOWN_MINUTES):
        logger.info("SKIP %s-%s: Cooldown", symbol, tf)
        return

    df = await fetch_kucoin_candles(symbol, tf, MAX_OHLCV)
    if df.empty:
        logger.info("SKIP %s-%s: DataFrame empty", symbol, tf)
        return

    if df["close"].iloc[-1] * df["volume"].iloc[-1] < MIN_VOLUME_USDT:
        logger.info("SKIP %s-%s: Low liquidity", symbol, tf)
        return

    # Тука би се вметнале indicator-и, ML и suggested_prices како претходно
    # За тест, само праќаме основна порака
    last_price = df["close"].iloc[-1]
    last_sent_time[key] = now
    last_price_sent[key] = last_price

    msg = f"⏰ {now_str()}\n📊 {symbol} | {tf}\n💰 Last Price: {last_price} USDT\n✅ TEST ALERT"
    logger.info("DEBUG: Sending Telegram for %s-%s", symbol, tf)
    send_telegram(msg)

# ================ MAIN ================
async def main():
    logger.info("Starting Hybrid Bot TEST")
    tasks = [analyze_symbol(sym, tf, None) for sym in KUCOIN_SYMBOLS for tf in TIMEFRAMES]
    await asyncio.gather(*tasks)
    logger.info("Run finished.")

if __name__ == "__main__":
    asyncio.run(main())
