# -*- coding: utf-8 -*-
import asyncio
from datetime import datetime, timedelta
import logging
import os
import pandas as pd
from kucoin.client import Market
from telegram import Bot

# ================= CONFIG =================
TIMEFRAME_WEIGHTS = {"1h": 0.2, "4h": 0.3, "1d": 0.5}
TEST_TELEGRAM_MODE = True  # Forced messages на GitHub
PRICE_ALERT_THRESHOLD = 0.001
COOLDOWN_MINUTES = 5
MAX_OHLCV = 200

last_price_sent = {}
last_sent_time = {}
adaptive_weights = {}

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# ================= KUCOIN =================
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")
client = Market(key=KUCOIN_API_KEY, secret=KUCOIN_API_SECRET, passphrase=KUCOIN_API_PASSPHRASE)

# ================= TELEGRAM =================
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=BOT_TOKEN)

async def send_telegram(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        logger.error("Telegram token or chat_id is missing!")
        return
    logger.info("Sending Telegram message:\n" + msg)
    await bot.send_message(chat_id=CHAT_ID, text=msg)

# ================= FETCH KLINES =================
async def fetch_kucoin_candles(symbol: str, timeframe: str, limit: int):
    kline_map = {"1h": "1hour", "4h": "4hour", "1d": "1day"}
    interval = kline_map.get(timeframe, "1day")
    loop = asyncio.get_event_loop()

    def get_kline():
        return client.get_kline(symbol=symbol, kline_type=interval, limit=limit)

    data = await loop.run_in_executor(None, get_kline)
    df = pd.DataFrame(data, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
    df[["time","open","close","high","low","volume","turnover"]] = df[["time","open","close","high","low","volume","turnover"]].apply(pd.to_numeric, errors='coerce')
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

# ================= INDICATORS =================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
    return df

def weighted_voting_signals(df: pd.DataFrame, token: str):
    return "HOLD", 0.0

def hybrid_price_targets(df: pd.DataFrame, last_price: float):
    buy = last_price * 0.98
    sell = last_price * 1.02
    fibs = [last_price * 0.95, last_price, last_price * 1.05]  # секогаш 3 елементи
    return buy, sell, fibs

def update_adaptive_weights(token: str, decision: str, price: float):
    if token not in adaptive_weights:
        adaptive_weights[token] = {
            "structure": 0.36,
            "momentum": 0.24,
            "volume": 0.14,
            "candles": 0.16,
            "exotic": 0.10
        }

# ================= WEIGHTED VOTING =================
def multi_timeframe_voting(dfs: dict, token: str):
    total_score = 0
    for tf, df in dfs.items():
        decision, score = weighted_voting_signals(df, token)
        total_score += score * TIMEFRAME_WEIGHTS.get(tf, 0.33)
    last_close = dfs["1d"]["close"].iloc[-1]
    atr = dfs["1d"]["ATR"].iloc[-1] if "ATR" in dfs["1d"].columns else 0.01
    threshold = max(0.2, atr / last_close)
    final_decision = "BUY" if total_score > threshold else "SELL" if total_score < -threshold else "HOLD"
    return final_decision, total_score

# ================= ANALYZE SYMBOL =================
async def analyze_symbol(symbol: str):
    token = symbol.replace("-USDT","")
    if token not in adaptive_weights:
        adaptive_weights[token] = {
            "structure": 0.36,
            "momentum": 0.24,
            "volume": 0.14,
            "candles": 0.16,
            "exotic": 0.10
        }

    dfs = {}
    for tf in TIMEFRAME_WEIGHTS.keys():
        df = await fetch_kucoin_candles(symbol, tf, MAX_OHLCV)
        if df.empty: return
        dfs[tf] = add_indicators(df).dropna()
    last_price = dfs["1d"]["close"].iloc[-1]

    decision, score = multi_timeframe_voting(dfs, token)
    buy, sell, fibs = hybrid_price_targets(dfs["1d"], last_price)

    key = symbol
    now = datetime.utcnow()

    send_alert = TEST_TELEGRAM_MODE or (
        abs(last_price - last_price_sent.get(key,last_price)) / max(last_price_sent.get(key,last_price),1e-9) >= PRICE_ALERT_THRESHOLD
        and (key not in last_sent_time or now - last_sent_time[key] >= timedelta(minutes=COOLDOWN_MINUTES))
    )

    if not send_alert:
        return

    last_price_sent[key] = last_price
    last_sent_time[key] = now

    msg = (
        f"Strategy: Production-Ready Hybrid\n"
        f"⏰ Time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"📊 Token: {symbol}\n"
        f"💰 Last Price: {last_price:.5f}\n"
        f"✅ Decision: {decision} (Score: {score:.5f})\n"
        f"🛒 Suggested Buy: {buy:.5f}\n"
        f"💵 Suggested Sell: {sell:.5f}\n"
        f"📊 Fib Levels: {{ '0.382': {fibs[0]:.5f}, '0.5': {fibs[1]:.5f}, '0.618': {fibs[2]:.5f} }}\n"
        f"⚖️ Adaptive Weights: {{ "
        f"'structure': {adaptive_weights[token].get('structure',0):.5f}, "
        f"'momentum': {adaptive_weights[token].get('momentum',0):.5f}, "
        f"'volume': {adaptive_weights[token].get('volume',0):.5f}, "
        f"'candles': {adaptive_weights[token].get('candles',0):.5f}, "
        f"'exotic': {adaptive_weights[token].get('exotic',0):.5f} }}"
    )

    await send_telegram(msg)
    update_adaptive_weights(token, decision, last_price)

# ================= MAIN LOOP =================
async def main_loop():
    symbols = ["BTC-USDT", "ETH-USDT"]
    tasks = [asyncio.create_task(analyze_symbol(symbol)) for symbol in symbols]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main_loop())
