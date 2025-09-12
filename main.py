# -*- coding: utf-8 -*-
"""
Hybrid Long-Term Crypto Bot - CONSOLOIDATED VERSION
- Selected tokens
- KuCoin public API (async)
- Indicators + ML ensemble
- Telegram alerts with priority Fib/Harmonic levels
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

try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    import ta
    TALIB_AVAILABLE = False

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

TOKENS = ["BTC","XRP","LINK","ONDO","AVAX","W","ACH","PEPE","PONKE","ICP",
          "FET","ALGO","HBAR","KAS","PYTH","IOTA","WAXL","ETH","ADA"]

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE:
    market_client = Market(key=KUCOIN_API_KEY, secret=KUCOIN_API_SECRET, passphrase=KUCOIN_API_PASSPHRASE)
else:
    market_client = Market()

MODEL_DIR = ".models"; os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "ensemble.joblib")
MODEL_LOCKPATH = MODEL_PATH + ".lock"
CSV_FILE = "hybrid_bot_log.csv"

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

# ================ FETCH CANDLES ================
async def fetch_kucoin_candles(symbol: str, tf: str, limit: int = 200):
    interval_map = {"1d":"1day","1w":"1week"}
    if tf not in interval_map: return pd.DataFrame()
    interval = interval_map[tf]
    loop = asyncio.get_running_loop()
    try:
        candles = await loop.run_in_executor(
            None, lambda: market_client.get_kline(symbol, interval, limit=limit)
        )
        if not candles: return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp","open","close","high","low","volume","turnover"])
        for col in ["open","close","high","low","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp","open","high","low","close","volume"]]
    except Exception as e:
        logger.error("Error fetching %s-%s candles: %s", symbol, tf, e)
        return pd.DataFrame()

# ================ INDICATORS ================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    df = df.copy()
    close, high, low, vol = df["close"].values, df["high"].values, df["low"].values, df["volume"].values
    if TALIB_AVAILABLE:
        df["EMA_fast"], df["EMA_slow"] = talib.EMA(close, EMA_FAST), talib.EMA(close, EMA_SLOW)
        df["RSI"] = talib.RSI(close, RSI_PERIOD)
        stoch_k, stoch_d = talib.STOCH(high, low, close, STOCH_FASTK, 3,3); df["%K"], df["%D"] = stoch_k, stoch_d
        tr = talib.TRANGE(high, low, close); df["ATR"] = pd.Series(tr).rolling(ATR_PERIOD).mean().values
    else:
        df["EMA_fast"], df["EMA_slow"] = df["close"].ewm(span=EMA_FAST).mean(), df["close"].ewm(span=EMA_SLOW).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi()
        df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD).average_true_range()
    high_v, low_v = df["close"].max(), df["close"].min(); diff = max(high_v-low_v,1e-9)
    df["Fib_0.382"], df["Fib_0.5"], df["Fib_0.618"] = high_v-0.382*diff, high_v-0.5*diff, high_v-0.618*diff
    return df

# ================ ML & VOTES ================
def build_features(df: pd.DataFrame):
    feats = pd.DataFrame(index=df.index)
    feats["EMA_fast"], feats["EMA_slow"], feats["EMA_ratio"] = df["EMA_fast"], df["EMA_slow"], df["EMA_fast"]/(df["EMA_slow"]+1e-9)
    feats["RSI"] = df["RSI"]
    feats = feats.replace([np.inf,-np.inf], np.nan).dropna()
    return feats

def run_ml_predict(df):
    feats = build_features(df)
    if len(feats)<30: return [], 0.5
    model = load_model_safe(MODEL_PATH)
    if model is None: return [], 0.5
    X_latest = feats.iloc[-1:].values
    pred = int(model.predict(X_latest)[0])
    conf = max(model.predict_proba(X_latest)[0]) if hasattr(model, "predict_proba") else 0.7
    return ["BUY"] if pred==1 else ["SELL"], conf

def indicator_votes(df: pd.DataFrame):
    if df.empty: return {}, 0.5
    votes={}
    row = df.iloc[-1]
    votes["EMA"] = ["BUY"] if row["EMA_fast"]>row["EMA_slow"] else ["SELL"]
    votes["ML"], ml_conf = run_ml_predict(df)
    votes["Harmonic"] = []
    harmonics = detect_harmonics(df)
    for p in harmonics:
        if "BUY" in p: votes["Harmonic"].extend(["BUY","BUY"])
        elif "SELL" in p: votes["Harmonic"].extend(["SELL","SELL"])
    return votes, ml_conf

def combine_votes(votes_dict, ml_conf):
    votes_flat=[]; [votes_flat.extend(v) for v in votes_dict.values()]
    buy, sell=votes_flat.count("BUY"), votes_flat.count("SELL")
    if buy>sell: return "BUY", buy, sell
    if sell>buy: return "SELL", buy, sell
    return "HOLD", buy, sell

# ================ HYBRID PRICE ================
def hybrid_suggested_price(df_daily: pd.DataFrame, df_weekly: pd.DataFrame, vote: str, strategy: str = "conservative"):
    last_price = df_daily["close"].iloc[-1]
    levels = []
    for df in [df_daily, df_weekly]:
        for f in ["Fib_0.382","Fib_0.5","Fib_0.618"]:
            if f in df.columns: levels.append(float(df[f].iloc[-1]))
        harmonics = detect_harmonics(df)
        for h in harmonics:
            try: levels.append(float(h.split("@")[-1]))
            except: pass
    if not levels:
        atr = df_daily["ATR"].iloc[-1] if "ATR" in df_daily.columns else last_price*0.01
        levels = [last_price-atr, last_price+atr]
    buy_candidates = [l for l in levels if l < last_price]
    sell_candidates = [l for l in levels if l > last_price]
    if strategy.lower()=="conservative":
        buy_price = max(buy_candidates) if buy_candidates else last_price
        sell_price = min(sell_candidates) if sell_candidates else last_price
    elif strategy.lower()=="aggressive":
        buy_price = min(buy_candidates) if buy_candidates else last_price
        sell_price = max(sell_candidates) if sell_candidates else last_price
    else:  # moderate
        buy_price = sum(buy_candidates)/len(buy_candidates) if buy_candidates else last_price
        sell_price = sum(sell_candidates)/len(sell_candidates) if sell_candidates else last_price
    return smart_round(buy_price), smart_round(sell_price)

def log_to_csv(symbol, tf, price, final, votes, buy, sell):
    flat=[]
    for k,v in votes.items(): flat.extend([f"{k}:{s}" for s in v])
    entry={"timestamp": now_str(), "symbol": symbol, "interval": tf, "price": round(float(price),6),
           "decision": final, "buy_votes": buy, "sell_votes": sell, "signals":",".join(flat)}
    pd.DataFrame([entry]).to_csv(CSV_FILE, mode="a", index=False, header=not os.path.exists(CSV_FILE))

# ================ ANALYZE SYMBOL ================
async def analyze_symbol_combined(symbol: str, strategy: str = "conservative"):
    now = datetime.utcnow()
    df_daily = await fetch_kucoin_candles(symbol, "1d", MAX_OHLCV)
    df_weekly = await fetch_kucoin_candles(symbol, "1w", MAX_OHLCV)
    if df_daily.empty or df_weekly.empty: return
    if df_daily["close"].iloc[-1] * df_daily["volume"].iloc[-1] < MIN_VOLUME_USDT: return
    df_daily = add_indicators(df_daily).dropna().reset_index(drop=True)
    df_weekly = add_indicators(df_weekly).dropna().reset_index(drop=True)
    votes, ml_conf = indicator_votes(df_daily)
    final_decision, buy_votes, sell_votes = combine_votes(votes, ml_conf)
    buy_p, sell_p = hybrid_suggested_price(df_daily, df_weekly, final_decision, strategy)
    last_price = df_daily["close"].iloc[-1]
    log_to_csv(symbol, "1d+1w", last_price, final_decision, votes, buy_votes, sell_votes)
    msg = f"{strategy.capitalize()} strategy\n⏰ {now_str()}\n📊 {symbol}\n💰 Last Price: {last_price} USDT\n\n"\
          f"✅ FINAL DECISION: {final_decision}\n🛒 Suggested Buy Price: {buy_p} USDT\n💵 Suggested Sell Price: {sell_p} USDT\n"\
          f"💡 ML Confidence: {round(ml_conf*100,2)}%"
    logger.info(f"DEBUG: Sending Telegram for {symbol} at price {last_price}")
    asyncio.create_task(send_telegram(msg))

# ================ MAIN LOOP ================
async def continuous_monitor(strategy: str = "conservative"):
    logger.info("Starting Continuous Hybrid Bot")
    while True:
        tasks = [analyze_symbol_combined(sym+"-USDT", strategy) for sym in TOKENS]
        await asyncio.gather(*tasks)
        await asyncio.sleep(60)

if __name__=="__main__":
    asyncio.run(continuous_monitor(strategy="conservative"))
