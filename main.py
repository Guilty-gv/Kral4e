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

TOKENS = ["BTC","XRP","LINK","ONDO","AVAX","W","ACH","PEPE","PONKE","ICP",
          "FET","ALGO","HBAR","KAS","PYTH","IOTA","WAXL","ETH","ADA"]

# KuCoin API config
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE:
    market_client = Market(
        key=KUCOIN_API_KEY,
        secret=KUCOIN_API_SECRET,
        passphrase=KUCOIN_API_PASSPHRASE
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
def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

async def send_telegram(msg: str):
    if not bot:
        return
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        logger.error("Telegram send error: %s", e)
        
        def smart_round(value: float) -> float:
    """Dynamic rounding според цената на токенот."""
    if value >= 1:
        return round(value, 2)   # за BTC, ETH итн.
    elif value >= 0.01:
        return round(value, 4)   # за midcap токени
    else:
        return round(value, 8)   # за микро-токени како PEPE, SHIB


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
        macd, macdsignal, macdhist = talib.MACD(close); df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd, macdsignal, macdhist
        upper, middle, lower = talib.BBANDS(close, 20); df["BB_upper"], df["BB_middle"], df["BB_lower"] = upper, middle, lower; df["BB_bw"] = (upper-lower)/(df["close"]+1e-9)
        tr = talib.TRANGE(high, low, close); df["ATR"] = pd.Series(tr).rolling(ATR_PERIOD).mean().values
        df["OBV"] = talib.OBV(close, vol)
        df["ADX"] = talib.ADX(high, low, close, ADX_PERIOD); df["+DI"] = talib.PLUS_DI(high, low, close, ADX_PERIOD); df["-DI"] = talib.MINUS_DI(high, low, close, ADX_PERIOD)
    else:
        df["EMA_fast"], df["EMA_slow"] = df["close"].ewm(span=EMA_FAST).mean(), df["close"].ewm(span=EMA_SLOW).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi()
        st = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], STOCH_FASTK); df["%K"], df["%D"] = st.stoch(), st.stoch_signal()
        macd = ta.trend.MACD(df["close"]); df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
        bb = ta.volatility.BollingerBands(df["close"]); df["BB_upper"], df["BB_lower"] = bb.bollinger_hband(), bb.bollinger_lband(); df["BB_bw"] = (df["BB_upper"]-df["BB_lower"])/(df["close"]+1e-9)
        df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD).average_true_range()
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        df["ADX"] = abs(df["close"].diff().fillna(0).rolling(ADX_PERIOD).mean())
        df["+DI"], df["-DI"] = df["close"].diff().clip(lower=0).rolling(ADX_PERIOD).mean(), (-df["close"].diff()).clip(lower=0).rolling(ADX_PERIOD).mean()
    df["VWAP"] = (df["close"]*df["volume"]).rolling(VWAP_PERIOD).sum() / (df["volume"].rolling(VWAP_PERIOD).sum() + 1e-9)
    df["BullishEngulfing"] = (df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1))
    df["BearishEngulfing"] = (df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1))
    df["Doji"] = abs(df["close"] - df["open"]) < (df["high"] - df["low"])*0.1
    df["Hammer"] = (df["close"]-df["low"])>2*(abs(df["close"]-df["open"]))
    high_v, low_v = df["close"].max(), df["close"].min(); diff = max(high_v-low_v,1e-9)
    df["Fib_0.382"], df["Fib_0.5"], df["Fib_0.618"] = high_v-0.382*diff, high_v-0.5*diff, high_v-0.618*diff
    df["ret1"], df["vol20"] = df["close"].pct_change(), df["close"].pct_change().rolling(20).std()
    return df

# ================ ML FUNCTIONS ================
def build_features(df: pd.DataFrame):
    feats = pd.DataFrame(index=df.index)
    feats["EMA_fast"], feats["EMA_slow"], feats["EMA_ratio"] = df["EMA_fast"], df["EMA_slow"], df["EMA_fast"]/(df["EMA_slow"]+1e-9)
    feats["RSI"], feats["MACD_hist"], feats["StochK"], feats["StochD"], feats["BB_bw"], feats["ATR"], feats["OBV"] = df["RSI"], df["MACD_hist"], df["%K"], df["%D"], df["BB_bw"], df["ATR"], df["OBV"]
    feats["ret1"] = df["ret1"]
    feats["vol20"] = df["vol20"]
    feats = feats.replace([np.inf,-np.inf], np.nan).dropna()
    return feats

def train_ensemble(X, y):
    lr = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
    xg = xgb.XGBClassifier(n_estimators=120,max_depth=4,use_label_encoder=False,eval_metric="logloss",n_jobs=-1)
    ensemble = VotingClassifier([("lr", lr), ("rf", rf), ("xg", xg)], voting="hard")
    ensemble.fit(X, y)
    return ensemble

def atomic_save_model(model, path):
    tmp = None
    lock = FileLock(MODEL_LOCKPATH, timeout=30)
    try:
        with lock:
            with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tf:
                dump(model, tf.name)
                tmp = tf.name
            os.replace(tmp, path)
    except Exception as e:
        logger.error("Failed to save model atomically: %s", e)
        if tmp and os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

def load_model_safe(path):
    if not os.path.exists(path):
        return None
    lock = FileLock(MODEL_LOCKPATH, timeout=30)
    try:
        with lock: return load(path)
    except Exception as e:
        logger.error("Failed to load model safely: %s", e)
        return None

def train_and_persist_model(df):
    feats = build_features(df)
    if len(feats)<100: logger.warning("Not enough data to train model"); return None
    y = (df["close"].shift(-1).loc[feats.index] > df["close"].loc[feats.index]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(feats, y, test_size=0.2, shuffle=False)
    model = train_ensemble(X_train, y_train)
    atomic_save_model(model, MODEL_PATH)
    return model

def run_ml_predict(df):
    feats = build_features(df)
    if len(feats)<30: return [], 0.5
    model = load_model_safe(MODEL_PATH)
    if model is None: model = train_and_persist_model(df)
    if model is None: return [], 0.5
    X_latest = feats.iloc[-1:].values
    pred = int(model.predict(X_latest)[0])
    conf = max(model.predict_proba(X_latest)[0]) if hasattr(model, "predict_proba") else 0.7
    return ["BUY"] if pred==1 else ["SELL"], conf

# ================ SIGNALS ================
def detect_candles(df: pd.DataFrame):
    if df.empty or len(df)<2: return []
    last = df.iloc[-1]; patt=[]
    if last.get("BullishEngulfing", False): patt.append("BUY")
    if last.get("BearishEngulfing", False): patt.append("SELL")
    if last.get("Doji", False): patt.append("HOLD")
    if last.get("Hammer", False): patt.append("BUY")
    return [p for p in patt if p in ("BUY","SELL")]

def combine_votes(votes_dict, ml_conf):
    votes_flat=[]; [votes_flat.extend(v) for v in votes_dict.values()]
    buy, sell=votes_flat.count("BUY"), votes_flat.count("SELL")
    if "ML" in votes_dict:
        if "BUY" in votes_dict["ML"]: buy += 2
        if "SELL" in votes_dict["ML"]: sell += 2
    if buy>sell: return "BUY", buy, sell
    if sell>buy: return "SELL", buy, sell
    return "HOLD", buy, sell

def indicator_votes(df: pd.DataFrame):
    if df.empty: return {}
    row = df.iloc[-1]; votes={}
    votes["EMA"] = ["BUY"] if row["EMA_fast"]>row["EMA_slow"] else ["SELL"]
    votes["RSI"] = ["BUY"] if row["RSI"]<30 else ["SELL"] if row["RSI"]>70 else []
    votes["Stoch"] = ["BUY"] if row["%K"]>row["%D"] else ["SELL"]
    votes["MACD"] = ["BUY"] if row["MACD"]>row["MACD_signal"] else ["SELL"]
    votes["Bollinger"] = ["BUY"] if row["close"]<row["BB_lower"] else ["SELL"] if row["close"]>row["BB_upper"] else []
    votes["ADX"] = ["BUY"] if row["ADX"]>25 and row["+DI"]>row["-DI"] else ["SELL"] if row["ADX"]>25 and row["+DI"]<row["-DI"] else []
    votes["VWAP"] = ["BUY"] if row["close"]>row["VWAP"] else ["SELL"] if row["close"]<row["VWAP"] else []
    votes["Candle"] = detect_candles(df)
    votes["ML"], ml_conf = run_ml_predict(df)
    votes["Harmonic"] = []
    harmonics = detect_harmonics(df)
    for p in harmonics:
        if "BUY" in p: votes["Harmonic"].extend(["BUY","BUY"])
        elif "SELL" in p: votes["Harmonic"].extend(["SELL","SELL"])
    return votes, ml_conf

def suggested_prices(df, vote):
    last = df["close"].iloc[-1]
    buy_price, sell_price = last * 0.99, last * 1.01
    levels = []
    for f in ["Fib_0.382", "Fib_0.5", "Fib_0.618"]:
        if f in df.columns:
            try: levels.append(float(df[f].iloc[-1]))
            except: pass
    harmonics = detect_harmonics(df)
    for h in harmonics:
        try: levels.append(float(h.split("@")[-1]))
        except: pass
    if levels:
        if vote=="BUY":
            below=[l for l in levels if l<last]; buy_price=max(below) if below else last*0.995
        elif vote=="SELL":
            above=[l for l in levels if l>last]; sell_price=min(above) if above else last*1.005
  return smart_round(buy_price), smart_round(sell_price)

def log_to_csv(symbol, tf, price, final, votes, buy, sell):
    flat=[]
    for k,v in votes.items(): flat.extend([f"{k}:{s}" for s in v])
    entry={"timestamp": now_str(), "symbol": symbol, "interval": tf, "price": round(float(price),6),
           "decision": final, "buy_votes": buy, "sell_votes": sell, "signals": ",".join(flat)}
    pd.DataFrame([entry]).to_csv(CSV_FILE, mode="a", index=False, header=not os.path.exists(CSV_FILE))

def format_message(symbol, tf, price, final, buy, sell, buy_p, sell_p, ml_conf):
    t = now_str()
    return (f"⏰ {t}\n📊 {symbol} | {tf}\n💰 Last Price: {round(float(price),2)} USDT\n\n"
            f"✅ FINAL DECISION: {final}\n🛒 Suggested Buy Price: {buy_p} USDT\n💵 Suggested Sell Price: {sell_p} USDT\n"
            f"💡 ML Confidence: {round(ml_conf*100,2)}%")

# ================ ANALYZE SYMBOL ================
async def analyze_symbol(symbol: str, tf: str):
    key = (symbol, tf)
    now = datetime.utcnow()

    # Проверка за cooldown
    if key in last_sent_time and now - last_sent_time[key] < timedelta(minutes=COOLDOWN_MINUTES):
        return

    df = await fetch_kucoin_candles(symbol, tf, MAX_OHLCV)
    if df.empty: 
        return
    if df["close"].iloc[-1] * df["volume"].iloc[-1] < MIN_VOLUME_USDT: 
        return

    df = add_indicators(df).dropna().reset_index(drop=True)
    votes, ml_conf = indicator_votes(df)
    final, buy, sell = combine_votes(votes, ml_conf)
    buy_p, sell_p = suggested_prices(df, final)
    last_price = df["close"].iloc[-1]

    # Проверка за минимална промена на цена
    if key in last_price_sent and abs(last_price - last_price_sent[key]) / max(last_price_sent[key], 1e-9) < PRICE_CHANGE_THRESHOLD:
        return
    last_price_sent[key] = last_price
    last_sent_time[key] = now

    # Логирање на CSV
    log_to_csv(symbol, tf, last_price, final, votes, buy, sell)

    # Формирање на пораката
    msg = format_message(symbol, tf, last_price, final, buy, sell, buy_p, sell_p, ml_conf)
    logger.info(f"DEBUG: Sending Telegram for {symbol} {tf} at price {last_price}")

    # Асинхроно испраќање порака
    asyncio.create_task(send_telegram(msg))

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
