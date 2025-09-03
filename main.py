# -*- coding: utf-8 -*-
"""
Crypto Swing Trading + XGBoost Analyzer + Telegram Notifier
Optimized for GitHub Actions
"""

import aiohttp, pandas as pd, ta, asyncio, numpy as np, os
from telegram import Bot
from datetime import datetime
import xgboost as xgb

# ================= CONFIG =================
BINANCE_PAIRS = [
    "BTCUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "ONDOUSDT",
    "ACHUSDT",
    "ALGOUSDT",
    "AVAXUSDT",
    "FETUSDT",
    "IOTAUSDT",
    "AXLUSDT",
    "HBARUSDT",
    "WUSDT"  # Wormhole (W)
]

COINGECKO_PAIRS = {"KASUSDT": "kas-network"}

TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# EMA комбинации по timeframe (прилагодено за swing trading)
EMA_MAP = {
    "15m": [9, 21],
    "1h": [20, 50],
    "4h": [50, 100],
    "1d": [50, 200]
}

RSI_PERIOD = 14
STOCH_PERIOD = 14
ATR_PERIOD = 14
PRICE_CHANGE_THRESHOLD = 0.05  # праќа само ако промена ≥ 5%
MAX_OHLCV = 200

BINANCE_URL = "https://api.binance.com/api/v3/klines"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

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
async def fetch_binance(symbol, interval="1h"):
    params = {"symbol": symbol, "interval": interval, "limit": MAX_OHLCV}
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.get(BINANCE_URL, params=params, timeout=30) as resp:
                    data = await resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.DataFrame(
                            data,
                            columns=[
                                "open_time","open","high","low","close","volume",
                                "close_time","quote_asset_volume","num_trades",
                                "taker_buy_base","taker_buy_quote","ignore"
                            ]
                        )
                        for c in ["open","high","low","close","volume"]:
                            df[c] = df[c].astype(float)
                        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
                        return df
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {symbol}: {e}")
                await asyncio.sleep(2)
    print(f"DEBUG: {symbol} {interval} - нема податоци")
    return pd.DataFrame()

async def fetch_coingecko(symbol_id, interval="hourly"):
    url = COINGECKO_URL.format(id=symbol_id)
    params = {"vs_currency":"usd","days":30,"interval":interval}
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    data = await resp.json()
                    prices = data.get("prices", [])
                    if len(prices) > 0:
                        df = pd.DataFrame(prices, columns=["timestamp","close"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df["open"] = df["close"]
                        df["high"] = df["close"]
                        df["low"] = df["close"]
                        df["volume"] = 0
                        return df
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {symbol_id}: {e}")
                await asyncio.sleep(2)
    print(f"DEBUG: {symbol_id} {interval} - нема податоци")
    return pd.DataFrame()

async def fetch_data(symbol, interval="1h"):
    if symbol in BINANCE_PAIRS: 
        return await fetch_binance(symbol, interval)
    elif symbol in COINGECKO_PAIRS: 
        return await fetch_coingecko(COINGECKO_PAIRS[symbol], interval)
    return pd.DataFrame()

# ================= INDICATORS =================
def add_indicators(df, tf):
    if df.empty: 
        return df
    for p in EMA_MAP[tf]:
        df[f"EMA{p}"] = df['close'].ewm(span=p, adjust=False).mean()
    try:
        df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min())/2
        df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min())/2
    except: 
        pass
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], RSI_PERIOD).rsi()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], STOCH_PERIOD)
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ATR_PERIOD).average_true_range()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    high = df['close'].max(); low = df['close'].min(); diff = high - low
    df['Fib_0.236'] = high - 0.236*diff
    df['Fib_0.382'] = high - 0.382*diff
    df['Fib_0.5'] = high - 0.5*diff
    df['Fib_0.618'] = high - 0.618*diff
    vol_ema = df['volume'].ewm(span=20, adjust=False).mean()
    df['VolumeSpike'] = df['volume'] >= 2*vol_ema
    return df

# ================= SIGNALS =================
def generate_signal(row, tf):
    if row.empty: return ["HOLD"]
    signals = []
    p = row['close']
    emas = EMA_MAP[tf]
    signals.append("BUY" if row[f"EMA{emas[0]}"] > row[f"EMA{emas[1]}"] else "SELL" if row[f"EMA{emas[0]}"] < row[f"EMA{emas[1]}"] else "")
    if 'Tenkan' in row and 'Kijun' in row:
        signals.append("BUY" if p > max(row['Tenkan'], row['Kijun']) else "SELL")
    signals.append("BUY" if row['RSI'] < 30 else "SELL" if row['RSI'] > 70 else "")
    signals.append("BUY" if row['%K'] > row['%D'] else "SELL")
    signals.append("BUY" if row['MACD'] > row['MACD_signal'] else "SELL")
    if p < row['BB_lower']: signals.append("BUY")
    elif p > row['BB_upper']: signals.append("SELL")
    for f in ['Fib_0.236','Fib_0.382','Fib_0.5','Fib_0.618']:
        if abs(p - row[f]) / p < 0.01: 
            signals.append("BUY" if p < row[f] else "SELL")
    if 'VolumeSpike' in row and row['VolumeSpike']: 
        signals.append("BUY")
    return [s for s in signals if s]

async def run_ml(df):
    signals = []
    try:
        feats = pd.DataFrame(index=df.index)
        feats["EMA20"] = df['close'].ewm(span=20, adjust=False).mean()
        feats["EMA50"] = df['close'].ewm(span=50, adjust=False).mean()
        feats["RSI"] = ta.momentum.RSIIndicator(df['close'], RSI_PERIOD).rsi()
        feats.dropna(inplace=True)
        if len(feats) >= 50:
            y = (df['close'].shift(-1).loc[feats.index] > df['close'].loc[feats.index]).astype(int)
            X = feats
            model = xgb.XGBClassifier(n_estimators=25, max_depth=3, use_label_encoder=False, eval_metric='logloss')
            model.fit(X, y)
            pred = int(model.predict(X.iloc[-1:].values)[0])
            signals.append("BUY" if pred == 1 else "SELL")
    except Exception as e:
        print(f"Error in run_ml: {e}")
    return signals

def combine_signals(indicator_signals, ml_signals):
    votes = indicator_signals.copy()
    votes.extend(ml_signals)
    buy = votes.count("BUY")
    sell = votes.count("SELL")
    if buy > sell: return "BUY"
    elif sell > buy: return "SELL"
    else: return "HOLD"

# ================= LOG =================
CSV_FILE = "crypto_signals_log.csv"
def log_to_csv(symbol, interval, price, final_signal, indicator_signals):
    entry = {
        "timestamp": now_str(),
        "symbol": symbol,
        "interval": interval,
        "price": round(float(price), 8),
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
        df = add_indicators(df, tf)
        indicator_signals = generate_signal(df.iloc[-1], tf)
        ml_signal = await run_ml(df)
        final_signal = combine_signals(indicator_signals, ml_signal)
        price = df['close'].iloc[-1]
        key = (symbol, tf)
        if key in last_price_sent and abs(price - last_price_sent[key]) / last_price_sent[key] < PRICE_CHANGE_THRESHOLD:
            continue
        last_price_sent[key] = price
        interval_msgs[tf] = final_signal
        log_to_csv(symbol, tf, price, final_signal, indicator_signals)
    if interval_msgs:
        msg_lines = [f"⏰ {now_str()}", f"📊 {symbol} Signals:"]
        for k,v in interval_msgs.items():
            msg_lines.append(f"{k} → {v}")
        msg = "\n".join(msg_lines)
        send_telegram(msg)

# ================= MAIN =================
async def main():
    tasks = [analyze_coin(sym) for sym in BINANCE_PAIRS + list(COINGECKO_PAIRS.keys())]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    print(f"{now_str()} ▶ Starting Crypto Signal Bot")
    asyncio.run(main())
