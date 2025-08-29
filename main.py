# -*- coding: utf-8 -*-
"""
Crypto Swing Trading + XGBoost Analyzer + Telegram Notifier
Optimized for Replit free plan
Includes daily and weekly Telegram reports with trend graphs
"""

import aiohttp, pandas as pd, ta, asyncio, numpy as np, os, matplotlib.pyplot as plt
from telegram import Bot
from datetime import datetime, timedelta
import xgboost as xgb

# ================= CONFIG =================
BINANCE_PAIRS = ["BTCUSDT","XRPUSDT","LINKUSDT","ONDOUSDT"]
COINGECKO_PAIRS = {"KASUSDT":"kas-network"}
TIMEFRAMES = ["1d","4h","1h","15m"]
EMA_PERIODS = [20,50]
RSI_PERIOD = 14
STOCH_PERIOD = 14
ATR_PERIOD = 14
PRICE_CHANGE_THRESHOLD = 0.05
MAX_OHLCV = 200

BINANCE_URL = "https://api.binance.com/api/v3/klines"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

TELEGRAM_TOKEN = "TVOJ_TOKEN"
CHAT_ID = "TVOJ_ID"
bot = Bot(token=TELEGRAM_TOKEN)

CSV_FILE = "crypto_signals_log.csv"
last_price_sent = {}
last_daily_report = None
last_weekly_report = None

# ================= HELPERS =================
def now_str(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def fetch_binance(symbol, interval="1h"):
    try:
        params = {"symbol":symbol,"interval":interval,"limit":MAX_OHLCV}
        async with aiohttp.ClientSession() as session:
            async with session.get(BINANCE_URL, params=params, timeout=20) as resp:
                data = await resp.json()
                if not isinstance(data,list) or len(data)==0: return pd.DataFrame()
                df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume",
                                                 "close_time","quote_asset_volume","num_trades",
                                                 "taker_buy_base","taker_buy_quote","ignore"])
                for c in ["open","high","low","close","volume"]: df[c]=df[c].astype(float)
                df["open_time"]=pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"]=pd.to_datetime(df["close_time"], unit="ms")
                return df
    except: return pd.DataFrame()

async def fetch_coingecko(symbol_id, interval="hourly"):
    try:
        url = COINGECKO_URL.format(id=symbol_id)
        params = {"vs_currency":"usd","days":30,"interval":interval}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=20) as resp:
                data = await resp.json()
                prices = data.get("prices", [])
                if len(prices)==0: return pd.DataFrame()
                df = pd.DataFrame(prices, columns=["timestamp","close"])
                df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms")
                df["open"]=df["close"]; df["high"]=df["close"]; df["low"]=df["close"]; df["volume"]=0
                return df
    except: return pd.DataFrame()

async def fetch_data(symbol, interval="1h"):
    if symbol in BINANCE_PAIRS: return await fetch_binance(symbol, interval)
    elif symbol in COINGECKO_PAIRS: return await fetch_coingecko(COINGECKO_PAIRS[symbol], interval)
    return pd.DataFrame()

# ================= INDICATORS =================
def add_indicators(df):
    if df.empty: return df
    for p in EMA_PERIODS: df[f"EMA{p}"] = df['close'].ewm(span=p, adjust=False).mean()
    try:
        df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min())/2
        df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min())/2
    except: pass
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], RSI_PERIOD).rsi()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], STOCH_PERIOD)
    df['%K'] = stoch.stoch(); df['%D'] = stoch.stoch_signal()
    macd = ta.trend.MACD(df['close']); df['MACD'] = macd.macd(); df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close']); df['BB_upper'] = bb.bollinger_hband(); df['BB_lower'] = bb.bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ATR_PERIOD).average_true_range()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    high = df['close'].max(); low = df['close'].min(); diff = high-low
    df['Fib_0.236']=high-0.236*diff; df['Fib_0.382']=high-0.382*diff; df['Fib_0.5']=high-0.5*diff; df['Fib_0.618']=high-0.618*diff
    vol_ema = df['volume'].ewm(span=20,adjust=False).mean()
    df['VolumeSpike'] = df['volume'] >= 2*vol_ema
    return df

# ================= SIGNAL =================
def generate_signal(row):
    if row.empty: return ["HOLD"]
    signals = []
    p = row['close']

    signals.append("BUY" if row['EMA20'] > row['EMA50'] else "SELL" if row['EMA20'] < row['EMA50'] else "")
    
    if 'Tenkan' in row and 'Kijun' in row:
        signals.append("BUY" if p > max(row['Tenkan'], row['Kijun']) else "SELL")
    
    signals.append("BUY" if row['RSI'] < 30 else "SELL" if row['RSI'] > 70 else "")
    signals.append("BUY" if row['%K'] > row['%D'] else "SELL")
    signals.append("BUY" if row['MACD'] > row['MACD_signal'] else "SELL")
    
    if p < row['BB_lower']: signals.append("BUY")
    elif p > row['BB_upper']: signals.append("SELL")
    
    for f in ['Fib_0.236','Fib_0.382','Fib_0.5','Fib_0.618']:
        if abs(p - row[f]) / p < 0.01: signals.append("BUY" if p < row[f] else "SELL")
    
    if 'VolumeSpike' in row and row['VolumeSpike']: signals.append("BUY")

    return [s for s in signals if s]  # отстрани празни редови


# ================= XGBoost ML =================
async def run_ml(df):
    signals=[]
    try:
        feats=pd.DataFrame(index=df.index)
        for p in EMA_PERIODS: feats[f"EMA{p}"]=df['close'].ewm(span=p,adjust=False).mean()
        feats["RSI"]=ta.momentum.RSIIndicator(df['close'],RSI_PERIOD).rsi()
        feats.dropna(inplace=True)
        if len(feats)>=50:
            y=(df['close'].shift(-1).loc[feats.index]>df['close'].loc[feats.index]).astype(int)
            X=feats
            model=xgb.XGBClassifier(n_estimators=25,max_depth=3,use_label_encoder=False,eval_metric='logloss')
            model.fit(X,y)
            pred=int(model.predict(X.iloc[-1:].values)[0])
            signals.append("BUY" if pred==1 else "SELL")
    except: pass
    return signals

# ================= MAJORITY VOTE =================
def combine_signals(indicator_signals, ml_signals):
    votes = indicator_signals.copy(); votes.extend(ml_signals)
    buy = votes.count("BUY"); sell = votes.count("SELL")
    if buy>sell: return "BUY"
    elif sell>buy: return "SELL"
    else: return "HOLD"

# ================= LOG TO CSV =================
def log_to_csv(symbol, interval, price, final_signal, indicator_signals):
    entry = {
        "timestamp": now_str(),
        "symbol": symbol,
        "interval": interval,
        "price": round(float(price),8),
        "signal": final_signal,
        "indicator_signals": ",".join(indicator_signals)
    }
    df_row = pd.DataFrame([entry])
    df_row.to_csv(CSV_FILE, mode="a", index=False, header=not os.path.exists(CSV_FILE))

# ================= DAILY REPORT =================
async def daily_report():
    global last_daily_report
    today = datetime.now().date()
    if last_daily_report == today: return
    if not os.path.exists(CSV_FILE): return

    df = pd.read_csv(CSV_FILE)
    df['timestamp']=pd.to_datetime(df['timestamp'])
    df['date']=df['timestamp'].dt.date
    df_today = df[df['date']==today]

    if df_today.empty: return

    summary = df_today.groupby('symbol')['signal'].value_counts().unstack(fill_value=0)
    msg_lines = [f"📅 Дневен извештај за {today}"]
    for sym in summary.index:
        counts = summary.loc[sym]
        msg_lines.append(f"{sym}: BUY={counts.get('BUY',0)} SELL={counts.get('SELL',0)} HOLD={counts.get('HOLD',0)}")
    msg = "\n".join(msg_lines)
    await asyncio.get_running_loop().run_in_executor(None, bot.send_message, CHAT_ID, msg)
    last_daily_report = today

# ================= WEEKLY REPORT =================
async def weekly_report():
    global last_weekly_report
    today = datetime.now().date()
    week_start = today - timedelta(days=today.weekday())  # Monday
    if last_weekly_report == week_start: return
    if not os.path.exists(CSV_FILE): return

    df = pd.read_csv(CSV_FILE)
    df['timestamp']=pd.to_datetime(df['timestamp'])
    df['date']=df['timestamp'].dt.date
    df_week = df[df['date']>=week_start]
    if df_week.empty: return

    summary = df_week.groupby(['date','symbol'])['signal'].value_counts().unstack(fill_value=0)
    # Plot graph
    for sym in summary.index.get_level_values(1).unique():
        sym_df = summary.xs(sym, level=1)
        sym_df.plot(kind='line', title=f"Weekly Signal Trend: {sym}", figsize=(8,4))
        plt.ylabel("Count")
        plt.savefig(f"{sym}_weekly.png")
        plt.close()
        # send image
        await asyncio.get_running_loop().run_in_executor(None, bot.send_photo, CHAT_ID, open(f"{sym}_weekly.png","rb"))
    last_weekly_report = week_start

# ================= ANALYSIS =================
async def analyze_coin(symbol):
    interval_msgs = {}
    global last_price_sent
    for tf in TIMEFRAMES:
        df = await fetch_data(symbol, tf)
        if df.empty: continue
        df = add_indicators(df)

        indicator_signals = generate_signal(df.iloc[-1])
        ml_signal = await run_ml(df)
        final_signal = combine_signals(indicator_signals, ml_signal)

        price = df['close'].iloc[-1]
        key = (symbol, tf)
        if key in last_price_sent and abs(price-last_price_sent[key])/last_price_sent[key]<PRICE_CHANGE_THRESHOLD:
            continue
        last_price_sent[key]=price
        interval_msgs[tf] = final_signal

        # Log to CSV
        log_to_csv(symbol, tf, price, final_signal, indicator_signals)

    if interval_msgs:
        msg_lines=[f"⏰ {now_str()}", f"📊 {symbol} Signals:"]
        for k,v in interval_msgs.items(): msg_lines.append(f"{k} → {v}")
        msg = "\n".join(msg_lines)
        await asyncio.get_running_loop().run_in_executor(None, bot.send_message, CHAT_ID, msg)

# ================= MAIN LOOP =================
async def main_loop():
    while True:
        tasks=[analyze_coin(sym) for sym in BINANCE_PAIRS+list(COINGECKO_PAIRS.keys())]
        await asyncio.gather(*tasks)
        # daily and weekly report
        await daily_report()
        await weekly_report()
        await asyncio.sleep(60*5)

if __name__=="__main__":
    print(f"{now_str()} ▶ Starting Crypto Signal Bot with Indicators + XGBoost + CSV Logging + Daily/Weekly Report")
    asyncio.run(main_loop())
