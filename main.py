# -*- coding: utf-8 -*-
"""
Long-Term Crypto Analyzer + ML + Telegram Notifier (KuCoin Only)
Optimized for 1d and 1week intervals
"""

import pandas as pd, ta, asyncio, numpy as np, os
from telegram import Bot
from datetime import datetime
import xgboost as xgb
from kucoin.client import Market

# ================= CONFIG =================
KUCOIN_PAIRS = [
    "BTCUSDT","XRPUSDT","LINKUSDT","ALGOUSDT","AVAXUSDT",
    "FETUSDT","IOTAUSDT","HBARUSDT","ACHUSDT","AXLUSDT","WUSDT","KASUSDT","ONDOUSDT","ADAUSDT","PEPEUSDT","PONKEUSDT"
]
TIMEFRAMES = ["1d","1week"]
PRICE_CHANGE_THRESHOLD = 0.01  # Испраќа само ако промена ≥ 1%
MAX_OHLCV = 200

market_client = Market()

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
def kucoin_interval_map(interval):
    return {"1d":"1day","1week":"1week"}.get(interval,"1day")

async def fetch_kucoin(symbol, interval="1d"):
    for attempt in range(3):
        try:
            kline_interval = kucoin_interval_map(interval)
            symbol_str = symbol.replace("USDT","-USDT")
            data = market_client.get_kline(symbol_str, kline_interval, limit=MAX_OHLCV)
            if data:
                df = pd.DataFrame(data, columns=["timestamp","open","close","high","low","volume","turnover"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                for c in ["open","close","high","low","volume","turnover"]:
                    df[c] = df[c].astype(float)
                return df
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {symbol}: {e}")
            await asyncio.sleep(2)
    print(f"DEBUG: {symbol} {interval} - нема податоци")
    return pd.DataFrame()

async def fetch_data(symbol, interval="1d"):
    return await fetch_kucoin(symbol, interval)

# ================= PATTERNS =================
def detect_butterfly(df):
    if len(df)<5: return False
    recent=df[-100:]
    X,A,B,C,D=recent['high'].iloc[0],recent['low'].iloc[1],recent['high'].iloc[2],recent['low'].iloc[3],recent['high'].iloc[4]
    XA,AB,BC,CD=abs(A-X),abs(B-A),abs(C-B),abs(D-C)
    if 0.786*XA<=AB<=0.886*XA and 0.382*AB<=BC<=0.886*AB and 1.618*BC<=CD<=2.618*BC:
        return True
    return False

def detect_engulfing(df):
    if len(df)<2: return None
    last, prev=df.iloc[-1], df.iloc[-2]
    if last['close']>last['open'] and prev['close']<prev['open'] and last['open']<prev['close'] and last['close']>prev['open']:
        return "Bullish Engulfing"
    if last['close']<last['open'] and prev['close']>prev['open'] and last['open']>prev['close'] and last['close']<prev['open']:
        return "Bearish Engulfing"
    return None

def detect_hammer_doji(df):
    last=df.iloc[-1]
    body=abs(last['close']-last['open'])
    candle_range=last['high']-last['low']
    upper_shadow=last['high']-max(last['close'],last['open'])
    lower_shadow=min(last['close'],last['open'])-last['low']
    if body<=0.3*candle_range and lower_shadow>=2*body:
        return "Hammer"
    if body<=0.1*candle_range and upper_shadow>=0 and lower_shadow>=0:
        return "Doji"
    return None

def add_patterns(df):
    df['Butterfly']=detect_butterfly(df)
    df['Engulfing']=detect_engulfing(df)
    df['Candle']=detect_hammer_doji(df)
    return df

# ================= INDICATORS =================
def add_indicators(df):
    if df.empty: return df
    for p in [50,200]: df[f"EMA{p}"]=df['close'].ewm(span=p, adjust=False).mean()
    df = add_patterns(df)
    return df

# ================= SIGNALS =================
def generate_signal(row):
    if row.empty: return ["HOLD"]
    signals=[]
    if 'EMA50' in row and 'EMA200' in row:
        signals.append("BUY" if row['EMA50']>row['EMA200'] else "SELL")
    if row.get('Butterfly'): signals.append("BUY")
    if row.get('Engulfing')=="Bullish Engulfing": signals.append("BUY")
    elif row.get('Engulfing')=="Bearish Engulfing": signals.append("SELL")
    if row.get('Candle')=="Hammer": signals.append("BUY")
    elif row.get('Candle')=="Doji": signals.append("HOLD")
    return [s for s in signals if s]

async def run_ml(df):
    signals=[]
    try:
        feats=pd.DataFrame(index=df.index)
        for p in [50,200]: feats[f"EMA{p}"]=df['close'].ewm(span=p,adjust=False).mean()
        feats.dropna(inplace=True)
        if len(feats)>=50:
            y=(df['close'].shift(-1).loc[feats.index]>df['close'].loc[feats.index]).astype(int)
            X=feats
            model=xgb.XGBClassifier(n_estimators=25,max_depth=3,use_label_encoder=False,eval_metric='logloss')
            model.fit(X,y)
            pred=int(model.predict(X.iloc[-1:].values)[0])
            signals.append("BUY" if pred==1 else "SELL")
    except Exception as e:
        print(f"Error in run_ml: {e}")
    return signals

def combine_signals(indicator_signals, ml_signals):
    votes=indicator_signals.copy()
    votes.extend(ml_signals)
    buy=votes.count("BUY"); sell=votes.count("SELL")
    if buy>sell: return "BUY"
    elif sell>buy: return "SELL"
    else: return "HOLD"

# ================= LOG =================
CSV_FILE="crypto_signals_log.csv"
def log_to_csv(symbol, interval, price, final_signal, indicator_signals):
    entry={"timestamp": now_str(), "symbol": symbol, "interval": interval, "price": round(float(price),8),
           "signal": final_signal, "indicator_signals": ",".join(indicator_signals)}
    df_row=pd.DataFrame([entry])
    df_row.to_csv(CSV_FILE, mode="a", index=False, header=not os.path.exists(CSV_FILE))

# ================= ANALYSIS =================
async def analyze_coin(symbol):
    interval_msgs={}
    global last_price_sent
    for tf in TIMEFRAMES:
        df = await fetch_data(symbol, tf)
        if df.empty: continue
        df = add_indicators(df)
        indicator_signals = generate_signal(df.iloc[-1])
        ml_signal = await run_ml(df)
        final_signal = combine_signals(indicator_signals, ml_signal)
        price=df['close'].iloc[-1]
        key=(symbol,tf)
        if key in last_price_sent and abs(price-last_price_sent[key])/last_price_sent[key]<PRICE_CHANGE_THRESHOLD:
            continue
        last_price_sent[key]=price
        interval_msgs[tf]=final_signal
        log_to_csv(symbol, tf, price, final_signal, indicator_signals)
    if interval_msgs:
        msg_lines=[f"⏰ {now_str()}", f"📊 {symbol} Analysis"]
        for k,v in interval_msgs.items():
            msg_lines.append(f"Interval: {k}\n💰 Price: {price} USDT\n✅ Final Decision: {v}")
        msg="\n\n".join(msg_lines)
        send_telegram(msg)

# ================= MAIN =================
async def main():
    tasks=[analyze_coin(sym) for sym in KUCOIN_PAIRS]
    await asyncio.gather(*tasks)

if __name__=="__main__":
    print(f"{now_str()} ▶ Starting Long-Term Crypto Bot (KuCoin Only)")
    asyncio.run(main())
