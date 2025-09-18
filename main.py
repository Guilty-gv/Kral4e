# -*- coding: utf-8 -*-
"""
Advanced Crypto Trading Bot with Precision Analysis
- Multi-timeframe analysis with weighted signals
- Advanced indicators: Volume Profile, VWAP, Divergence, Harmonic Patterns
- Elliott Wave theory integration
- Machine Learning with Random Forest
- Precision price targets with confidence levels
- Sophisticated risk management
- NO TA-LIB DEPENDENCY - Uses pure Python alternatives
"""

import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kucoin.client import Market, Trade, User
from telegram import Bot
from sklearn.ensemble import RandomForestClassifier
import ta  # Pure Python alternative to TA-Lib

# ================= CONFIG =================
TOKENS = ["BTC","XRP","LINK","ONDO","AVAX","W","ACH","PEPE","PONKE","ICP",
          "FET","ALGO","HBAR","KAS","PYTH","IOTA","WAXL","ETH","ADA","PAXG","IMX","OP","FIL","PYTH","QNT"]
TIMEFRAMES = ["1h", "4h", "1d"]
TIMEFRAME_WEIGHTS = {"1h": 0.2, "4h": 0.3, "1d": 0.5}

MAX_OHLCV = 500
PRICE_ALERT_THRESHOLD = 0.03
COOLDOWN_MINUTES = 60
EMA_FAST, EMA_SLOW, RSI_PERIOD, ATR_PERIOD = 20, 50, 14, 14
VWAP_PERIOD = 20
STOCH_PERIOD = 14
BB_PERIOD = 20
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

CATEGORY_WEIGHTS = {
    "structure": 0.25,
    "momentum": 0.20,
    "volume": 0.15,
    "candles": 0.15,
    "exotic": 0.10,
    "harmonics": 0.15
}

# ================= KUCOIN CLIENTS =================
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

market_client = Market(
    key=KUCOIN_API_KEY, secret=KUCOIN_API_SECRET, passphrase=KUCOIN_API_PASSPHRASE
) if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE else Market()

trade_client = Trade(
    KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE
) if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE else None

user_client = User(
    KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE
) if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE else None

# ================= TELEGRAM =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

# ================= STATE =================
last_price_sent = {}
last_sent_time = {}
adaptive_weights = {sym: CATEGORY_WEIGHTS.copy() for sym in TOKENS}
signal_history = {sym: [] for sym in TOKENS}
ml_models = {sym: None for sym in TOKENS}
pattern_history = {sym: [] for sym in TOKENS}

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("advanced_crypto_bot")

# ================= UTILITIES =================
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
    if value >= 1:
        return round(value, 2)
    elif value >= 0.01:
        return round(value, 4)
    else:
        return round(value, 8)

# ================= CHECK CURRENT PRICE =================
def get_current_price(symbol: str) -> float:
    """Го враќа тековниот ticker price за даден симбол"""
    ticker = market_client.get_ticker(symbol)
    price = float(ticker['price'])
    return price

# ================= MAIN LOOP =================
async def continuous_monitor():
    logger.info("Starting Advanced Crypto Bot with Multi-Timeframe Analysis")
    
    # Train ML models at startup
    for sym in TOKENS:
        symbol = sym + "-USDT"
        df = await fetch_kucoin_candles(symbol, "1d", MAX_OHLCV)
        if not df.empty:
            train_ml_model(df, sym)
    
    # Main loop
    while True:
        for sym in TOKENS:
            symbol = sym + "-USDT"
            try:
                report = await enhanced_analyze_symbol(symbol)
                if report and report['signal']['direction'] != 'HOLD':
                    # Prepare message
                    signal = report['signal']
                    msg = (f"🔔 CRYPTO SIGNAL ALERT\n"
                           f"⏰ {now_str()}\n📊 {symbol}\n💰 Current Price: {report['current_price']}\n\n"
                           f"🎯 SIGNAL: {signal['direction']} ({signal['strength']}/10 strength)\n"
                           f"📈 Trend: {report['trend']}\n\n")
                    
                    if signal['buy_targets']:
                        msg += "🎯 BUY TARGETS:\n"
                        for i, (name, price, confidence) in enumerate(signal['buy_targets'], 1):
                            msg += f"{i}. {name} @ ${price} ({int(confidence*100)}% confidence)\n"
                    
                    if signal['sell_targets']:
                        msg += "\n🎯 SELL TARGETS:\n"
                        for i, (name, price, confidence) in enumerate(signal['sell_targets'], 1):
                            msg += f"{i}. {name} @ ${price} ({int(confidence*100)}% confidence)\n"
                    
                    # Add additional info
                    msg += f"\n📊 Additional Info:\n"
                    msg += f"- RSI: {report['current_price']}\n"
                    
                    if report['divergences']:
                        msg += f"- {len(report['divergences'])} divergence(s) detected\n"
                    
                    # Check if we should send alert
                    now = datetime.utcnow()
                    key = symbol
                    
                    change = abs(report['current_price'] - last_price_sent.get(key, report['current_price'])) / max(last_price_sent.get(key, report['current_price']), 1e-9)
                    if change < PRICE_ALERT_THRESHOLD and key in last_price_sent:
                        continue
                    if key in last_sent_time and now - last_sent_time[key] < timedelta(minutes=COOLDOWN_MINUTES):
                        continue
                    
                    logger.info(msg)
                    asyncio.create_task(send_telegram(msg))
                    
                    # Update state
                    last_price_sent[key] = report['current_price']
                    last_sent_time[key] = now
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        await asyncio.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    asyncio.run(continuous_monitor())
