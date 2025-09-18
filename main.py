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
from kucoin.client import Market
from telegram import Bot
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

market_client = Market(
    key=KUCOIN_API_KEY, secret=KUCOIN_API_SECRET, passphrase=KUCOIN_API_PASSPHRASE
) if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE else Market()

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

# ================= FETCH CANDLES =================
async def fetch_kucoin_candles(symbol: str, tf: str = "1d", limit: int = 200):
    interval_map = {"1d": "1day", "4h": "4hour", "1h": "1hour", "1w": "1week", "15m": "15min"}
    interval = interval_map.get(tf, "1day")
    loop = asyncio.get_running_loop()
    try:
        candles = await loop.run_in_executor(
            None, lambda: market_client.get_kline(symbol, interval, limit=limit)
        )
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        return df.sort_values("timestamp").reset_index(drop=True)[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error("Error fetching %s: %s", symbol, e)
        return pd.DataFrame()

# ================= INDICATORS (NO TA-LIB) =================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    
    # EMA indicators
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA_100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    # SMA indicators
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["SMA_200"] = df["close"].rolling(200).mean()
    
    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=RSI_PERIOD).rsi()
    
    # OBV
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    
    # ATR
    df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_PERIOD).average_true_range()
    
    # MACD
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=STOCH_PERIOD)
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=BB_PERIOD)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    
    # ADX
    df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    
    # Supertrend indicator
    df = add_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    
    # VWAP
    df["VWAP"] = (df["close"] * df["volume"]).rolling(VWAP_PERIOD).sum() / (df["volume"].rolling(VWAP_PERIOD).sum() + 1e-9)
    
    # Advanced VWAP with deviations
    df = calculate_advanced_vwap(df)
    
    return df

def add_supertrend(df, period=10, multiplier=3):
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate ATR using ta library
    atr = ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # First value
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(df)):
        if close.iloc[i] > supertrend.iloc[i-1]:
            direction.iloc[i] = 1
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
        else:
            direction.iloc[i] = -1
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
    
    df['Supertrend'] = supertrend
    df['Supertrend_Direction'] = direction
    
    return df

# ================= VOLUME PROFILE & VWAP =================

def calculate_volume_profile(df, num_bins=20):
    """Пресметува Volume Profile (POC, Value Area)"""
    # Сортирај ги цените во bins
    price_range = df['high'].max() - df['low'].min()
    bin_size = price_range / num_bins
    bins = [df['low'].min() + i * bin_size for i in range(num_bins + 1)]
    
    # Пресметај волумен за секој bin
    volume_per_bin = {}
    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i + 1]
        mask = (df['close'] >= lower) & (df['close'] < upper)
        volume_per_bin[f"{lower:.2f}-{upper:.2f}"] = df.loc[mask, 'volume'].sum()
    
    # Најди Point of Control (POC)
    poc_bin = max(volume_per_bin, key=volume_per_bin.get)
    poc_value = (float(poc_bin.split('-')[0]) + float(poc_bin.split('-')[1])) / 2
    
    # Најди Value Area (70% од волуменот)
    total_volume = sum(volume_per_bin.values())
    sorted_bins = sorted(volume_per_bin.items(), key=lambda x: x[1], reverse=True)
    
    value_area_volume = 0
    value_area_bins = []
    for bin_name, volume in sorted_bins:
        if value_area_volume < total_volume * 0.7:
            value_area_volume += volume
            value_area_bins.append(bin_name)
    
    # Најди ги границите на Value Area
    value_area_prices = []
    for bin_name in value_area_bins:
        value_area_prices.extend([float(bin_name.split('-')[0]), float(bin_name.split('-')[1])])
    
    value_area_high = max(value_area_prices)
    value_area_low = min(value_area_prices)
    
    return {
        'poc': poc_value,
        'value_area_high': value_area_high,
        'value_area_low': value_area_low,
        'volume_profile': volume_per_bin
    }

def calculate_advanced_vwap(df):
    """Пресметува напреден VWAP со отстапувања"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Пресметај стандардно отстапување
    std_dev = (typical_price - vwap).rolling(20).std()
    
    # VWAP линии со отстапувања
    df['VWAP'] = vwap
    df['VWAP_upper_1'] = vwap + std_dev
    df['VWAP_upper_2'] = vwap + 2 * std_dev
    df['VWAP_lower_1'] = vwap - std_dev
    df['VWAP_lower_2'] = vwap - 2 * std_dev
    
    return df

# ================= HARMONIC PATTERNS =================
def find_swing_points(df, lookback=5):
    highs = df['high'].values
    lows = df['low'].values
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df)-lookback):
        # Find swing highs
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_highs.append((i, highs[i], df['timestamp'].iloc[i]))
        # Find swing lows
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_lows.append((i, lows[i], df['timestamp'].iloc[i]))
    
    return swing_highs, swing_lows

def check_harmonic_patterns(swing_highs, swing_lows):
    patterns = []
    
    # We need at least 5 points for most harmonic patterns (X, A, B, C, D)
    if len(swing_highs) < 5 or len(swing_lows) < 5:
        return patterns
    
    # Combine all swing points in chronological order
    all_swings = []
    for high in swing_highs:
        all_swings.append(('high', high[0], high[1], high[2]))
    for low in swing_lows:
        all_swings.append(('low', low[0], low[1], low[2]))
    
    all_swings.sort(key=lambda x: x[1])  # Sort by index
    
    # Check for various harmonic patterns
    for i in range(len(all_swings) - 4):
        points = all_swings[i:i+5]
        
        # Check if we have alternating high/low points
        if not (points[0][0] != points[1][0] and points[1][0] != points[2][0] and 
                points[2][0] != points[3][0] and points[3][0] != points[4][0]):
            continue
        
        # Extract price values
        X = points[0][2]
        A = points[1][2]
        B = points[2][2]
        C = points[3][2]
        D = points[4][2]
        
        # Calculate ratios
        AB = abs(B - A)
        BC = abs(C - B)
        CD = abs(D - C)
        XA = abs(A - X)
        
        AB_ratio = AB / XA if XA != 0 else 0
        BC_ratio = BC / AB if AB != 0 else 0
        CD_ratio = CD / BC if BC != 0 else 0
        XA_ratio = XA / abs(X) if X != 0 else 0
        
        # Check for Gartley pattern
        if (0.55 <= AB_ratio <= 0.65 and 0.35 <= BC_ratio <= 0.45 and 
            1.0 <= CD_ratio <= 1.2 and 0.75 <= (abs(D - X) / XA) <= 0.85):
            pattern_type = "Bullish Gartley" if D < X else "Bearish Gartley"
            patterns.append((pattern_type, D, points[4][3]))
        
        # Check for Bat pattern
        elif (0.35 <= AB_ratio <= 0.50 and 0.35 <= BC_ratio <= 0.45 and 
              1.6 <= CD_ratio <= 1.8 and 0.85 <= (abs(D - X) / XA) <= 0.90):
            pattern_type = "Bullish Bat" if D < X else "Bearish Bat"
            patterns.append((pattern_type, D, points[4][3]))
        
        # Check for Butterfly pattern
        elif (0.70 <= AB_ratio <= 0.80 and 0.35 <= BC_ratio <= 0.45 and 
              1.2 <= CD_ratio <= 1.4 and 1.25 <= (abs(D - X) / XA) <= 1.30):
            pattern_type = "Bullish Butterfly" if D < X else "Bearish Butterfly"
            patterns.append((pattern_type, D, points[4][3]))
    
    return patterns

# ================= FIBONACCI & ELLIOTT WAVE =================
def fib_levels(df, lookback=100):
    recent = df.tail(lookback)
    high, low = recent["high"].max(), recent["low"].min()
    diff = high - low
    return {
        "0.0": high,
        "0.236": high - diff * 0.236,
        "0.382": high - diff * 0.382,
        "0.5": high - diff * 0.5,
        "0.618": high - diff * 0.618,
        "0.786": high - diff * 0.786,
        "1.0": low,
        "1.272": low - diff * 0.272,
        "1.414": low - diff * 0.414,
        "1.618": low - diff * 0.618
    }

def detect_elliott_waves(swing_highs, swing_lows):
    waves = []
    
    # Simple Elliott Wave detection based on Fibonacci ratios
    for i in range(len(swing_highs) - 5):
        # Check for impulse wave (5 waves)
        wave1 = abs(swing_highs[i+1][1] - swing_lows[i][1])
        wave2 = abs(swing_lows[i+1][1] - swing_highs[i+1][1])
        wave3 = abs(swing_highs[i+2][1] - swing_lows[i+1][1])
        wave4 = abs(swing_lows[i+2][1] - swing_highs[i+2][1])
        wave5 = abs(swing_highs[i+3][1] - swing_lows[i+2][1])
        
        # Check Fibonacci relationships between waves
        if (0.38 <= wave2/wave1 <= 0.62 and 1.618 <= wave3/wave1 <= 2.618 and
            0.38 <= wave4/wave3 <= 0.50 and 0.62 <= wave5/wave1 <= 1.00):
            waves.append(("Impulse Wave", swing_highs[i+3][1], swing_highs[i+3][2]))
    
    return waves

# ================= DIVERGENCE DETECTION =================

def detect_all_divergences(df, lookback=14):
    """Детектира различен тип на дивергенции"""
    divergences = []
    
    # RSI дивергенција
    rsi = df['RSI'].values
    price = df['close'].values
    
    # Најди peaks и troughs за цена и RSI
    price_peaks = [i for i in range(lookback, len(df)-lookback) 
                  if price[i] == max(price[i-lookback:i+lookback+1])]
    price_troughs = [i for i in range(lookback, len(df)-lookback) 
                    if price[i] == min(price[i-lookback:i+lookback+1])]
    
    rsi_peaks = [i for i in range(lookback, len(df)-lookback) 
                if rsi[i] == max(rsi[i-lookback:i+lookback+1])]
    rsi_troughs = [i for i in range(lookback, len(df)-lookback) 
                  if rsi[i] == min(rsi[i-lookback:i+lookback+1])]
    
    # Regular bearish дивергенција (цената прави повисоки врвови, RSI пониски)
    for i in range(1, min(len(price_peaks), len(rsi_peaks))):
        if (price[price_peaks[i]] > price[price_peaks[i-1]] and
            rsi[rsi_peaks[i]] < rsi[rsi_peaks[i-1]]):
            divergences.append(('BEARISH_DIVERGENCE', price_peaks[i], 'RSI'))
    
    # Regular bullish дивергенција (цената прави пониски дна, RSI повисоки)
    for i in range(1, min(len(price_troughs), len(rsi_troughs))):
        if (price[price_troughs[i]] < price[price_troughs[i-1]] and
            rsi[rsi_troughs[i]] > rsi[rsi_troughs[i-1]]):
            divergences.append(('BULLISH_DIVERGENCE', price_troughs[i], 'RSI'))
    
    # MACD дивергенција (иста логика како RSI)
    if 'MACD' in df.columns:
        macd = df['MACD'].values
        macd_peaks = [i for i in range(lookback, len(df)-lookback) 
                     if macd[i] == max(macd[i-lookback:i+lookback+1])]
        macd_troughs = [i for i in range(lookback, len(df)-lookback) 
                       if macd[i] == min(macd[i-lookback:i+lookback+1])]
        
        # Додади MACD дивергенција detection
        for i in range(1, min(len(price_peaks), len(macd_peaks))):
            if (price[price_peaks[i]] > price[price_peaks[i-1]] and
                macd[macd_peaks[i]] < macd[macd_peaks[i-1]]):
                divergences.append(('BEARISH_DIVERGENCE', price_peaks[i], 'MACD'))
        
        for i in range(1, min(len(price_troughs), len(macd_troughs))):
            if (price[price_troughs[i]] < price[price_troughs[i-1]] and
                macd[macd_troughs[i]] > macd[macd_troughs[i-1]]):
                divergences.append(('BULLISH_DIVERGENCE', price_troughs[i], 'MACD'))
    
    return divergences

# ================= CANDLE PATTERNS =================
def detect_candle_patterns(df):
    patterns = []
    open_price = df['open'].iloc[-1]
    close_price = df['close'].iloc[-1]
    high_price = df['high'].iloc[-1]
    low_price = df['low'].iloc[-1]
    prev_open = df['open'].iloc[-2] if len(df) > 1 else open_price
    prev_close = df['close'].iloc[-2] if len(df) > 1 else close_price
    prev_high = df['high'].iloc[-2] if len(df) > 1 else high_price
    prev_low = df['low'].iloc[-2] if len(df) > 1 else low_price
    
    # Calculate candle properties
    body = abs(close_price - open_price)
    total_range = high_price - low_price
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    
    # Doji pattern
    if body < total_range * 0.1:
        patterns.append("DOJI")
    
    # Hammer pattern
    if lower_wick > body * 2 and upper_wick < body * 0.1 and close_price > open_price:
        patterns.append("HAMMER")
    
    # Shooting Star pattern
    if upper_wick > body * 2 and lower_wick < body * 0.1 and close_price < open_price:
        patterns.append("SHOOTING_STAR")
    
    # Engulfing pattern
    if (body > abs(prev_close - prev_open) * 1.5 and
        ((close_price > open_price and prev_close < prev_open and close_price > prev_open and open_price < prev_close) or
         (close_price < open_price and prev_close > prev_open and close_price < prev_open and open_price > prev_close))):
        patterns.append("BULLISH_ENGULFING" if close_price > open_price else "BEARISH_ENGULFING")
    
    return patterns

# ================= ENHANCED PRICE TARGETS =================

def calculate_precise_price_targets(df, current_price):
    """Пресметува прецизни buy/sell цели комбинирајќи повеќе методи"""
    targets = {
        'buy_targets': [],
        'sell_targets': [],
        'confidence': 0
    }
    
    # 1. Fibonacci нивоа
    fib_levels_dict = fib_levels(df)
    for level, price in fib_levels_dict.items():
        if price < current_price:
            targets['buy_targets'].append(('Fibonacci ' + level, price, 0.7))
        else:
            targets['sell_targets'].append(('Fibonacci ' + level, price, 0.7))
    
    # 2. Volume Profile POC и Value Area
    volume_profile = calculate_volume_profile(df)
    targets['buy_targets'].append(('Volume POC', volume_profile['poc'], 0.8))
    targets['buy_targets'].append(('Value Area Low', volume_profile['value_area_low'], 0.6))
    targets['sell_targets'].append(('Value Area High', volume_profile['value_area_high'], 0.6))
    
    # 3. Хармонични патерни
    swing_highs, swing_lows = find_swing_points(df)
    harmonic_patterns = check_harmonic_patterns(swing_highs, swing_lows)
    
    for pattern, price, timestamp in harmonic_patterns:
        if "Bullish" in pattern and price < current_price * 1.05:  # Within 5%
            targets['buy_targets'].append((pattern, price, 0.9))
        elif "Bearish" in pattern and price > current_price * 0.95:  # Within 5%
            targets['sell_targets'].append((pattern, price, 0.9))
    
    # 4. VWAP нивоа
    df_with_vwap = calculate_advanced_vwap(df)
    vwap = df_with_vwap['VWAP'].iloc[-1]
    vwap_upper = df_with_vwap['VWAP_upper_1'].iloc[-1]
    vwap_lower = df_with_vwap['VWAP_lower_1'].iloc[-1]
    
    targets['buy_targets'].append(('VWAP Support', vwap_lower, 0.6))
    targets['buy_targets'].append(('VWAP', vwap, 0.5))
    targets['sell_targets'].append(('VWAP Resistance', vwap_upper, 0.6))
    
    # 5. Подредување на целите според confidence и растојание од тековната цена
    def sort_and_filter(target_list, current_price, is_buy=True):
        # Филтрирај цели кои се соодветно позиционирани
        if is_buy:
            filtered = [t for t in target_list if t[1] < current_price * 0.98]  # Барем 2% подолу
        else:
            filtered = [t for t in target_list if t[1] > current_price * 1.02]  # Барем 2% погоре
        
        # Сортирај по confidence (3ти елемент)
        filtered.sort(key=lambda x: x[2], reverse=True)
        
        # Групирај слични цели (во ренџ од 1%)
        grouped = []
        for target in filtered:
            name, price, confidence = target
            found_group = False
            for i, group in enumerate(grouped):
                if abs(price - group[1]) / group[1] < 0.01:  # Within 1%
                    # Агрегирај ги целите
                    grouped[i] = (
                        f"{group[0]}+{name}",
                        (group[1] + price) / 2,  # Просечна цена
                        max(group[2], confidence)  # Највисок confidence
                    )
                    found_group = True
                    break
            
            if not found_group:
                grouped.append(target)
        
        return grouped[:3]  # Врати ги топ 3 цели
    
    targets['buy_targets'] = sort_and_filter(targets['buy_targets'], current_price, True)
    targets['sell_targets'] = sort_and_filter(targets['sell_targets'], current_price, False)
    
    # Пресметај го overall confidence
    if targets['buy_targets'] and targets['sell_targets']:
        buy_conf = sum(t[2] for t in targets['buy_targets']) / len(targets['buy_targets'])
        sell_conf = sum(t[2] for t in targets['sell_targets']) / len(targets['sell_targets'])
        targets['confidence'] = (buy_conf + sell_conf) / 2
    
    return targets

# ================= TREND ANALYSIS =================
def is_bullish_trend(df):
    return (df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1] and 
            df['close'].iloc[-1] > df['EMA_50'].iloc[-1] and
            df['Supertrend_Direction'].iloc[-1] == 1)

def is_bearish_trend(df):
    return (df['EMA_50'].iloc[-1] < df['EMA_200'].iloc[-1] and 
            df['close'].iloc[-1] < df['EMA_50'].iloc[-1] and
            df['Supertrend_Direction'].iloc[-1] == -1)

# ================= WEIGHTED VOTING =================
def weighted_voting_signals(df: pd.DataFrame, token: str, timeframe: str) -> float:
    votes = []
    weights = adaptive_weights.get(token, CATEGORY_WEIGHTS)
    
    # Structure signals
    ema20 = df["EMA_20"].iloc[-1]
    ema50 = df["EMA_50"].iloc[-1]
    ema200 = df["EMA_200"].iloc[-1]
    close = df["close"].iloc[-1]
    
    if close > ema20 > ema50 > ema200:
        votes.append(("structure", "BUY", weights["structure"]))
    elif close < ema20 < ema50 < ema200:
        votes.append(("structure", "SELL", weights["structure"]))
    else:
        votes.append(("structure", "HOLD", weights["structure"] * 0.5))
    
    # Momentum signals
    rsi = df["RSI"].iloc[-1]
    if rsi < 30:
        votes.append(("momentum", "BUY", weights["momentum"]))
    elif rsi > 70:
        votes.append(("momentum", "SELL", weights["momentum"]))
    else:
        votes.append(("momentum", "HOLD", weights["momentum"] * 0.5))
    
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_signal"].iloc[-1]
    if macd > macd_signal and df["MACD"].iloc[-2] <= df["MACD_signal"].iloc[-2]:
        votes.append(("momentum", "BUY", weights["momentum"] * 0.7))
    elif macd < macd_signal and df["MACD"].iloc[-2] >= df["MACD_signal"].iloc[-2]:
        votes.append(("momentum", "SELL", weights["momentum"] * 0.7))
    
    # Volume signals
    if len(df["OBV"]) >= 20:
        if df["OBV"].iloc[-1] > df["OBV"].iloc[-20]:
            votes.append(("volume", "BUY", weights["volume"]))
        else:
            votes.append(("volume", "SELL", weights["volume"]))
    else:
        votes.append(("volume", "HOLD", weights["volume"] * 0.5))
    
    # Candle patterns
    candle_patterns = detect_candle_patterns(df)
    if "HAMMER" in candle_patterns or "BULLISH_ENGULFING" in candle_patterns:
        votes.append(("candles", "BUY", weights["candles"]))
    elif "SHOOTING_STAR" in candle_patterns or "BEARISH_ENGULFING" in candle_patterns:
        votes.append(("candles", "SELL", weights["candles"]))
    else:
        votes.append(("candles", "HOLD", weights["candles"] * 0.5))
    
    # Exotic indicators (Bollinger Bands, Stochastic)
    bb_position = (close - df["BB_lower"].iloc[-1]) / (df["BB_upper"].iloc[-1] - df["BB_lower"].iloc[-1] + 1e-9)
    if bb_position < 0.2:
        votes.append(("exotic", "BUY", weights["exotic"]))
    elif bb_position > 0.8:
        votes.append(("exotic", "SELL", weights["exotic"]))
    
    stoch_k = df["STOCH_K"].iloc[-1]
    stoch_d = df["STOCH_D"].iloc[-1]
    if stoch_k < 20 and stoch_d < 20:
        votes.append(("exotic", "BUY", weights["exotic"] * 0.7))
    elif stoch_k > 80 and stoch_d > 80:
        votes.append(("exotic", "SELL", weights["exotic"] * 0.7))
    
    # Supertrend
    if df['Supertrend_Direction'].iloc[-1] == 1:
        votes.append(("structure", "BUY", weights["structure"] * 0.5))
    else:
        votes.append(("structure", "SELL", weights["structure"] * 0.5))
    
    # Harmonic patterns
    swing_highs, swing_lows = find_swing_points(df)
    harmonic_patterns = check_harmonic_patterns(swing_highs, swing_lows)
    
    for pattern, price, timestamp in harmonic_patterns:
        if "Bullish" in pattern and close <= price * 1.02:  # Within 2% of pattern target
            votes.append(("harmonics", "BUY", weights["harmonics"]))
        elif "Bearish" in pattern and close >= price * 0.98:  # Within 2% of pattern target
            votes.append(("harmonics", "SELL", weights["harmonics"]))
    
    # Calculate weighted score
    score = 0
    for category, vote, weight in votes:
        if vote == "BUY":
            score += weight
        elif vote == "SELL":
            score -= weight
    
    # Adjust based on trend
    if timeframe == "1d":  # Use higher timeframe for trend confirmation
        if is_bullish_trend(df):
            score += 0.1
        elif is_bearish_trend(df):
            score -= 0.1
    
    return score

# ================= ML MODEL =================
def train_ml_model(df: pd.DataFrame, token: str):
    df = add_indicators(df).dropna()
    df["target"] = (df["close"].shift(-1) - df["close"]).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    features = ["EMA_20", "EMA_50", "EMA_200", "RSI", "OBV", "ATR", "VWAP", "MACD", 
                "STOCH_K", "STOCH_D", "BB_upper", "BB_middle", "BB_lower", "ADX"]
    
    df = df.dropna(subset=features + ["target"])
    if len(df) < 100:  # Need enough data
        return
    
    X = df[features]
    y = df["target"]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X, y)
    ml_models[token] = clf
    logger.info(f"ML model trained for {token} with {len(df)} samples")

def ml_confidence(df: pd.DataFrame, token: str) -> float:
    clf = ml_models.get(token)
    if not clf:
        return 0
    
    features = ["EMA_20", "EMA_50", "EMA_200", "RSI", "OBV", "ATR", "VWAP", "MACD", 
                "STOCH_K", "STOCH_D", "BB_upper", "BB_middle", "BB_lower", "ADX"]
    
    X_last = df[features].iloc[[-1]]
    try:
        probs = clf.predict_proba(X_last)[0]
        # Assuming classes are [-1, 0, 1] for [SELL, HOLD, BUY]
        if hasattr(clf, 'classes_') and len(clf.classes_) == 3:
            sell_prob = probs[0] if clf.classes_[0] == -1 else 0
            buy_prob = probs[2] if clf.classes_[2] == 1 else 0
            return buy_prob - sell_prob
        else:
            return 0
    except:
        return 0

# ================= MULTI-TIMEFRAME ANALYSIS =================
async def multi_timeframe_analysis(symbol: str):
    scores = {}
    decisions = {}
    
    for tf in TIMEFRAMES:
        df = await fetch_kucoin_candles(symbol, tf, MAX_OHLCV)
        if df.empty:
            continue
        
        df = add_indicators(df).dropna()
        if len(df) < 50:  # Need enough data
            continue
        
        score = weighted_voting_signals(df, symbol.replace("-USDT", ""), tf)
        scores[tf] = score
        
        # Determine decision for this timeframe
        threshold = 0.2
        if score > threshold:
            decisions[tf] = "BUY"
        elif score < -threshold:
            decisions[tf] = "SELL"
        else:
            decisions[tf] = "HOLD"
    
    # Weighted combination of timeframe scores
    combined_score = 0
    for tf in TIMEFRAMES:
        if tf in scores:
            combined_score += scores[tf] * TIMEFRAME_WEIGHTS[tf]
    
    return combined_score, decisions, scores

# ================= ENHANCED ANALYZE SYMBOL =================

async def enhanced_analyze_symbol(symbol: str):
    """Напредна анализа на симбол со сите индикатори"""
    token = symbol.replace("-USDT", "")
    
    # Земи податоци од повеќе тајмфрејмови
    daily_df = await fetch_kucoin_candles(symbol, "1d", MAX_OHLCV)
    if daily_df.empty:
        return None
    
    daily_df = add_indicators(daily_df).dropna()
    if len(daily_df) < 50:
        return None
    
    current_price = daily_df["close"].iloc[-1]
    
    # Пресметај ги сите индикатори
    volume_profile = calculate_volume_profile(daily_df)
    divergences = detect_all_divergences(daily_df)
    price_targets = calculate_precise_price_targets(daily_df, current_price)
    
    # Детектирај тренд
    bullish_trend = is_bullish_trend(daily_df)
    bearish_trend = is_bearish_trend(daily_df)
    
    # Генерирај финален сигнал
    final_signal = generate_final_signal(
        daily_df, volume_profile, divergences, price_targets, 
        bullish_trend, bearish_trend
    )
    
    # Подготви извештај
    report = {
        'symbol': symbol,
        'current_price': current_price,
        'signal': final_signal,
        'price_targets': price_targets,
        'volume_profile': volume_profile,
        'divergences': divergences,
        'trend': 'BULLISH' if bullish_trend else 'BEARISH' if bearish_trend else 'NEUTRAL',
        'timestamp': datetime.utcnow()
    }
    
    return report

def generate_final_signal(df, volume_profile, divergences, price_targets, bullish, bearish):
    """Генерира финален торговен сигнал врз основа на сите индикатори"""
    signal_strength = 0
    signal_direction = "HOLD"
    
    # 1. Провери дивергенции (моќен сигнал)
    for div_type, index, indicator in divergences:
        if div_type == 'BULLISH_DIVERGENCE' and index >= len(df) - 3:  # Многу неодамнешна
            signal_strength += 8
            signal_direction = "BUY"
        elif div_type == 'BEARISH_DIVERGENCE' and index >= len(df) - 3:
            signal_strength += 8
            signal_direction = "SELL"
    
    # 2. Провери дали цената е близу Volume POC
    current_price = df['close'].iloc[-1]
    poc = volume_profile['poc']
    if abs(current_price - poc) / poc < 0.02:  # Within 2%
        signal_strength += 5
        # Ако е над POC во bullish тренд, купувај на падови кон POC
        if current_price > poc and bullish:
            signal_direction = "BUY" if signal_strength < 5 else signal_direction
        elif current_price < poc and bearish:
            signal_direction = "SELL" if signal_strength < 5 else signal_direction
    
    # 3. Провери RSI
    rsi = df['RSI'].iloc[-1]
    if rsi < 30:
        signal_strength += 4
        signal_direction = "BUY"
    elif rsi > 70:
        signal_strength += 4
        signal_direction = "SELL"
    
    # 4. Провери дали имаме силни ценовни цели
    if price_targets['buy_targets'] and price_targets['confidence'] > 0.7:
        signal_strength += 3
        signal_direction = "BUY"
    elif price_targets['sell_targets'] and price_targets['confidence'] > 0.7:
        signal_strength += 3
        signal_direction = "SELL"
    
    # 5. Финална одлука
    if signal_strength >= 10:
        return {
            'direction': signal_direction,
            'strength': min(signal_strength, 10),
            'buy_targets': price_targets['buy_targets'],
            'sell_targets': price_targets['sell_targets'],
            'confidence': price_targets['confidence']
        }
    else:
        return {
            'direction': "HOLD",
            'strength': signal_strength,
            'buy_targets': [],
            'sell_targets': [],
            'confidence': 0
        }

# ================= RISK MANAGEMENT =================
def calculate_position_size(atr, current_price, portfolio_value, risk_per_trade=0.02):
    """
    Calculate position size based on volatility (ATR).
    risk_per_trade: 2% of portfolio per trade
    """
    dollar_risk = portfolio_value * risk_per_trade
    atr_dollars = atr * current_price  # ATR in dollars
    position_size = dollar_risk / (atr_dollars * 2)  # Use 2x ATR for stop loss
    return max(0.001, position_size)  # Minimum position size

# ================= ADAPTIVE WEIGHTS =================
def update_adaptive_weights(token: str, decision: str, last_price: float, success: bool):
    history = signal_history[token]
    history.append((decision, last_price, success))
    if len(history) > 100:
        history.pop(0)
    
    # Calculate success rate for each category (simplified)
    success_rate = sum(1 for _, _, s in history if s) / len(history) if history else 0.5
    
    # Adjust weights based on overall success
    for cat in adaptive_weights[token]:
        if success_rate > 0.6:  # Good performance, increase weight slightly
            adaptive_weights[token][cat] *= 1.01
        elif success_rate < 0.4:  # Poor performance, decrease weight
            adaptive_weights[token][cat] *= 0.99
    
    # Normalize weights
    total = sum(adaptive_weights[token].values())
    for cat in adaptive_weights[token]:
        adaptive_weights[token][cat] /= total

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
