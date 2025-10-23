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
- GitHub Actions compatible
"""

import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ================= FIXED KUCOIN IMPORT =================
try:
    # For newer versions of python-kucoin
    from kucoin.client import Client
    KUCOIN_NEW_VERSION = True
except ImportError as e:
    logging.error(f"❌ Cannot import Kucoin libraries: {e}")
    logging.info("💡 Install with: pip install python-kucoin")
    KUCOIN_NEW_VERSION = False

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("⚠️ Telegram library not available. Install with: pip install python-telegram-bot")

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("⚠️ Scikit-learn not available. ML features disabled.")

try:
    import ta  # Pure Python alternative to TA-Lib
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("⚠️ TA library not available. Install with: pip install ta")

# ================= CONFIG =================
TOKENS = ["BTC", "ETH", "XRP", "HBAR", "LINK", "ONDO", "W", "ACH", "FET", "AVAX"]
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

# ================= LOGGING SETUP =================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("advanced_crypto_bot")

# ================= KUCOIN CLIENT INIT =================
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

market_client = None
try:
    from kucoin.client import Client
    
    if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE:
        # For python-kucoin version 2.2.0
        market_client = Client(
            KUCOIN_API_KEY,
            KUCOIN_API_SECRET, 
            KUCOIN_API_PASSPHRASE
        )
        logger.info("✅ KuCoin client initialized successfully")
        
        # Simple test - try to get ticker data to verify connection
        try:
            test_symbol = "BTC-USDT"
            test_data = market_client.get_ticker(test_symbol)
            if test_data:
                logger.info(f"✅ KuCoin connection verified - {test_symbol} data received")
            else:
                logger.warning("⚠️ KuCoin test returned no data, but client is initialized")
        except Exception as e:
            logger.warning(f"⚠️ KuCoin test call failed, but client may still work: {e}")
            
    else:
        logger.warning("❌ KuCoin client not initialized - missing API keys")
        
except ImportError as e:
    logger.error(f"❌ Cannot import Kucoin client: {e}")
    market_client = None
except Exception as e:
    logger.error(f"❌ KuCoin client initialization failed: {e}")
    market_client = None

# ================= TELEGRAM BOT INIT =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = None
if TELEGRAM_AVAILABLE and TELEGRAM_TOKEN and CHAT_ID:
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        logger.info("✅ Telegram bot initialized successfully")
    except Exception as e:
        logger.error(f"❌ Telegram bot initialization failed: {e}")
        bot = None
else:
    if not TELEGRAM_AVAILABLE:
        logger.warning("⚠️ Telegram library not installed")
    else:
        logger.warning("⚠️ Telegram bot not initialized - missing credentials")

# ================= STATE =================
last_price_sent = {}
last_sent_time = {}
adaptive_weights = {sym: CATEGORY_WEIGHTS.copy() for sym in TOKENS}
signal_history = {sym: [] for sym in TOKENS}
ml_models = {sym: None for sym in TOKENS}
pattern_history = {sym: [] for sym in TOKENS}

# ================= UTILITIES =================
def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

async def send_telegram(msg: str):
    if not bot:
        logger.error("❌ Telegram bot not initialized - cannot send message")
        return False
        
    if not CHAT_ID:
        logger.error("❌ CHAT_ID not set - cannot send message")
        return False
        
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg)
        logger.info("✅ Telegram message sent successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Telegram send error: {e}")
        return False

def smart_round(value: float) -> float:
    if value >= 1:
        return round(value, 2)
    elif value >= 0.01:
        return round(value, 4)
    else:
        return round(value, 8)

def format_price(price: float) -> str:
    """Форматирај цена со соодветен број на децимали според големината на цената"""
    if price >= 1000:
        return f"${price:.2f}"
    elif price >= 100:
        return f"${price:.3f}"
    elif price >= 10:
        return f"${price:.4f}"
    elif price >= 1:
        return f"${price:.5f}"
    elif price >= 0.1:
        return f"${price:.6f}"
    elif price >= 0.01:
        return f"${price:.7f}"
    else:
        return f"${price:.8f}"

# ================= FETCH CANDLES =================
async def fetch_kucoin_candles(symbol: str, tf: str = "1d", limit: int = 200):
    if market_client is None:
        logger.error(f"❌ KuCoin client not available for {symbol}")
        return pd.DataFrame()
        
    interval_map = {"1d": "1day", "4h": "4hour", "1h": "1hour", "1w": "1week", "15m": "15min"}
    interval = interval_map.get(tf, "1day")
    
    try:
        # Use the correct method for kucoin client v2.2.0 - get_kline_data instead of get_kline
        candles = market_client.get_kline_data(symbol, interval, limit=limit)
        
        if not candles:
            logger.error(f"❌ No data returned for {symbol}")
            return pd.DataFrame()
        
        # KuCoin returns data in format: [timestamp, open, close, high, low, volume, turnover]
        df = pd.DataFrame(candles, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
        
        # Convert to numeric types
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Remove rows with invalid data
        df = df.dropna(subset=["close", "volume"])
        
        # Convert timestamp - FIXED to avoid warning
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", errors="coerce")
        
        # Sort by timestamp and return
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"✅ Successfully fetched {len(df)} candles for {symbol}. Last price: {format_price(df['close'].iloc[-1])}")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
        
    except Exception as e:
        logger.error(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

# ================= SIMPLIFIED INDICATORS =================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:  # Minimum data needed
        return df
        
    df = df.copy()
    
    try:
        # Calculate indicators that don't require many previous values first
        df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
        
        # Simple RSI calculation (less aggressive on data requirements)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Simple OBV
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Simple MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        # Simple Bollinger Bands
        df["BB_middle"] = df["close"].rolling(window=20, min_periods=1).mean()
        bb_std = df["close"].rolling(window=20, min_periods=1).std()
        df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
        df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
        
        # VWAP
        df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        
        # Fill NaN values instead of dropping them
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='bfill').fillna(method='ffill')
        
    except Exception as e:
        logger.error(f"❌ Error calculating indicators: {e}")
    
    return df

# ================= VOLUME PROFILE & VWAP =================
def calculate_volume_profile(df, num_bins=20):
    """Пресметува Volume Profile (POC, Value Area)"""
    if df.empty:
        return {'poc': 0, 'value_area_high': 0, 'value_area_low': 0, 'volume_profile': {}}
        
    try:
        # Use only recent data for volume profile
        recent_df = df.tail(50) if len(df) > 50 else df
        
        # Сортирај ги цените во bins
        price_range = recent_df['high'].max() - recent_df['low'].min()
        if price_range == 0:  # Avoid division by zero
            return {'poc': recent_df['close'].iloc[-1], 'value_area_high': recent_df['close'].iloc[-1], 'value_area_low': recent_df['close'].iloc[-1], 'volume_profile': {}}
            
        bin_size = price_range / num_bins
        bins = [recent_df['low'].min() + i * bin_size for i in range(num_bins + 1)]
        
        # Пресметај волумен за секој bin
        volume_per_bin = {}
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            mask = (recent_df['close'] >= lower) & (recent_df['close'] < upper)
            volume_per_bin[f"{lower:.4f}-{upper:.4f}"] = recent_df.loc[mask, 'volume'].sum()
        
        if not volume_per_bin:
            return {'poc': recent_df['close'].iloc[-1], 'value_area_high': recent_df['close'].iloc[-1], 'value_area_low': recent_df['close'].iloc[-1], 'volume_profile': {}}
        
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
        
        value_area_high = max(value_area_prices) if value_area_prices else poc_value
        value_area_low = min(value_area_prices) if value_area_prices else poc_value
        
        return {
            'poc': poc_value,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'volume_profile': volume_per_bin
        }
    except Exception as e:
        logger.error(f"❌ Error calculating volume profile: {e}")
        return {'poc': df['close'].iloc[-1], 'value_area_high': df['close'].iloc[-1], 'value_area_low': df['close'].iloc[-1], 'volume_profile': {}}

def calculate_advanced_vwap(df):
    """Пресметува напреден VWAP со отстапувања"""
    try:
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
        
    except Exception as e:
        logger.error(f"❌ Error calculating advanced VWAP: {e}")
        # Add default VWAP
        df['VWAP'] = df['close']
        df['VWAP_upper_1'] = df['close'] * 1.01
        df['VWAP_upper_2'] = df['close'] * 1.02
        df['VWAP_lower_1'] = df['close'] * 0.99
        df['VWAP_lower_2'] = df['close'] * 0.98
    
    return df

# ================= HARMONIC PATTERNS =================
def find_swing_points(df, lookback=5):
    if df.empty or len(df) < lookback * 2:
        return [], []
        
    try:
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
    except Exception as e:
        logger.error(f"❌ Error finding swing points: {e}")
        return [], []

def check_harmonic_patterns(swing_highs, swing_lows):
    patterns = []
    
    try:
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
            
            if XA == 0:  # Avoid division by zero
                continue
                
            AB_ratio = AB / XA
            BC_ratio = BC / AB if AB != 0 else 0
            CD_ratio = CD / BC if BC != 0 else 0
            
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
    
    except Exception as e:
        logger.error(f"❌ Error checking harmonic patterns: {e}")
    
    return patterns

# ================= FIBONACCI & ELLIOTT WAVE =================
def fib_levels(df, lookback=100):
    if df.empty:
        return {}
        
    try:
        recent = df.tail(lookback)
        high, low = recent["high"].max(), recent["low"].min()
        diff = high - low
        if diff == 0:
            return {"0.0": high, "1.0": low}
            
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
    except Exception as e:
        logger.error(f"❌ Error calculating Fibonacci levels: {e}")
        return {"0.0": df['close'].iloc[-1], "1.0": df['close'].iloc[-1]}

# ================= DIVERGENCE DETECTION =================
def detect_all_divergences(df, lookback=14):
    """Детектира различен тип на дивергенции"""
    divergences = []
    
    if df.empty or 'RSI' not in df.columns:
        return divergences
    
    try:
        # Use only recent data for divergence detection
        recent_df = df.tail(30) if len(df) > 30 else df
        
        # RSI дивергенција
        rsi = recent_df['RSI'].values
        price = recent_df['close'].values
        
        # Најди peaks и troughs за цена и RSI
        price_peaks = [i for i in range(lookback, len(recent_df)-lookback) 
                      if price[i] == max(price[i-lookback:i+lookback+1])]
        price_troughs = [i for i in range(lookback, len(recent_df)-lookback) 
                        if price[i] == min(price[i-lookback:i+lookback+1])]
        
        rsi_peaks = [i for i in range(lookback, len(recent_df)-lookback) 
                    if rsi[i] == max(rsi[i-lookback:i+lookback+1])]
        rsi_troughs = [i for i in range(lookback, len(recent_df)-lookback) 
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
    
    except Exception as e:
        logger.error(f"❌ Error detecting divergences: {e}")
    
    return divergences

# ================= CANDLE PATTERNS =================
def detect_candle_patterns(df):
    patterns = []
    if len(df) < 2:
        return patterns
        
    try:
        # Use only last 2 candles for pattern detection
        open_price = df['open'].iloc[-1]
        close_price = df['close'].iloc[-1]
        high_price = df['high'].iloc[-1]
        low_price = df['low'].iloc[-1]
        prev_open = df['open'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        
        # Calculate candle properties
        body = abs(close_price - open_price)
        total_range = high_price - low_price
        if total_range == 0:
            return patterns
            
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
        prev_body = abs(prev_close - prev_open)
        if (body > prev_body * 1.5 and
            ((close_price > open_price and prev_close < prev_open and close_price > prev_open and open_price < prev_close) or
             (close_price < open_price and prev_close > prev_open and close_price < prev_open and open_price > prev_close))):
            patterns.append("BULLISH_ENGULFING" if close_price > open_price else "BEARISH_ENGULFING")
    
    except Exception as e:
        logger.error(f"❌ Error detecting candle patterns: {e}")
    
    return patterns

# ================= ENHANCED PRICE TARGETS =================
def calculate_precise_price_targets(df, current_price):
    """Пресметува прецизни buy/sell цели комбинирајќи повеќе методи"""
    targets = {
        'buy_targets': [],
        'sell_targets': [],
        'confidence': 0
    }
    
    if df.empty:
        return targets
    
    try:
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
        
        # 3. VWAP нивоа
        df_with_vwap = calculate_advanced_vwap(df)
        vwap = df_with_vwap['VWAP'].iloc[-1] if 'VWAP' in df_with_vwap.columns else current_price
        vwap_upper = df_with_vwap['VWAP_upper_1'].iloc[-1] if 'VWAP_upper_1' in df_with_vwap.columns else current_price * 1.02
        vwap_lower = df_with_vwap['VWAP_lower_1'].iloc[-1] if 'VWAP_lower_1' in df_with_vwap.columns else current_price * 0.98
        
        targets['buy_targets'].append(('VWAP Support', vwap_lower, 0.6))
        targets['buy_targets'].append(('VWAP', vwap, 0.5))
        targets['sell_targets'].append(('VWAP Resistance', vwap_upper, 0.6))
        
        # 4. Подредување на целите според confidence и растојание од тековната цена
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
    
    except Exception as e:
        logger.error(f"❌ Error calculating price targets: {e}")
    
    return targets

# ================= TREND ANALYSIS =================
def is_bullish_trend(df):
    if df.empty or 'EMA_20' not in df.columns:
        return False
    return df['close'].iloc[-1] > df['EMA_20'].iloc[-1]

def is_bearish_trend(df):
    if df.empty or 'EMA_20' not in df.columns:
        return False
    return df['close'].iloc[-1] < df['EMA_20'].iloc[-1]

# ================= SIMPLIFIED SIGNAL GENERATION =================
def generate_final_signal(df, volume_profile, divergences, price_targets, bullish, bearish):
    """Генерира финален торговен сигнал"""
    try:
        signal_strength = 0
        signal_direction = "HOLD"
        current_price = df['close'].iloc[-1]
        
        # 1. RSI signals
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        if rsi < 30:
            signal_strength += 3
            signal_direction = "BUY"
        elif rsi > 70:
            signal_strength += 3
            signal_direction = "SELL"
        
        # 2. MACD signals
        if 'MACD' in df.columns and 'MACD_signal' in df.columns and len(df) > 1:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            if macd > macd_signal:
                signal_strength += 2
                signal_direction = "BUY" if signal_strength < 3 else signal_direction
            else:
                signal_strength += 2
                signal_direction = "SELL" if signal_strength < 3 else signal_direction
        
        # 3. Trend signals
        if bullish:
            signal_strength += 1
        elif bearish:
            signal_strength -= 1
        
        # 4. Final decision
        if abs(signal_strength) >= 3:  # Lower threshold
            return {
                'direction': signal_direction,
                'strength': min(abs(signal_strength), 10),
                'buy_targets': price_targets['buy_targets'][:2],  # Only top 2 targets
                'sell_targets': price_targets['sell_targets'][:2], # Only top 2 targets
                'confidence': min(abs(signal_strength) / 10.0, 1.0)
            }
        else:
            return {
                'direction': "HOLD",
                'strength': 0,
                'buy_targets': [],
                'sell_targets': [],
                'confidence': 0
            }
            
    except Exception as e:
        logger.error(f"❌ Error generating final signal: {e}")
        return {
            'direction': "HOLD",
            'strength': 0,
            'buy_targets': [],
            'sell_targets': [],
            'confidence': 0
        }

# ================= ML MODEL =================
def train_ml_model(df: pd.DataFrame, token: str):
    if not SKLEARN_AVAILABLE:
        return
        
    try:
        df = add_indicators(df)
        if len(df) < 10:  # Need enough data
            return
            
        df["target"] = (df["close"].shift(-1) - df["close"]).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        features = ["EMA_20", "EMA_50", "RSI", "OBV", "MACD", "BB_upper", "BB_middle", "BB_lower"]
        
        # Keep only features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        df = df.dropna(subset=available_features + ["target"])
        
        if len(df) < 20:  # Need enough data
            return
        
        X = df[available_features]
        y = df["target"]
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
        clf.fit(X, y)
        ml_models[token] = clf
        logger.info(f"✅ ML model trained for {token} with {len(df)} samples")
    
    except Exception as e:
        logger.error(f"❌ Error training ML model for {token}: {e}")

# ================= ENHANCED ANALYZE SYMBOL =================
async def enhanced_analyze_symbol(symbol: str):
    """Напредна анализа на симбол со сите индикатори"""
    try:
        # Земи податоци од дневен тајмфрејм
        daily_df = await fetch_kucoin_candles(symbol, "1d", 200)
        if daily_df.empty:
            logger.error(f"❌ No data available for {symbol} - skipping")
            return None
        
        logger.info(f"📊 Raw data for {symbol}: {len(daily_df)} candles")
        
        # Add indicators without dropping NaN values aggressively
        daily_df = add_indicators(daily_df)
        
        # Keep only rows that have the essential indicators
        essential_cols = ['close', 'EMA_20', 'RSI', 'MACD']
        available_cols = [col for col in essential_cols if col in daily_df.columns]
        daily_df = daily_df.dropna(subset=available_cols)
        
        if len(daily_df) < 10:  # Reduced minimum requirement
            logger.error(f"❌ Not enough valid data for {symbol} - skipping (have {len(daily_df)}, need 10)")
            return None
        
        current_price = daily_df["close"].iloc[-1]
        logger.info(f"💰 {symbol} current price: {format_price(current_price)}, Valid data points: {len(daily_df)}")
        
        # Calculate basic indicators only
        volume_profile = calculate_volume_profile(daily_df.tail(50))  # Use last 50 points
        divergences = detect_all_divergences(daily_df.tail(30))  # Use last 30 points
        price_targets = calculate_precise_price_targets(daily_df, current_price)
        
        # Simple trend detection
        bullish_trend = len(daily_df) > 20 and daily_df['close'].iloc[-1] > daily_df['EMA_20'].iloc[-1]
        bearish_trend = len(daily_df) > 20 and daily_df['close'].iloc[-1] < daily_df['EMA_20'].iloc[-1]
        
        # Generate final signal
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
        
    except Exception as e:
        logger.error(f"❌ Error analyzing {symbol}: {e}")
        return None

# ================= RISK MANAGEMENT =================
def calculate_position_size(atr, current_price, portfolio_value, risk_per_trade=0.02):
    """
    Calculate position size based on volatility (ATR).
    risk_per_trade: 2% of portfolio per trade
    """
    try:
        dollar_risk = portfolio_value * risk_per_trade
        atr_dollars = atr * current_price  # ATR in dollars
        if atr_dollars == 0:
            return 0.001
        position_size = dollar_risk / (atr_dollars * 2)  # Use 2x ATR for stop loss
        return max(0.001, position_size)  # Minimum position size
    except Exception as e:
        logger.error(f"❌ Error calculating position size: {e}")
        return 0.001

# ================= DEBUG KUCOIN CONNECTION =================
async def debug_kucoin_connection():
    """Debug KuCoin connection"""
    if not market_client:
        logger.error("❌ KuCoin client not available for debug")
        return False
    
    try:
        test_symbol = "BTC-USDT"
        logger.info(f"🔧 Testing KuCoin with {test_symbol}...")
        
        # Test ticker
        ticker = market_client.get_ticker(test_symbol)
        logger.info(f"✅ KuCoin ticker: {format_price(float(ticker.get('price', 0)))}")
        
        # Test klines
        klines = market_client.get_kline_data(test_symbol, '1hour', limit=5)
        logger.info(f"✅ KuCoin klines: {len(klines)} candles")
        
        return True
    except Exception as e:
        logger.error(f"❌ KuCoin debug failed: {e}")
        return False

# ================= MAIN EXECUTION =================
async def github_actions_production():
    """Production режим за GitHub Actions со вистински податоци и Telegram - SIMPLIFIED"""
    logger.info("🚀 Starting analysis with real market data...")
    
    # Debug KuCoin connection first
    await debug_kucoin_connection()
    
    # Check Telegram configuration
    if not TELEGRAM_AVAILABLE:
        logger.error("❌ Telegram library not available")
    elif not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN not set")
    elif not CHAT_ID:
        logger.error("❌ CHAT_ID not set")
    else:
        logger.info("✅ Telegram configuration OK")
    
    # Test Telegram connection
    if TELEGRAM_AVAILABLE and TELEGRAM_TOKEN and CHAT_ID:
        try:
            test_msg = "🤖 Crypto Bot Started Successfully!\n📊 Beginning market analysis..."
            await send_telegram(test_msg)
            logger.info("✅ Test Telegram message sent")
        except Exception as e:
            logger.error(f"❌ Test Telegram message failed: {e}")
    
    # Train ML models with real data
    if SKLEARN_AVAILABLE:
        logger.info("📚 Training ML models...")
        for sym in TOKENS:
            symbol = sym + "-USDT"
            try:
                df = await fetch_kucoin_candles(symbol, "1d", 200)
                if not df.empty:
                    logger.info(f"📈 Data for {symbol}: {len(df)} candles, last price: {format_price(df['close'].iloc[-1])}")
                    train_ml_model(df, sym)
                    logger.info(f"✅ Trained model for {symbol}")
                else:
                    logger.warning(f"⚠️ No data for {symbol}")
            except Exception as e:
                logger.error(f"Error training {symbol}: {e}")
    
    # Analyze all tokens and send ONLY meaningful signals
    logger.info("🔍 Analyzing tokens...")
    strong_signals = 0
    analyzed_tokens = 0
    
    for sym in TOKENS:
        symbol = sym + "-USDT"
        try:
            logger.info(f"🔍 Analyzing {symbol}...")
            report = await enhanced_analyze_symbol(symbol)
            if report:
                analyzed_tokens += 1
                signal = report['signal']
                
                logger.info(f"📊 {symbol} analysis: {signal['direction']} (Strength: {signal['strength']}/10)")
                
                # Send message ONLY for meaningful signals (strength >= 3)
                if signal['strength'] >= 3:
                    # Create clean signal message
                    direction_emoji = "🟢" if signal['direction'] == "BUY" else "🔴" if signal['direction'] == "SELL" else "🟡"
                    
                    msg = (f"{direction_emoji} **{signal['direction']} SIGNAL** {direction_emoji}\n"
                           f"📊 **{symbol}**\n"
                           f"💰 **Price: {format_price(report['current_price'])}**\n"
                           f"💪 Strength: {signal['strength']}/10\n"
                           f"📈 Trend: {report['trend']}")
                    
                    # Add targets if available
                    if signal['buy_targets']:
                        msg += f"\n🎯 **BUY TARGETS:**\n"
                        for i, (name, price, confidence) in enumerate(signal['buy_targets'][:2], 1):
                            msg += f"{i}. {format_price(price)} ({int(confidence*100)}%)\n"
                    
                    if signal['sell_targets']:
                        msg += f"\n🎯 **SELL TARGETS:**\n"
                        for i, (name, price, confidence) in enumerate(signal['sell_targets'][:2], 1):
                            msg += f"{i}. {format_price(price)} ({int(confidence*100)}%)\n"
                    
                    # Send message
                    logger.info(f"📨 Attempting to send Telegram for {symbol}...")
                    success = await send_telegram(msg)
                    if success:
                        strong_signals += 1
                        logger.info(f"✅ Sent signal for {symbol}: {signal['direction']} (Strength: {signal['strength']})")
                    else:
                        logger.error(f"❌ Failed to send Telegram for {symbol}")
                    
                    # Wait between messages
                    await asyncio.sleep(2)
                else:
                    logger.info(f"⏭️  Skipping weak signal for {symbol}: {signal['direction']} (Strength: {signal['strength']})")
            else:
                logger.warning(f"⚠️ No report generated for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
    
    logger.info(f"✅ Analysis completed. Analyzed: {analyzed_tokens}/{len(TOKENS)}, Strong signals: {strong_signals}")
    
    # Send summary message
    if TELEGRAM_AVAILABLE and TELEGRAM_TOKEN and CHAT_ID:
        summary_msg = f"📊 Analysis Complete\nAnalyzed: {analyzed_tokens}/{len(TOKENS)} tokens\nStrong signals: {strong_signals}"
        await send_telegram(summary_msg)
    
    return analyzed_tokens > 0

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    # Check if we're in GitHub Actions
    is_github_actions = os.getenv("GITHUB_ACTIONS")
    
    # Check if we have all required API keys
    has_api_keys = all([
        os.getenv("KUCOIN_API_KEY"),
        os.getenv("KUCOIN_API_SECRET"), 
        os.getenv("KUCOIN_API_PASSPHRASE")
    ])
    
    if is_github_actions:
        if has_api_keys:
            logger.info("🚀 Starting analysis with real market data")
            try:
                # Run production analysis with 5 minute timeout
                result = asyncio.run(asyncio.wait_for(github_actions_production(), timeout=300))
                if result:
                    logger.info("✅ Analysis completed successfully!")
                    exit(0)
                else:
                    logger.error("❌ Analysis failed")
                    exit(1)
            except asyncio.TimeoutError:
                logger.warning("⏰ Analysis timed out")
                exit(0)
            except Exception as e:
                logger.error(f"💥 Analysis failed: {e}")
                exit(1)
        else:
            logger.error("❌ Missing KuCoin API keys")
            exit(1)
    else:
        # Local execution
        if has_api_keys:
            logger.info("🚀 Starting local analysis")
            # Then run main analysis
            asyncio.run(github_actions_production())
        else:
            logger.error("❌ Missing KuCoin API keys for local execution")
            exit(1)
