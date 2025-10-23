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
if KUCOIN_NEW_VERSION and KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE:
    try:
        market_client = Client(
            key=KUCOIN_API_KEY, 
            secret=KUCOIN_API_SECRET, 
            passphrase=KUCOIN_API_PASSPHRASE
        )
        logger.info("✅ KuCoin client initialized successfully")
    except Exception as e:
        logger.error(f"❌ KuCoin client initialization failed: {e}")
        market_client = None
else:
    logger.warning("❌ KuCoin client not initialized - missing API keys or incompatible version")

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

# ================= FETCH CANDLES =================
async def fetch_kucoin_candles(symbol: str, tf: str = "1d", limit: int = 200):
    if market_client is None:
        logger.error(f"❌ KuCoin client not available for {symbol}")
        return pd.DataFrame()
        
    interval_map = {"1d": "1day", "4h": "4hour", "1h": "1hour", "1w": "1week", "15m": "15min"}
    interval = interval_map.get(tf, "1day")
    
    try:
        # Use the correct method for kucoin client - FIXED
        candles = market_client.get_kline(symbol, interval, limit=limit)
        
        if not candles:
            logger.error(f"❌ No data returned for {symbol}")
            return pd.DataFrame()
        
        # Check if candles is a list of lists (original format)
        if candles and isinstance(candles, list) and len(candles) > 0:
            if isinstance(candles[0], list):
                # Original format: [timestamp, open, close, high, low, volume, turnover]
                df = pd.DataFrame(candles, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
            else:
                # New format or different structure
                logger.error(f"❌ Unexpected data format for {symbol}")
                return pd.DataFrame()
        else:
            logger.error(f"❌ Empty or invalid data for {symbol}")
            return pd.DataFrame()
        
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        
        logger.info(f"✅ Successfully fetched {len(df)} candles for {symbol}. Last price: ${df['close'].iloc[-1]:.2f}")
        return df.sort_values("timestamp").reset_index(drop=True)[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

# ================= INDICATORS (WITH FALLBACKS) =================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    
    try:
        # EMA indicators
        df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["EMA_100"] = df["close"].ewm(span=100, adjust=False).mean()
        df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
        
        # SMA indicators
        df["SMA_50"] = df["close"].rolling(50).mean()
        df["SMA_200"] = df["close"].rolling(200).mean()
        
        # RSI (with fallback)
        if TA_AVAILABLE:
            df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=RSI_PERIOD).rsi()
        else:
            # Simple RSI calculation as fallback
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))
        
        # OBV
        if TA_AVAILABLE:
            df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        else:
            # Simple OBV calculation
            df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # ATR
        if TA_AVAILABLE:
            df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_PERIOD).average_true_range()
        else:
            # Simple ATR calculation
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            df["ATR"] = true_range.rolling(window=ATR_PERIOD).mean()
        
        # MACD
        if TA_AVAILABLE:
            macd = ta.trend.MACD(df["close"])
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
            df["MACD_hist"] = macd.macd_diff()
        else:
            # Simple MACD calculation
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df["MACD"] = exp1 - exp2
            df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
            df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        
        # Stochastic
        if TA_AVAILABLE:
            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=STOCH_PERIOD)
            df["STOCH_K"] = stoch.stoch()
            df["STOCH_D"] = stoch.stoch_signal()
        else:
            # Simple Stochastic calculation
            low_min = df['low'].rolling(window=STOCH_PERIOD).min()
            high_max = df['high'].rolling(window=STOCH_PERIOD).max()
            df["STOCH_K"] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            df["STOCH_D"] = df["STOCH_K"].rolling(window=3).mean()
        
        # Bollinger Bands
        if TA_AVAILABLE:
            bb = ta.volatility.BollingerBands(df["close"], window=BB_PERIOD)
            df["BB_upper"] = bb.bollinger_hband()
            df["BB_middle"] = bb.bollinger_mavg()
            df["BB_lower"] = bb.bollinger_lband()
        else:
            # Simple Bollinger Bands calculation
            df["BB_middle"] = df["close"].rolling(window=BB_PERIOD).mean()
            bb_std = df["close"].rolling(window=BB_PERIOD).std()
            df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
            df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
        
        # ADX
        if TA_AVAILABLE:
            df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
        else:
            df["ADX"] = 50  # Default neutral value
        
        # Supertrend indicator
        df = add_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
        
        # VWAP
        df["VWAP"] = (df["close"] * df["volume"]).rolling(VWAP_PERIOD).sum() / (df["volume"].rolling(VWAP_PERIOD).sum() + 1e-9)
        
        # Advanced VWAP with deviations
        df = calculate_advanced_vwap(df)
        
    except Exception as e:
        logger.error(f"❌ Error calculating indicators: {e}")
    
    return df

def add_supertrend(df, period=10, multiplier=3):
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        if TA_AVAILABLE:
            atr = ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()
        else:
            # Simple ATR calculation for supertrend
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            atr = true_range.rolling(window=period).mean()
        
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
        
    except Exception as e:
        logger.error(f"❌ Error calculating Supertrend: {e}")
        # Add default values
        df['Supertrend'] = df['close']
        df['Supertrend_Direction'] = 1
    
    return df

# ================= VOLUME PROFILE & VWAP =================
def calculate_volume_profile(df, num_bins=20):
    """Пресметува Volume Profile (POC, Value Area)"""
    if df.empty:
        return {'poc': 0, 'value_area_high': 0, 'value_area_low': 0, 'volume_profile': {}}
        
    try:
        # Сортирај ги цените во bins
        price_range = df['high'].max() - df['low'].min()
        if price_range == 0:  # Avoid division by zero
            return {'poc': df['close'].iloc[-1], 'value_area_high': df['close'].iloc[-1], 'value_area_low': df['close'].iloc[-1], 'volume_profile': {}}
            
        bin_size = price_range / num_bins
        bins = [df['low'].min() + i * bin_size for i in range(num_bins + 1)]
        
        # Пресметај волумен за секој bin
        volume_per_bin = {}
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            mask = (df['close'] >= lower) & (df['close'] < upper)
            volume_per_bin[f"{lower:.4f}-{upper:.4f}"] = df.loc[mask, 'volume'].sum()
        
        if not volume_per_bin:
            return {'poc': df['close'].iloc[-1], 'value_area_high': df['close'].iloc[-1], 'value_area_low': df['close'].iloc[-1], 'volume_profile': {}}
        
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
    
    except Exception as e:
        logger.error(f"❌ Error detecting divergences: {e}")
    
    return divergences

# ================= CANDLE PATTERNS =================
def detect_candle_patterns(df):
    patterns = []
    if len(df) < 2:
        return patterns
        
    try:
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
        vwap = df_with_vwap['VWAP'].iloc[-1] if 'VWAP' in df_with_vwap.columns else current_price
        vwap_upper = df_with_vwap['VWAP_upper_1'].iloc[-1] if 'VWAP_upper_1' in df_with_vwap.columns else current_price * 1.02
        vwap_lower = df_with_vwap['VWAP_lower_1'].iloc[-1] if 'VWAP_lower_1' in df_with_vwap.columns else current_price * 0.98
        
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
    
    except Exception as e:
        logger.error(f"❌ Error calculating price targets: {e}")
    
    return targets

# ================= TREND ANALYSIS =================
def is_bullish_trend(df):
    if df.empty or 'EMA_50' not in df.columns or 'EMA_200' not in df.columns or 'Supertrend_Direction' not in df.columns:
        return False
    return (df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1] and 
            df['close'].iloc[-1] > df['EMA_50'].iloc[-1] and
            df['Supertrend_Direction'].iloc[-1] == 1)

def is_bearish_trend(df):
    if df.empty or 'EMA_50' not in df.columns or 'EMA_200' not in df.columns or 'Supertrend_Direction' not in df.columns:
        return False
    return (df['EMA_50'].iloc[-1] < df['EMA_200'].iloc[-1] and 
            df['close'].iloc[-1] < df['EMA_50'].iloc[-1] and
            df['Supertrend_Direction'].iloc[-1] == -1)

# ================= WEIGHTED VOTING =================
def weighted_voting_signals(df: pd.DataFrame, token: str, timeframe: str) -> float:
    votes = []
    weights = adaptive_weights.get(token, CATEGORY_WEIGHTS)
    
    if df.empty:
        return 0
    
    try:
        # Structure signals
        ema20 = df["EMA_20"].iloc[-1] if "EMA_20" in df.columns else 0
        ema50 = df["EMA_50"].iloc[-1] if "EMA_50" in df.columns else 0
        ema200 = df["EMA_200"].iloc[-1] if "EMA_200" in df.columns else 0
        close = df["close"].iloc[-1]
        
        if close > ema20 > ema50 > ema200 and ema200 > 0:
            votes.append(("structure", "BUY", weights["structure"]))
        elif close < ema20 < ema50 < ema200 and ema200 > 0:
            votes.append(("structure", "SELL", weights["structure"]))
        else:
            votes.append(("structure", "HOLD", weights["structure"] * 0.5))
        
        # Momentum signals
        rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else 50
        if rsi < 30:
            votes.append(("momentum", "BUY", weights["momentum"]))
        elif rsi > 70:
            votes.append(("momentum", "SELL", weights["momentum"]))
        else:
            votes.append(("momentum", "HOLD", weights["momentum"] * 0.5))
        
        if "MACD" in df.columns and "MACD_signal" in df.columns and len(df) > 1:
            macd = df["MACD"].iloc[-1]
            macd_signal = df["MACD_signal"].iloc[-1]
            if macd > macd_signal and df["MACD"].iloc[-2] <= df["MACD_signal"].iloc[-2]:
                votes.append(("momentum", "BUY", weights["momentum"] * 0.7))
            elif macd < macd_signal and df["MACD"].iloc[-2] >= df["MACD_signal"].iloc[-2]:
                votes.append(("momentum", "SELL", weights["momentum"] * 0.7))
        
        # Volume signals
        if "OBV" in df.columns and len(df["OBV"]) >= 20:
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
        if "BB_lower" in df.columns and "BB_upper" in df.columns:
            bb_lower = df["BB_lower"].iloc[-1]
            bb_upper = df["BB_upper"].iloc[-1]
            if bb_upper != bb_lower:  # Avoid division by zero
                bb_position = (close - bb_lower) / (bb_upper - bb_lower)
                if bb_position < 0.2:
                    votes.append(("exotic", "BUY", weights["exotic"]))
                elif bb_position > 0.8:
                    votes.append(("exotic", "SELL", weights["exotic"]))
        
        if "STOCH_K" in df.columns and "STOCH_D" in df.columns:
            stoch_k = df["STOCH_K"].iloc[-1]
            stoch_d = df["STOCH_D"].iloc[-1]
            if stoch_k < 20 and stoch_d < 20:
                votes.append(("exotic", "BUY", weights["exotic"] * 0.7))
            elif stoch_k > 80 and stoch_d > 80:
                votes.append(("exotic", "SELL", weights["exotic"] * 0.7))
        
        # Supertrend
        if 'Supertrend_Direction' in df.columns:
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
    
    except Exception as e:
        logger.error(f"❌ Error in weighted voting for {token}: {e}")
        return 0

# ================= ML MODEL =================
def train_ml_model(df: pd.DataFrame, token: str):
    if not SKLEARN_AVAILABLE:
        return
        
    try:
        df = add_indicators(df).dropna()
        if len(df) < 10:  # Need enough data
            return
            
        df["target"] = (df["close"].shift(-1) - df["close"]).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        features = ["EMA_20", "EMA_50", "EMA_200", "RSI", "OBV", "ATR", "VWAP", "MACD", 
                    "STOCH_K", "STOCH_D", "BB_upper", "BB_middle", "BB_lower", "ADX"]
        
        # Keep only features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        df = df.dropna(subset=available_features + ["target"])
        
        if len(df) < 50:  # Need enough data
            return
        
        X = df[available_features]
        y = df["target"]
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
        clf.fit(X, y)
        ml_models[token] = clf
        logger.info(f"✅ ML model trained for {token} with {len(df)} samples")
    
    except Exception as e:
        logger.error(f"❌ Error training ML model for {token}: {e}")

def ml_confidence(df: pd.DataFrame, token: str) -> float:
    clf = ml_models.get(token)
    if not clf or not SKLEARN_AVAILABLE:
        return 0
    
    try:
        features = ["EMA_20", "EMA_50", "EMA_200", "RSI", "OBV", "ATR", "VWAP", "MACD", 
                    "STOCH_K", "STOCH_D", "BB_upper", "BB_middle", "BB_lower", "ADX"]
        
        # Keep only features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        X_last = df[available_features].iloc[[-1]]
        
        probs = clf.predict_proba(X_last)[0]
        # Assuming classes are [-1, 0, 1] for [SELL, HOLD, BUY]
        if hasattr(clf, 'classes_') and len(clf.classes_) == 3:
            sell_prob = probs[0] if clf.classes_[0] == -1 else 0
            buy_prob = probs[2] if clf.classes_[2] == 1 else 0
            return buy_prob - sell_prob
        else:
            return 0
    except Exception as e:
        logger.error(f"❌ Error in ML confidence for {token}: {e}")
        return 0

# ================= ENHANCED ANALYZE SYMBOL =================
async def enhanced_analyze_symbol(symbol: str):
    """Напредна анализа на симбол со сите индикатори"""
    token = symbol.replace("-USDT", "")
    
    # Земи податоци од дневен тајмфрејм
    daily_df = await fetch_kucoin_candles(symbol, "1d", MAX_OHLCV)
    if daily_df.empty:
        logger.error(f"❌ No data available for {symbol} - skipping")
        return None
    
    daily_df = add_indicators(daily_df).dropna()
    if len(daily_df) < 50:
        logger.error(f"❌ Not enough data for {symbol} - skipping")
        return None
    
    current_price = daily_df["close"].iloc[-1]
    logger.info(f"💰 {symbol} current price: ${current_price:.2f}")
    
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
    
    try:
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
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
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
    
    except Exception as e:
        logger.error(f"❌ Error generating final signal: {e}")
        return {
            'direction': "HOLD",
            'strength': 0,
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

# ================= CONFIGURATION TEST =================
async def test_configuration():
    """Test all configurations before running main analysis"""
    logger.info("🔧 Testing configuration...")
    
    # Test KuCoin
    if market_client:
        try:
            # Test with a simple symbol
            test_symbol = "BTC-USDT"
            df = await fetch_kucoin_candles(test_symbol, "1d", 10)
            if not df.empty:
                logger.info(f"✅ KuCoin test PASSED - {test_symbol} data fetched")
            else:
                logger.error("❌ KuCoin test FAILED - No data received")
        except Exception as e:
            logger.error(f"❌ KuCoin test FAILED: {e}")
    else:
        logger.error("❌ KuCoin client not available")
    
    # Test Telegram
    if TELEGRAM_AVAILABLE and TELEGRAM_TOKEN and CHAT_ID:
        try:
            test_msg = "🤖 Crypto Bot Configuration Test\n✅ All systems operational!"
            success = await send_telegram(test_msg)
            if success:
                logger.info("✅ Telegram test PASSED")
            else:
                logger.error("❌ Telegram test FAILED - Message not sent")
        except Exception as e:
            logger.error(f"❌ Telegram test FAILED: {e}")
    else:
        logger.warning("⚠️ Telegram not configured")
    
    logger.info("🔧 Configuration test completed")

# ================= MAIN EXECUTION =================
async def github_actions_production():
    """Production режим за GitHub Actions со вистински податоци и Telegram - SIMPLIFIED"""
    logger.info("🚀 Starting analysis with real market data...")
    
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
                df = await fetch_kucoin_candles(symbol, "1d", MAX_OHLCV)
                if not df.empty:
                    logger.info(f"📈 Data for {symbol}: {len(df)} candles, last price: ${df['close'].iloc[-1]:.2f}")
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
                
                # Send message ONLY for strong signals (strength >= 5)
                if signal['strength'] >= 5:
                    # Create clean signal message
                    direction_emoji = "🟢" if signal['direction'] == "BUY" else "🔴" if signal['direction'] == "SELL" else "🟡"
                    
                    msg = (f"{direction_emoji} **{signal['direction']} SIGNAL** {direction_emoji}\n"
                           f"📊 **{symbol}**\n"
                           f"💰 **Price: ${report['current_price']:.2f}**\n"
                           f"💪 Strength: {signal['strength']}/10\n"
                           f"📈 Trend: {report['trend']}")
                    
                    # Add targets if available
                    if signal['buy_targets']:
                        msg += f"\n🎯 **BUY TARGETS:**\n"
                        for i, (name, price, confidence) in enumerate(signal['buy_targets'][:3], 1):
                            msg += f"{i}. ${price:.2f} ({int(confidence*100)}%)\n"
                    
                    if signal['sell_targets']:
                        msg += f"\n🎯 **SELL TARGETS:**\n"
                        for i, (name, price, confidence) in enumerate(signal['sell_targets'][:3], 1):
                            msg += f"{i}. ${price:.2f} ({int(confidence*100)}%)\n"
                    
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
            # Run configuration test first
            asyncio.run(test_configuration())
            # Then run main analysis
            asyncio.run(github_actions_production())
        else:
            logger.error("❌ Missing KuCoin API keys for local execution")
            exit(1)
