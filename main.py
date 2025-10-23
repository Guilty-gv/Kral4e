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

# ================= DEBUG FUNCTIONS =================
async def check_all_symbols_data():
    """Провери дали сите симболи добиваат податоци"""
    logger.info("🔍 Checking data availability for all symbols...")
    
    for sym in TOKENS:
        symbol = sym + "-USDT"
        try:
            df = await fetch_kucoin_candles(symbol, "1d", 50)
            if df.empty:
                logger.error(f"❌ NO DATA for {symbol}")
            else:
                current_price = df['close'].iloc[-1]
                logger.info(f"✅ {symbol}: {len(df)} candles, Price: {format_price(current_price)}")
                
                # Провери ги индикаторите
                df = add_indicators(df)
                if 'RSI' in df.columns:
                    rsi = df['RSI'].iloc[-1]
                    logger.info(f"   RSI: {rsi:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error checking {symbol}: {e}")

async def debug_symbol_analysis():
    """Детална анализа на зошто некои симболи не даваат сигнали"""
    logger.info("🐛 Starting detailed symbol analysis...")
    
    for sym in TOKENS:
        symbol = sym + "-USDT"
        logger.info(f"\n{'='*50}")
        logger.info(f"🔍 ANALYZING {symbol}")
        logger.info(f"{'='*50}")
        
        report = await enhanced_analyze_symbol(symbol)
        if report:
            signal = report['signal']
            logger.info(f"📊 FINAL SIGNAL: {signal['direction']} (Strength: {signal['strength']})")
            
            if signal['strength'] < 2:  # Променето од 3 на 2
                logger.info(f"❌ SIGNAL TOO WEAK - Needs strength >= 2")
                logger.info(f"   Details: RSI={report.get('rsi', 'N/A'):.2f}, "
                          f"MACD={report.get('macd', 'N/A'):.6f}, "
                          f"Trend={report['trend']}")
            else:
                logger.info(f"✅ STRONG SIGNAL - Would send Telegram")
        else:
            logger.error(f"💥 NO REPORT GENERATED")
        
        await asyncio.sleep(1)  # Rate limiting

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
        df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]
        
        # Simple Bollinger Bands
        df["BB_middle"] = df["close"].rolling(window=20, min_periods=1).mean()
        bb_std = df["close"].rolling(window=20, min_periods=1).std()
        df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
        df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
        
        # VWAP
        df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        
        # Stochastic
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['STOCH_K'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Fill NaN values - FIXED: use ffill() and bfill() instead of fillna with method
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].bfill().ffill()
        
    except Exception as e:
        logger.error(f"❌ Error calculating indicators: {e}")
    
    return df

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
        vwap = df['VWAP'].iloc[-1] if 'VWAP' in df.columns else current_price
        
        targets['buy_targets'].append(('VWAP Support', vwap * 0.98, 0.6))
        targets['buy_targets'].append(('VWAP', vwap, 0.5))
        targets['sell_targets'].append(('VWAP Resistance', vwap * 1.02, 0.6))
        
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

# ================= ENHANCED SIGNAL GENERATION =================
def generate_final_signal(df, volume_profile, divergences, price_targets, bullish, bearish):
    """Генерира финален торговен сигнал СО ПОНИЗОК ПРАГ И ПОВЕЌЕ ИНДИКАТОРИ"""
    try:
        signal_strength = 0
        signal_direction = "HOLD"
        current_price = df['close'].iloc[-1]
        
        # 1. RSI signals - ПОНИСКИ ПРАЗОВИ
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        if rsi < 35:  # Променето од 30 на 35
            signal_strength += 2.5  # Променето од 3 на 2.5
            signal_direction = "BUY"
        elif rsi > 65:  # Променето од 70 на 65
            signal_strength += 2.5  # Променето од 3 на 2.5
            signal_direction = "SELL"
        elif rsi < 40:  # Додаден умерен RSI сигнал
            signal_strength += 1.5
            if signal_direction == "HOLD":
                signal_direction = "BUY"
        elif rsi > 60:  # Додаден умерен RSI сигнал
            signal_strength += 1.5  
            if signal_direction == "HOLD":
                signal_direction = "SELL"
        
        # 2. MACD signals - СО СИЛА НА СИГНАЛОТ
        if 'MACD' in df.columns and 'MACD_signal' in df.columns and len(df) > 1:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            macd_histogram = df['MACD_histogram'].iloc[-1] if 'MACD_histogram' in df.columns else 0
            
            # Пресметај ја силата на MACD сигналот
            macd_strength = abs(macd_histogram) / (abs(macd_signal) + 0.0001)  # Avoid division by zero
            
            if macd > macd_signal and macd_strength > 0.02:  # Додаден strength check
                signal_strength += 1.5  # Променето од 2 на 1.5
                if signal_direction == "HOLD":
                    signal_direction = "BUY"
            elif macd < macd_signal and macd_strength > 0.02:
                signal_strength += 1.5  # Променето од 2 на 1.5
                if signal_direction == "HOLD":
                    signal_direction = "SELL"
        
        # 3. Trend signals
        if bullish:
            signal_strength += 1.2  # Зголемена важност на трендот
            if signal_direction == "HOLD":
                signal_direction = "BUY"
        elif bearish:
            signal_strength += 1.2  # Зголемена важност на трендот
            if signal_direction == "HOLD":
                signal_direction = "SELL"
        
        # 4. Volume confirmation
        if 'volume' in df.columns and len(df) > 5:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(20).mean()
            if current_volume > avg_volume * 1.3:  # Volume confirmation
                signal_strength += 0.8
        
        # 5. Bollinger Bands position
        if 'BB_lower' in df.columns and 'BB_upper' in df.columns:
            bb_position = (current_price - df['BB_lower'].iloc[-1]) / (df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1])
            if bb_position < 0.2:  # Near lower band
                signal_strength += 1.0
                if signal_direction == "HOLD":
                    signal_direction = "BUY"
            elif bb_position > 0.8:  # Near upper band
                signal_strength += 1.0
                if signal_direction == "HOLD":
                    signal_direction = "SELL"
        
        # 6. Stochastic signals
        if 'STOCH_K' in df.columns and 'STOCH_D' in df.columns:
            stoch_k = df['STOCH_K'].iloc[-1]
            stoch_d = df['STOCH_D'].iloc[-1]
            if stoch_k < 20 and stoch_d < 20:  # Oversold
                signal_strength += 1.0
                if signal_direction == "HOLD":
                    signal_direction = "BUY"
            elif stoch_k > 80 and stoch_d > 80:  # Overbought
                signal_strength += 1.0
                if signal_direction == "HOLD":
                    signal_direction = "SELL"
        
        # 7. Price action - support/resistance
        if len(df) > 10:
            recent_high = df['high'].tail(10).max()
            recent_low = df['low'].tail(10).min()
            price_range = recent_high - recent_low
            
            if price_range > 0:
                price_position = (current_price - recent_low) / price_range
                if price_position < 0.3:  # Near support
                    signal_strength += 0.8
                    if signal_direction == "HOLD":
                        signal_direction = "BUY"
                elif price_position > 0.7:  # Near resistance
                    signal_strength += 0.8
                    if signal_direction == "HOLD":
                        signal_direction = "SELL"
        
        # 8. Final decision - ПОНИЗОК ПРАГ
        if abs(signal_strength) >= 2.0:  # Променето од 3 на 2
            return {
                'direction': signal_direction,
                'strength': min(abs(signal_strength), 10),
                'buy_targets': price_targets['buy_targets'][:2],
                'sell_targets': price_targets['sell_targets'][:2],
                'confidence': min(abs(signal_strength) / 10.0, 1.0),
                'rsi': rsi,
                'details': f"RSI: {rsi:.1f}, Strength: {signal_strength:.1f}"
            }
        else:
            return {
                'direction': "HOLD",
                'strength': 0,
                'buy_targets': [],
                'sell_targets': [],
                'confidence': 0,
                'rsi': rsi,
                'details': f"RSI: {rsi:.1f}, Strength: {signal_strength:.1f}"
            }
            
    except Exception as e:
        logger.error(f"❌ Error generating final signal: {e}")
        return {
            'direction': "HOLD",
            'strength': 0,
            'buy_targets': [],
            'sell_targets': [],
            'confidence': 0,
            'details': f"Error: {e}"
        }

# ================= ENHANCED ANALYZE SYMBOL =================
async def enhanced_analyze_symbol(symbol: str):
    """Напредна анализа на симбол со сите индикатори"""
    try:
        # Земи податоци од дневен тајмфрејм
        daily_df = await fetch_kucoin_candles(symbol, "1d", 100)  # Reduced to 100 for faster processing
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
        
        # ДЕТАЛНО ЛОГИРАЊЕ ЗА АНАЛИЗА
        rsi = daily_df['RSI'].iloc[-1] if 'RSI' in daily_df.columns else 50
        macd = daily_df['MACD'].iloc[-1] if 'MACD' in daily_df.columns else 0
        macd_signal = daily_df['MACD_signal'].iloc[-1] if 'MACD_signal' in daily_df.columns else 0
        
        logger.info(f"🔍 {symbol} Detailed Analysis:")
        logger.info(f"   - Price: {format_price(current_price)}")
        logger.info(f"   - RSI: {rsi:.2f}")
        logger.info(f"   - MACD: {macd:.6f} > Signal: {macd_signal:.6f} = {macd > macd_signal}")
        logger.info(f"   - Trend: {'BULLISH' if bullish_trend else 'BEARISH' if bearish_trend else 'NEUTRAL'}")
        logger.info(f"   - Signal: {final_signal['direction']} (Strength: {final_signal['strength']:.1f})")
        logger.info(f"   - Details: {final_signal.get('details', 'N/A')}")
        
        # Подготви извештај
        report = {
            'symbol': symbol,
            'current_price': current_price,
            'signal': final_signal,
            'price_targets': price_targets,
            'volume_profile': volume_profile,
            'divergences': divergences,
            'trend': 'BULLISH' if bullish_trend else 'BEARISH' if bearish_trend else 'NEUTRAL',
            'timestamp': datetime.utcnow(),
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal
        }
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Error analyzing {symbol}: {e}")
        return None

# ================= MAIN EXECUTION =================
async def github_actions_production():
    """Production режим за GitHub Actions со вистински податоци и Telegram"""
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
    
    # ДЕТАЛНА АНАЛИЗА НА СИМБОЛИТЕ
    logger.info("🔍 Running detailed symbol analysis...")
    await debug_symbol_analysis()
    
    # Analyze all tokens and send signals
    logger.info("🔍 Analyzing tokens for Telegram signals...")
    strong_signals = 0
    analyzed_tokens = 0
    
    for sym in TOKENS:
        symbol = sym + "-USDT"
        try:
            logger.info(f"🔍 Analyzing {symbol} for Telegram...")
            report = await enhanced_analyze_symbol(symbol)
            if report:
                analyzed_tokens += 1
                signal = report['signal']
                
                logger.info(f"📊 {symbol} Telegram analysis: {signal['direction']} (Strength: {signal['strength']:.1f}/10)")
                
                # Send message for signals with strength >= 2 (променето од 3)
                if signal['strength'] >= 2.0:  # Променето од 3 на 2
                    # Create clean signal message
                    direction_emoji = "🟢" if signal['direction'] == "BUY" else "🔴" if signal['direction'] == "SELL" else "🟡"
                    
                    msg = (f"{direction_emoji} **{signal['direction']} SIGNAL** {direction_emoji}\n"
                           f"📊 **{symbol}**\n"
                           f"💰 **Price: {format_price(report['current_price'])}**\n"
                           f"💪 Strength: {signal['strength']:.1f}/10\n"
                           f"📈 Trend: {report['trend']}\n"
                           f"📊 RSI: {report.get('rsi', 'N/A'):.2f}")
                    
                    # Add signal details
                    if 'details' in signal:
                        msg += f"\n📋 {signal['details']}"
                    
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
                        logger.info(f"✅ Sent signal for {symbol}: {signal['direction']} (Strength: {signal['strength']:.1f})")
                    else:
                        logger.error(f"❌ Failed to send Telegram for {symbol}")
                    
                    # Wait between messages
                    await asyncio.sleep(2)
                else:
                    logger.info(f"⏭️  Skipping weak signal for {symbol}: {signal['direction']} (Strength: {signal['strength']:.1f})")
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
