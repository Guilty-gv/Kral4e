# -*- coding: utf-8 -*-
"""
QUANTUM HYBRID BOT v3.0 - COMPLETE ENHANCED VERSION
- Market Regime Detection
- Advanced Risk Management  
- Enhanced Pattern Recognition
- Sentiment Analysis
- Lightweight Deep Learning
- Quantum-Inspired Optimization
- Real-time Adaptation
- Advanced Technical Analysis Structure
"""

import os, asyncio, logging, math, random, json, aiohttp
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# KUCOIN IMPORT FIX
try:
    from kucoin.client import Client
    market_client = None
except ImportError as e:
    print(f"KuCoin import error: {e}")
    market_client = None

from telegram import Bot
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import differential_evolution

# ============ ENHANCED CONFIGURATION ============
TOKENS = ["BTC","ETH","ONDO","XRP","LINK","FET","W","ACH","WAXL","HBAR"]
MAX_OHLCV = 500
TIMEFRAMES = ["1h","4h","1d"]
PRICE_ALERT_THRESHOLD = 0.03
COOLDOWN_MINUTES = 60

# ENHANCED BASE WEIGHTS WITH CATEGORIES
BASE_WEIGHTS = {
    "structure": 0.25,    # Најважно - тренд и структура
    "momentum": 0.20,     # Второ важно - моментум  
    "volume": 0.15,       # Волумен потврда
    "candles": 0.15,      # Свеќки патерни
    "exotic": 0.10,       # Хармонични патерни
    "ema_sets": 0.08,     # EMA комбинации
    "momentum_ex": 0.07   # Дополнителен моментум
}

# REGIME-BASED WEIGHT ADJUSTMENTS
REGIME_WEIGHTS = {
    "BULL": {
        "momentum": 1.3,      # ↑ Моментумот е поважен
        "structure": 0.8,     # ↓ Структурата помалку важна
        "volume": 1.1         # ↑ Волуменот потврда
    },
    "BEAR": {
        "structure": 1.4,     # ↑ Структурата многу важна
        "volume": 1.2,        # ↑ Волумен потврда
        "momentum": 0.7       # ↓ Моментумот помалку важен
    },
    "SIDEWAYS": {
        "ema_sets": 1.6,      # ↑ EMA кросовери важни
        "candles": 1.3,       # ↑ Свеќки патерни важни
        "momentum_ex": 0.7    # ↓ Моментумот неважен
    },
    "VOLATILE": {
        "exotic": 1.5,        # ↑ Хармонични патерни важни
        "candles": 1.3,       # ↑ Свеќки патерни важни
        "momentum": 0.8       # ↓ Моментумот неважен
    }
}

# API Setup
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

# Initialize market client
if KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE:
    try:
        market_client = Client(
            api_key=KUCOIN_API_KEY,
            api_secret=KUCOIN_API_SECRET,
            api_passphrase=KUCOIN_API_PASSPHRASE
        )
        logger.info("✅ KuCoin client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize KuCoin client: {e}")
        market_client = None
else:
    print("KuCoin API credentials not found")
    market_client = None

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quantum_bot_v3")

# ============ ADVANCED TECHNICAL ANALYSIS ============
class AdvancedTechnicalAnalyzer:
    def __init__(self):
        self.pattern_history = {}
    
    def calculate_structure_score(self, df: pd.DataFrame) -> float:
        """Calculate structure score (25% weight) - BOS/CHoCH, EMA trend, Support/Resistance"""
        score = 0.0
        
        try:
            # 1. EMA Trend Structure
            if all(col in df.columns for col in ['EMA_50', 'EMA_200']):
                if df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1]:
                    score += 0.4  # Bullish trend structure
                else:
                    score -= 0.4  # Bearish trend structure
            
            # 2. Price Channel Breakout (BOS - Break of Structure)
            if len(df) > 20:
                recent_high = df['high'].tail(20).max()
                recent_low = df['low'].tail(20).min()
                current_price = df['close'].iloc[-1]
                
                # Breakout above resistance
                if current_price > recent_high:
                    score += 0.3
                # Breakdown below support  
                elif current_price < recent_low:
                    score -= 0.3
            
            # 3. Higher Highs/Lower Lows (CHoCH - Change of Character)
            if len(df) > 50:
                # Check for trend reversal patterns
                highs = df['high'].tail(50)
                lows = df['low'].tail(50)
                
                # Higher Highs pattern
                if (highs.iloc[-1] > highs.iloc[-3] and 
                    highs.iloc[-3] > highs.iloc[-5]):
                    score += 0.3
                # Lower Lows pattern
                elif (lows.iloc[-1] < lows.iloc[-3] and 
                      lows.iloc[-3] < lows.iloc[-5]):
                    score -= 0.3
            
        except Exception as e:
            logger.error(f"Structure score calculation error: {e}")
        
        return max(-1.0, min(1.0, score))
    
    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score (20% weight) - RSI, MACD, Stochastic"""
        score = 0.0
        
        try:
            # 1. RSI Momentum
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if rsi < 30:
                    score += 0.4  # Oversold bounce potential
                elif rsi > 70:
                    score -= 0.4  # Overbought pullback potential
                elif 40 < rsi < 60:
                    score += 0.1  # Healthy momentum
            
            # 2. MACD Momentum
            if all(col in df.columns for col in ['MACD', 'MACD_signal']):
                macd = df['MACD'].iloc[-1]
                macd_signal = df['MACD_signal'].iloc[-1]
                if macd > macd_signal:
                    score += 0.3
                else:
                    score -= 0.3
            
            # 3. Stochastic Momentum
            if 'STOCH_K' in df.columns:
                stoch_k = df['STOCH_K'].iloc[-1]
                if stoch_k < 20:
                    score += 0.2
                elif stoch_k > 80:
                    score -= 0.2
            
            # 4. Price Momentum
            if len(df) > 10:
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
                if abs(price_change) > 0.05:  # 5% move
                    score += np.sign(price_change) * 0.1
            
        except Exception as e:
            logger.error(f"Momentum score calculation error: {e}")
        
        return max(-1.0, min(1.0, score))
    
    def calculate_volume_score(self, df: pd.DataFrame) -> float:
        """Calculate volume score (15% weight) - OBV, VWAP, Volume Divergence"""
        score = 0.0
        
        try:
            # 1. OBV Trend
            if 'OBV' in df.columns and len(df) > 20:
                obv_trend = df['OBV'].iloc[-1] - df['OBV'].iloc[-20]
                price_trend = df['close'].iloc[-1] - df['close'].iloc[-20]
                
                # Volume confirmation
                if obv_trend * price_trend > 0:
                    score += 0.4  # Volume confirms price movement
                else:
                    score -= 0.3  # Volume divergence
            
            # 2. VWAP Position
            if 'VWAP' in df.columns:
                vwap_position = (df['close'].iloc[-1] - df['VWAP'].iloc[-1]) / df['VWAP'].iloc[-1]
                if vwap_position > 0.02:  # Above VWAP
                    score += 0.3
                elif vwap_position < -0.02:  # Below VWAP
                    score -= 0.3
            
            # 3. Volume Spike
            if 'volume' in df.columns and len(df) > 10:
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].tail(10).mean()
                if current_volume > avg_volume * 1.5:
                    score += 0.3  # High volume confirmation
            
        except Exception as e:
            logger.error(f"Volume score calculation error: {e}")
        
        return max(-1.0, min(1.0, score))
    
    def calculate_candle_patterns_score(self, df: pd.DataFrame) -> float:
        """Calculate candle patterns score (15% weight) - Doji, Morning/Evening Star, etc."""
        score = 0.0
        
        try:
            if len(df) < 3:
                return score
            
            current = df.iloc[-1]
            prev1 = df.iloc[-2]
            prev2 = df.iloc[-3]
            
            # 1. Doji Pattern (Indecision)
            body_size = abs(current['close'] - current['open'])
            range_size = current['high'] - current['low']
            
            if range_size > 0 and body_size / range_size < 0.1:
                # Doji detected - market indecision
                score -= 0.2
            
            # 2. Morning Star Pattern (Bullish Reversal)
            if (prev2['close'] < prev2['open'] and  # First red candle
                abs(prev1['close'] - prev1['open']) / (prev1['high'] - prev1['low']) < 0.3 and  # Small body
                current['close'] > current['open'] and  # Green candle
                current['close'] > (prev2['open'] + prev2['close']) / 2):  # Closes above first candle midpoint
                score += 0.5
            
            # 3. Evening Star Pattern (Bearish Reversal)
            if (prev2['close'] > prev2['open'] and  # First green candle
                abs(prev1['close'] - prev1['open']) / (prev1['high'] - prev1['low']) < 0.3 and  # Small body
                current['close'] < current['open'] and  # Red candle
                current['close'] < (prev2['open'] + prev2['close']) / 2):  # Closes below first candle midpoint
                score -= 0.5
            
            # 4. Three White Soldiers (Strong Bullish)
            if (len(df) >= 3 and
                all(df['close'].iloc[-i] > df['open'].iloc[-i] for i in range(1, 4)) and
                all(df['close'].iloc[-i] > df['close'].iloc[-i-1] for i in range(1, 3))):
                score += 0.4
            
            # 5. Three Black Crows (Strong Bearish)
            if (len(df) >= 3 and
                all(df['close'].iloc[-i] < df['open'].iloc[-i] for i in range(1, 4)) and
                all(df['close'].iloc[-i] < df['close'].iloc[-i-1] for i in range(1, 3))):
                score -= 0.4
            
        except Exception as e:
            logger.error(f"Candle patterns score calculation error: {e}")
        
        return max(-1.0, min(1.0, score))
    
    def calculate_exotic_patterns_score(self, df: pd.DataFrame) -> float:
        """Calculate exotic patterns score (10% weight) - Harmonic, Elliott Waves"""
        score = 0.0
        
        try:
            # Simplified harmonic pattern detection
            if len(df) < 30:
                return score
            
            # 1. ABCD Pattern Detection
            abcd_score = self.detect_abcd_pattern(df)
            score += abcd_score * 0.3
            
            # 2. Simple Elliott Wave count
            wave_score = self.simple_elliott_wave_detection(df)
            score += wave_score * 0.2
            
            # 3. Fibonacci Retracement levels
            fib_score = self.fibonacci_retracement_analysis(df)
            score += fib_score * 0.5
            
        except Exception as e:
            logger.error(f"Exotic patterns score calculation error: {e}")
        
        return max(-1.0, min(1.0, score))
    
    def detect_abcd_pattern(self, df: pd.DataFrame) -> float:
        """Detect basic ABCD harmonic pattern"""
        try:
            if len(df) < 20:
                return 0.0
            
            # Simplified ABCD pattern detection
            prices = df['close'].tail(20).values
            
            # Find potential swing points
            swings = []
            for i in range(2, len(prices)-2):
                if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and
                    prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                    swings.append(('high', i, prices[i]))
                elif (prices[i] < prices[i-1] and prices[i] < prices[i-2] and
                      prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                    swings.append(('low', i, prices[i]))
            
            # Basic ABCD pattern logic
            if len(swings) >= 4:
                # Look for A-B-C-D pattern
                for i in range(len(swings)-3):
                    a, b, c, d = swings[i], swings[i+1], swings[i+2], swings[i+3]
                    
                    # Bullish ABCD pattern
                    if (a[0] == 'low' and b[0] == 'high' and 
                        c[0] == 'low' and d[0] == 'high' and
                        d[2] > b[2]):  # D higher than B
                        return 0.8
                    
                    # Bearish ABCD pattern  
                    elif (a[0] == 'high' and b[0] == 'low' and
                          c[0] == 'high' and d[0] == 'low' and
                          d[2] < b[2]):  # D lower than B
                        return -0.8
            
        except Exception as e:
            logger.error(f"ABCD pattern detection error: {e}")
        
        return 0.0
    
    def simple_elliott_wave_detection(self, df: pd.DataFrame) -> float:
        """Simple Elliott Wave pattern detection"""
        try:
            if len(df) < 50:
                return 0.0
            
            prices = df['close'].tail(50).values
            
            # Calculate price movement characteristics
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            # Simple impulse wave detection
            if len(returns) >= 5:
                # Look for 3 advancing waves and 2 corrective waves
                advances = sum(1 for r in returns[-5:] if r > 0)
                declines = sum(1 for r in returns[-5:] if r < 0)
                
                if advances >= 3:
                    return 0.6  # Potential impulse wave up
                elif declines >= 3:
                    return -0.6  # Potential impulse wave down
            
        except Exception as e:
            logger.error(f"Elliott wave detection error: {e}")
        
        return 0.0
    
    def fibonacci_retracement_analysis(self, df: pd.DataFrame) -> float:
        """Fibonacci retracement level analysis"""
        try:
            if len(df) < 30:
                return 0.0
            
            recent = df.tail(30)
            high = recent['high'].max()
            low = recent['low'].min()
            current = df['close'].iloc[-1]
            
            if high == low:
                return 0.0
            
            # Fibonacci levels
            fib_levels = {
                0.0: high,
                0.236: high - (high - low) * 0.236,
                0.382: high - (high - low) * 0.382,
                0.5: high - (high - low) * 0.5,
                0.618: high - (high - low) * 0.618,
                0.786: high - (high - low) * 0.786,
                1.0: low
            }
            
            # Find which Fibonacci level we're near
            for level, price in fib_levels.items():
                if abs(current - price) / price < 0.02:  # Within 2%
                    if level in [0.382, 0.5]:  # Common bounce levels
                        return 0.7
                    elif level in [0.618, 0.786]:  # Deep retracement
                        return -0.5
                    elif level in [0.0, 0.236]:  # Resistance levels
                        return -0.3
                    elif level == 1.0:  # Support level
                        return 0.3
            
        except Exception as e:
            logger.error(f"Fibonacci analysis error: {e}")
        
        return 0.0
    
    def calculate_ema_sets_score(self, df: pd.DataFrame) -> float:
        """Calculate EMA sets score (8% weight) - Multiple EMA crossovers"""
        score = 0.0
        
        try:
            # Multiple EMA crossover analysis
            ema_pairs = [(9, 21), (20, 50), (50, 200)]
            bullish_crossovers = 0
            total_pairs = 0
            
            for fast, slow in ema_pairs:
                fast_col = f'EMA_{fast}'
                slow_col = f'EMA_{slow}'
                
                if fast_col in df.columns and slow_col in df.columns:
                    total_pairs += 1
                    if df[fast_col].iloc[-1] > df[slow_col].iloc[-1]:
                        bullish_crossovers += 1
            
            if total_pairs > 0:
                bull_ratio = bullish_crossovers / total_pairs
                if bull_ratio >= 0.67:  # 2/3 or more bullish
                    score += 0.8
                elif bull_ratio <= 0.33:  # 1/3 or less bullish
                    score -= 0.8
                else:  # Mixed signals
                    score += (bull_ratio - 0.5) * 0.4
            
        except Exception as e:
            logger.error(f"EMA sets score calculation error: {e}")
        
        return max(-1.0, min(1.0, score))
    
    def calculate_momentum_ex_score(self, df: pd.DataFrame) -> float:
        """Calculate extra momentum score (7% weight) - Additional momentum indicators"""
        score = 0.0
        
        try:
            # 1. Rate of Change (ROC)
            if len(df) > 10:
                roc = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
                if abs(roc) > 0.05:  # 5% move in 10 periods
                    score += np.sign(roc) * 0.3
            
            # 2. Price Acceleration
            if len(df) > 5:
                recent_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                prev_change = (df['close'].iloc[-5] - df['close'].iloc[-10]) / df['close'].iloc[-10]
                
                if recent_change > prev_change * 1.5:  # Acceleration
                    score += 0.2
                elif recent_change < prev_change * 0.5:  # Deceleration
                    score -= 0.2
            
            # 3. Volatility-adjusted momentum
            if 'ATR' in df.columns and df['ATR'].iloc[-1] > 0:
                atr_ratio = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['ATR'].iloc[-1]
                if abs(atr_ratio) > 2:  # Strong move relative to volatility
                    score += np.sign(atr_ratio) * 0.2
            
        except Exception as e:
            logger.error(f"Extra momentum score calculation error: {e}")
        
        return max(-1.0, min(1.0, score))
    
    def get_advanced_technical_score(self, df: pd.DataFrame, regime: str) -> Dict:
        """Calculate comprehensive technical score with all categories"""
        scores = {}
        
        try:
            # Calculate individual category scores
            scores['structure'] = self.calculate_structure_score(df)
            scores['momentum'] = self.calculate_momentum_score(df)
            scores['volume'] = self.calculate_volume_score(df)
            scores['candles'] = self.calculate_candle_patterns_score(df)
            scores['exotic'] = self.calculate_exotic_patterns_score(df)
            scores['ema_sets'] = self.calculate_ema_sets_score(df)
            scores['momentum_ex'] = self.calculate_momentum_ex_score(df)
            
            # Apply regime-based weight adjustments
            adjusted_weights = BASE_WEIGHTS.copy()
            if regime in REGIME_WEIGHTS:
                for category, multiplier in REGIME_WEIGHTS[regime].items():
                    if category in adjusted_weights:
                        adjusted_weights[category] *= multiplier
            
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            for category in adjusted_weights:
                adjusted_weights[category] /= total_weight
            
            # Calculate weighted final score
            final_score = 0.0
            for category, weight in adjusted_weights.items():
                final_score += scores[category] * weight
            
            return {
                'final_score': max(-1.0, min(1.0, final_score)),
                'category_scores': scores,
                'adjusted_weights': adjusted_weights,
                'regime': regime
            }
            
        except Exception as e:
            logger.error(f"Advanced technical score calculation error: {e}")
            return {
                'final_score': 0.0,
                'category_scores': {},
                'adjusted_weights': BASE_WEIGHTS,
                'regime': regime
            }

# ============ ENHANCED MAIN BOT CLASS ============
class QuantumHybridBotV3:
    def __init__(self):
        # Core components
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = AdvancedRiskManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = AdvancedTechnicalAnalyzer()  # NEW
        
        # New enhanced components
        self.light_dl = LightweightDeepLearner()
        self.quantum_optimizer = QuantumInspiredOptimizer()
        self.realtime_adaptor = RealTimeAdaptor()
        
        # State
        self.adaptive_weights = {sym: BASE_WEIGHTS.copy() for sym in TOKENS}
        self.signal_history = {sym: [] for sym in TOKENS}
        self.ml_models = {sym: None for sym in TOKENS}
        self.last_price_sent = {}
        self.last_sent_time = {}
    
    async def analyze_symbol(self, symbol: str, account_balance=10000):
        """Complete enhanced analysis with advanced technical structure"""
        token = symbol.replace("-USDT", "")
        
        # Fetch data and basic indicators
        df = await self.fetch_data_with_indicators(symbol, "1h")
        if df.empty:
            return
        
        last_price = df['close'].iloc[-1]
        
        # 1. MARKET REGIME DETECTION
        regime, regime_confidence = self.regime_detector.detect_regime(df)
        
        # 2. ADVANCED TECHNICAL ANALYSIS (NEW STRUCTURE)
        technical_analysis = self.technical_analyzer.get_advanced_technical_score(df, regime)
        technical_score = technical_analysis['final_score']
        
        # 3. SENTIMENT ANALYSIS
        fear_greed = await self.sentiment_analyzer.get_fear_greed_index()
        sentiment_label, sentiment_score = self.sentiment_analyzer.interpret_sentiment(fear_greed)
        
        # 4. LIGHT DEEP LEARNING PREDICTION
        dl_signal, dl_confidence = self.light_dl.predict(df, token)
        
        # 5. QUANTUM OPTIMIZATION (periodic)
        if len(self.signal_history[token]) > 50 and len(self.signal_history[token]) % 50 == 0:
            performance_data = self.quantum_optimizer.collect_performance_data(self.signal_history)
            optimized_weights = self.quantum_optimizer.optimize_weights(performance_data)
            if optimized_weights:
                self.adaptive_weights[token] = optimized_weights
                logger.info(f"Updated weights for {token}: {optimized_weights}")
        
        # 6. REAL-TIME ADAPTATION
        adaptation_boost = self.realtime_adaptor.get_boost(token, regime)
        
        # 7. COMBINE ALL SIGNALS WITH ENHANCED FORMULA
        final_score = self.combine_signals_enhanced(
            technical_score=technical_score,
            dl_signal=dl_signal,
            dl_confidence=dl_confidence,
            sentiment_score=sentiment_score,
            regime_impact=self.calculate_regime_impact(regime),
            adaptation_boost=adaptation_boost,
            technical_details=technical_analysis
        )
        
        # Enhanced decision making with dynamic thresholds
        threshold = self.calculate_dynamic_threshold(regime, fear_greed)
        decision = "BUY" if final_score > threshold else "SELL" if final_score < -threshold else "HOLD"
        
        # Risk management
        volatility = df['ATR'].iloc[-1] / last_price if 'ATR' in df.columns else 0.02
        position_size = self.risk_manager.calculate_position_size(
            account_balance, final_score, volatility, regime
        )
        
        # Enhanced price targets
        buy_target, sell_target, fibs = self.hybrid_price_targets(df, last_price)
        
        # Check alert conditions
        if not self.should_alert(symbol, last_price, decision):
            return
        
        # 8. ENHANCED REPORTING WITH DETAILED ANALYSIS
        await self.send_enhanced_alert_detailed(
            symbol=symbol,
            decision=decision,
            last_price=last_price,
            final_score=final_score,
            regime=regime,
            sentiment=(sentiment_label, fear_greed),
            dl_confidence=dl_confidence,
            position_size=position_size,
            buy_target=buy_target,
            sell_target=sell_target,
            technical_analysis=technical_analysis
        )
        
        # Record for learning
        self.record_signal_enhanced(token, decision, final_score, regime, technical_analysis)
        
        # Update DL model periodically
        if len(self.signal_history[token]) % 100 == 0:
            self.light_dl.train_model(df, token)
    
    def combine_signals_enhanced(self, technical_score, dl_signal, dl_confidence, 
                               sentiment_score, regime_impact, adaptation_boost, technical_details):
        """Enhanced signal combination formula"""
        # Weighted combination based on your structure
        final = (technical_score * 0.4 +          # Технички_Скор × 40%
                dl_signal * dl_confidence * 0.3 + # Deep_Learning_Скор × 30%
                sentiment_score * 0.2 +           # Сентимент_Скор × 20%
                regime_impact * 0.1)              # Режим_Влијание × 10%
        
        # Apply adaptation boost
        final *= adaptation_boost
        
        return max(-1.0, min(1.0, final))
    
    def calculate_regime_impact(self, regime):
        """Enhanced regime-based impact"""
        impacts = {
            'BULL': 0.15,      # Positive in bull markets
            'BEAR': -0.15,     # Negative in bear markets  
            'SIDEWAYS': 0.0,   # Neutral in sideways
            'VOLATILE': -0.08  # Slightly negative in volatile
        }
        return impacts.get(regime, 0.0)
    
    def calculate_dynamic_threshold(self, regime, fear_greed):
        """Enhanced dynamic decision threshold"""
        base_threshold = 0.15
        
        # Regime adjustments
        regime_adjustments = {
            'VOLATILE': 0.10,   # Higher threshold in volatility
            'BULL': 0.08,       # Lower threshold in bull markets
            'BEAR': 0.12,       # Higher threshold in bear markets
            'SIDEWAYS': 0.09    # Medium threshold in sideways
        }
        threshold = base_threshold + regime_adjustments.get(regime, 0.0)
        
        # Sentiment adjustment
        if fear_greed >= 75 or fear_greed <= 25:  # Extreme sentiment
            threshold += 0.05
        
        return max(0.1, min(0.3, threshold))  # Keep within reasonable bounds
    
    async def send_enhanced_alert_detailed(self, symbol, decision, last_price, final_score,
                                         regime, sentiment, dl_confidence, position_size,
                                         buy_target, sell_target, technical_analysis):
        """Send detailed enhanced Telegram alert with category breakdown"""
        sentiment_label, fear_greed = sentiment
        category_scores = technical_analysis.get('category_scores', {})
        
        direction_emoji = "🟢" if decision == "BUY" else "🔴" if decision == "SELL" else "🟡"
        
        msg = (f"{direction_emoji} **QUANTUM BOT v3.0 - ADVANCED ANALYSIS** {direction_emoji}\n"
               f"⏰ {now_str()}\n"
               f"📌 **{symbol}**\n"
               f"💰 **Price**: {format_price(last_price)}\n"
               f"🔔 **Decision**: **{decision}** (Score: {final_score:.3f})\n"
               f"📊 **Regime**: {regime}\n"
               f"😊 **Sentiment**: {sentiment_label} (F&G: {fear_greed})\n"
               f"🧠 **DL Confidence**: {dl_confidence:.2f}\n"
               f"💼 **Position**: ${position_size:.2f}\n\n"
               f"🎯 **TECHNICAL BREAKDOWN:**\n")
        
        # Add category scores
        for category, score in category_scores.items():
            score_emoji = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "🟡"
            msg += f"{score_emoji} {category.upper()}: {score:.3f}\n"
        
        msg += f"\n🎯 **TARGETS:**\n"
        msg += f"🛒 Buy: {format_price(buy_target)}\n"
        msg += f"💵 Sell: {format_price(sell_target)}\n"
        msg += f"⚡ **Advanced Structure Analysis**")
        
        logger.info(f"v3 Advanced Analysis: {symbol} -> {decision} (Score: {final_score:.3f})")
        await send_telegram(msg)
    
    def record_signal_enhanced(self, token, decision, score, regime, technical_analysis):
        """Enhanced signal recording with technical details"""
        signal_data = {
            'timestamp': datetime.utcnow(),
            'decision': decision,
            'score': score,
            'regime': regime,
            'category_scores': technical_analysis.get('category_scores', {}),
            'adjusted_weights': technical_analysis.get('adjusted_weights', {}),
            'technical_score': technical_analysis.get('final_score', 0)
        }
        
        self.signal_history[token].append(signal_data)
        if len(self.signal_history[token]) > 1000:
            self.signal_history[token].pop(0)

# ============ ENHANCED UTILITY FUNCTIONS ============
def format_price(price: float) -> str:
    """Format price with appropriate decimals based on price size"""
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

# Keep the rest of your existing functions (fetch_kucoin_candles, add_core_indicators, etc.)
# but enhance add_core_indicators to include all the new indicators:

def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced core technical indicators with all required metrics"""
    if df.empty: 
        return df
        
    df = df.copy()
    
    try:
        # EMAs for different timeframes
        for span in [9, 20, 21, 50, 200]:
            df[f"EMA_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["ATR"] = true_range.rolling(14, min_periods=1).mean()
        
        # Bollinger Bands
        df["BB_middle"] = df["close"].rolling(window=20, min_periods=1).mean()
        bb_std = df["close"].rolling(window=20, min_periods=1).std()
        df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
        df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
        
        # Stochastic
        lowest_low = df['low'].rolling(window=14, min_periods=1).min()
        highest_high = df['high'].rolling(window=14, min_periods=1).max()
        df['STOCH_K'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df['STOCH_D'] = df['STOCH_K'].rolling(window=3, min_periods=1).mean()
        
        # OBV (On-Balance Volume)
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # VWAP
        df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Fill NaN values safely
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].bfill().ffill()
        
    except Exception as e:
        logger.error(f"Error calculating enhanced indicators: {e}")
    
    return df

# ============ MAIN EXECUTION ============
bot_v3 = QuantumHybridBotV3()

async def main_loop():
    """Enhanced main execution loop"""
    logger.info("🚀 Starting Quantum Hybrid Bot v3.0 with Advanced Technical Analysis...")
    
    # Check if KuCoin client is available
    if not market_client:
        logger.error("KuCoin client not available. Check API credentials.")
        return
    
    # Enhanced initialization
    logger.info("🧠 Initializing advanced technical analysis system...")
    
    # Test data fetching and indicator calculation
    logger.info("🔍 Testing advanced indicator calculation...")
    for sym in TOKENS[:2]:  # Test with first 2 symbols
        symbol = sym + "-USDT"
        df = await fetch_kucoin_candles(symbol, "1h", 100)
        if not df.empty:
            df = add_core_indicators(df)
            regime, _ = bot_v3.regime_detector.detect_regime(df)
            technical_analysis = bot_v3.technical_analyzer.get_advanced_technical_score(df, regime)
            logger.info(f"✅ {symbol} technical analysis ready. Score: {technical_analysis['final_score']:.3f}")
    
    # Initial training for DL models
    logger.info("🧠 Initializing DL models...")
    successful_models = 0
    for sym in TOKENS:
        symbol = sym + "-USDT"
        df = await fetch_kucoin_candles(symbol, "1h", 500)
        if not df.empty:
            df = add_core_indicators(df)
            bot_v3.light_dl.train_model(df, sym)
            successful_models += 1
    
    logger.info(f"✅ Advanced system initialized. DL models: {successful_models}/{len(TOKENS)}")
    logger.info("🔄 Starting main analysis loop with advanced structure...")
    
    while True:
        try:
            tasks = [bot_v3.analyze_symbol(sym + "-USDT") for sym in TOKENS]
            await asyncio.gather(*tasks)
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Quantum Hybrid Bot v3.0 with Advanced Analysis")
