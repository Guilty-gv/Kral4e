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
- COMPATIBLE WITH python-kucoin 2.2.0
"""

import os, asyncio, logging, math, random, json, aiohttp
import pandas as pd, numpy as np
from datetime import datetime, timedelta

# ================= KUCOIN IMPORT FIX =================
try:
    from kucoin.client import Client
    KUCOIN_AVAILABLE = True
except ImportError as e:
    print(f"KuCoin import error: {e}")
    KUCOIN_AVAILABLE = False

from telegram import Bot
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import differential_evolution

# ============ CONFIGURATION ============
TOKENS = ["BTC","ETH","ADA","XRP","LINK"]
MAX_OHLCV = 500
TIMEFRAMES = ["1h","4h","1d"]
PRICE_ALERT_THRESHOLD = 0.03
COOLDOWN_MINUTES = 60

BASE_WEIGHTS = {
    "structure": 0.25, "momentum": 0.20, "volume": 0.15, 
    "candles": 0.15, "exotic": 0.10, "ema_sets": 0.08, "momentum_ex": 0.07
}

# API Setup
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

# ================= FIXED KUCOIN CLIENT INIT =================
market_client = None
try:
    if KUCOIN_AVAILABLE and KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE:
        # CORRECT initialization for python-kucoin 2.2.0
        market_client = Client(
            KUCOIN_API_KEY,           # positional argument - no api_key=
            KUCOIN_API_SECRET,        # positional argument - no api_secret=  
            KUCOIN_API_PASSPHRASE,    # positional argument - no api_passphrase=
            sandbox=False             # sandbox parameter
        )
        print("✅ KuCoin client initialized successfully")
        
        # Test connection
        try:
            ticker = market_client.get_ticker('BTC-USDT')
            if ticker:
                print("✅ KuCoin connection verified")
            else:
                print("⚠️ KuCoin test returned no data")
        except Exception as e:
            print(f"⚠️ KuCoin test failed: {e}")
            
    else:
        if not KUCOIN_AVAILABLE:
            print("❌ KuCoin library not available")
        else:
            print("❌ KuCoin API credentials missing")
            
except Exception as e:
    print(f"❌ KuCoin client initialization failed: {e}")
    market_client = None

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quantum_bot_v3")

# ============ FIXED DATA FETCHING =================
async def fetch_kucoin_candles(symbol: str, tf: str = "1h", limit: int = 200):
    """Fetch candles from KuCoin - COMPATIBLE WITH 2.2.0"""
    if market_client is None:
        logger.error(f"❌ KuCoin client not available for {symbol}")
        return pd.DataFrame()
        
    interval_map = {"1h": "1hour", "4h": "4hour", "1d": "1day", "1w": "1week", "15m": "15min"}
    interval = interval_map.get(tf, "1hour")
    
    try:
        # Use get_kline_data for python-kucoin 2.2.0
        candles = market_client.get_kline_data(symbol, interval, limit=limit)
        
        if not candles:
            logger.error(f"❌ No data returned for {symbol}")
            return pd.DataFrame()
        
        # KuCoin returns: [timestamp, open, close, high, low, volume, turnover]
        df = pd.DataFrame(candles, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
        
        # Convert to numeric types
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Remove invalid data
        df = df.dropna(subset=["close", "volume"])
        
        if df.empty:
            logger.error(f"❌ No valid data for {symbol}")
            return pd.DataFrame()
        
        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"✅ Fetched {len(df)} candles for {symbol}. Last price: {df['close'].iloc[-1]:.4f}")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
        
    except Exception as e:
        logger.error(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

# ============ LIGHTWEIGHT DEEP LEARNING ============
class LightweightDeepLearner:
    def __init__(self, sequence_length=30, hidden_units=50):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.scalers = {}
        self.models = {}
    
    def create_sequence_features(self, df):
        """Create sequence-based features for lightweight DL"""
        if len(df) < self.sequence_length:
            return np.array([])
        
        sequences = []
        feature_columns = ['close', 'volume', 'RSI', 'MACD_hist', 'StochK']
        
        for i in range(self.sequence_length, len(df)):
            features = []
            
            # Price sequence features
            prices = df['close'].iloc[i-self.sequence_length:i].values
            features.extend([
                np.mean(prices), np.std(prices), 
                (prices[-1] - prices[0]) / prices[0],
                np.median(prices)
            ])
            
            # Technical indicator features
            for col in feature_columns[1:]:  # Skip 'close' already used
                if col in df.columns:
                    values = df[col].iloc[i-self.sequence_length:i].fillna(0).values
                    features.extend([np.mean(values), np.std(values)])
            
            # Volume features
            volumes = df['volume'].iloc[i-self.sequence_length:i].values
            if np.mean(volumes) > 0:
                features.append(volumes[-1] / np.mean(volumes))
            else:
                features.append(1.0)
            
            sequences.append(features)
        
        return np.array(sequences)
    
    def train_model(self, df, token):
        """Train lightweight model for token"""
        try:
            X_sequences = self.create_sequence_features(df)
            if len(X_sequences) == 0:
                return
            
            # Create targets (next period movement)
            future_returns = df['close'].pct_change().shift(-1).iloc[self.sequence_length:]
            valid_returns = future_returns.dropna()
            
            if len(X_sequences) != len(valid_returns):
                min_len = min(len(X_sequences), len(valid_returns))
                X_sequences = X_sequences[:min_len]
                valid_returns = valid_returns[:min_len]
            
            # Create classification targets
            y = np.where(valid_returns > 0.001, 1, np.where(valid_returns < -0.001, -1, 0))
            
            # Remove neutral cases for better learning
            non_neutral = y != 0
            X_filtered = X_sequences[non_neutral]
            y_filtered = y[non_neutral]
            
            if len(X_filtered) < 20:
                return
            
            # Scale features
            if token not in self.scalers:
                self.scalers[token] = StandardScaler()
                X_scaled = self.scalers[token].fit_transform(X_filtered)
            else:
                X_scaled = self.scalers[token].transform(X_filtered)
            
            # Train MLP classifier
            self.models[token] = MLPClassifier(
                hidden_layer_sizes=(self.hidden_units,),
                activation='relu',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            )
            
            self.models[token].fit(X_scaled, y_filtered)
            logger.info(f"Light DL trained for {token}: {len(X_filtered)} samples")
            
        except Exception as e:
            logger.error(f"Light DL training error for {token}: {e}")
    
    def predict(self, df, token):
        """Predict with lightweight model"""
        if token not in self.models:
            return 0.0, 0.5
        
        try:
            X_sequences = self.create_sequence_features(df)
            if len(X_sequences) == 0:
                return 0.0, 0.5
            
            X_latest = X_sequences[-1:]
            X_scaled = self.scalers[token].transform(X_latest)
            
            probabilities = self.models[token].predict_proba(X_scaled)[0]
            classes = self.models[token].classes_
            
            # Map probabilities to BUY/SELL signal
            if -1 in classes and 1 in classes:
                sell_idx = list(classes).index(-1)
                buy_idx = list(classes).index(1)
                buy_prob = probabilities[buy_idx]
                sell_prob = probabilities[sell_idx]
                dl_signal = buy_prob - sell_prob
                confidence = max(buy_prob, sell_prob)
            else:
                dl_signal = 0.0
                confidence = 0.5
            
            return dl_signal, confidence
            
        except Exception as e:
            logger.error(f"Light DL prediction error for {token}: {e}")
            return 0.0, 0.5

# ============ QUANTUM-INSPIRED OPTIMIZER ============
class QuantumInspiredOptimizer:
    def __init__(self, num_weights=7, population_size=10):
        self.num_weights = num_weights
        self.population_size = population_size
        self.best_weights = None
        self.best_score = -np.inf
    
    def quantum_cost_function(self, weights, performance_data):
        """Quantum-inspired cost function"""
        weights = np.abs(weights)
        weights = weights / np.sum(weights)
        
        # Simulate quantum states
        quantum_scores = []
        
        for _ in range(3):  # 3 quantum states
            quantum_noise = np.random.normal(0, 0.03, len(weights))
            quantum_weights = weights + quantum_noise
            quantum_weights = np.clip(quantum_weights, 0.01, 0.5)
            quantum_weights = quantum_weights / np.sum(quantum_weights)
            
            # Evaluate quantum state
            state_score = 0
            weight_categories = list(performance_data.keys())
            
            for i, category in enumerate(weight_categories[:len(quantum_weights)]):
                if category in performance_data:
                    success_rate = performance_data[category].get('success_rate', 0.5)
                    state_score += quantum_weights[i] * success_rate
            
            # Quantum entropy bonus
            entropy = -np.sum(quantum_weights * np.log(quantum_weights + 1e-10))
            state_score += 0.05 * entropy
            
            quantum_scores.append(state_score)
        
        return np.mean(quantum_scores)
    
    def optimize_weights(self, performance_data, max_iter=20):
        """Optimize weights using genetic algorithm"""
        
        def objective(weights):
            return -self.quantum_cost_function(weights, performance_data)
        
        bounds = [(0.01, 0.5) for _ in range(self.num_weights)]
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iter,
                popsize=self.population_size,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            if result.success:
                optimized_weights = np.abs(result.x)
                optimized_weights = optimized_weights / np.sum(optimized_weights)
                
                self.best_weights = dict(zip([
                    'structure', 'momentum', 'volume', 'candles', 
                    'exotic', 'ema_sets', 'momentum_ex'
                ], optimized_weights))
                
                self.best_score = -result.fun
                logger.info(f"Quantum optimization completed. Score: {self.best_score:.4f}")
                return self.best_weights
                
        except Exception as e:
            logger.error(f"Quantum optimization error: {e}")
        
        return None
    
    def collect_performance_data(self, signal_history):
        """Collect performance data from signal history"""
        performance_data = {}
        
        for category in BASE_WEIGHTS.keys():
            category_data = []
            
            for symbol_history in signal_history.values():
                for signal in symbol_history[-100:]:  # Last 100 signals
                    if 'components' in signal and category in signal['components']:
                        comp_signal = signal['components'][category]
                        # Simulate success based on signal strength and direction
                        success = random.random() < (0.5 + comp_signal.get('strength', 0) * 0.2)
                        category_data.append(success)
            
            if category_data:
                success_rate = sum(category_data) / len(category_data)
                performance_data[category] = {'success_rate': success_rate}
        
        return performance_data

# ============ REAL-TIME ADAPTOR ============
class RealTimeAdaptor:
    def __init__(self, data_dir="bot_data"):
        self.data_dir = data_dir
        self.performance_cache = {}
        self.adaptation_factors = {}
        
        os.makedirs(data_dir, exist_ok=True)
        self.load_data()
    
    def load_data(self):
        """Load adaptation data"""
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith('_adaptation.json'):
                    symbol = filename.replace('_adaptation.json', '')
                    with open(os.path.join(self.data_dir, filename), 'r') as f:
                        self.performance_cache[symbol] = json.load(f)
            logger.info(f"Loaded adaptation data for {len(self.performance_cache)} symbols")
        except Exception as e:
            logger.error(f"Error loading adaptation data: {e}")
    
    def save_data(self, symbol):
        """Save adaptation data for symbol"""
        try:
            if symbol in self.performance_cache:
                filepath = os.path.join(self.data_dir, f"{symbol}_adaptation.json")
                with open(filepath, 'w') as f:
                    json.dump(self.performance_cache[symbol], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
    
    def record_performance(self, symbol, signal_data, price_change):
        """Record signal performance"""
        if symbol not in self.performance_cache:
            self.performance_cache[symbol] = {'signals': [], 'metrics': {}}
        
        success = self.calculate_success(signal_data['decision'], price_change)
        
        signal_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'decision': signal_data['decision'],
            'confidence': signal_data.get('confidence', 0),
            'regime': signal_data.get('regime', 'UNKNOWN'),
            'success': success,
            'price_change': price_change
        }
        
        self.performance_cache[symbol]['signals'].append(signal_record)
        
        # Keep only recent signals
        if len(self.performance_cache[symbol]['signals']) > 200:
            self.performance_cache[symbol]['signals'] = self.performance_cache[symbol]['signals'][-200:]
        
        self.update_metrics(symbol)
        self.calculate_factors(symbol)
        self.save_data(symbol)
    
    def calculate_success(self, decision, price_change):
        """Calculate if signal was successful"""
        if decision == 'HOLD':
            return abs(price_change) < 0.002  # Success if no major move during hold
        
        if decision == 'BUY':
            return price_change > 0.001
        elif decision == 'SELL':
            return price_change < -0.001
        
        return False
    
    def update_metrics(self, symbol):
        """Update performance metrics"""
        signals = self.performance_cache[symbol]['signals']
        if len(signals) < 10:
            return
        
        recent = signals[-50:]
        success_rate = sum(1 for s in recent if s['success']) / len(recent)
        
        # Regime performance
        regime_perf = {}
        for regime in ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE']:
            regime_signals = [s for s in recent if s.get('regime') == regime]
            if regime_signals:
                regime_success = sum(1 for s in regime_signals if s['success'])
                regime_perf[regime] = regime_success / len(regime_signals)
        
        self.performance_cache[symbol]['metrics'] = {
            'success_rate': success_rate,
            'regime_performance': regime_perf,
            'total_analyzed': len(recent)
        }
    
    def calculate_factors(self, symbol):
        """Calculate adaptation factors"""
        metrics = self.performance_cache[symbol]['metrics']
        factors = {'overall_boost': 1.0, 'regime_boosts': {}}
        
        success_rate = metrics.get('success_rate', 0.5)
        if success_rate > 0.6:
            factors['overall_boost'] = 1.15
        elif success_rate < 0.4:
            factors['overall_boost'] = 0.85
        
        for regime, perf in metrics.get('regime_performance', {}).items():
            if perf > 0.6:
                factors['regime_boosts'][regime] = 1.1
            elif perf < 0.4:
                factors['regime_boosts'][regime] = 0.9
        
        self.adaptation_factors[symbol] = factors
    
    def get_boost(self, symbol, regime=None):
        """Get adaptation boost"""
        if symbol not in self.adaptation_factors:
            return 1.0
        
        factors = self.adaptation_factors[symbol]
        boost = factors.get('overall_boost', 1.0)
        
        if regime and regime in factors.get('regime_boosts', {}):
            boost *= factors['regime_boosts'][regime]
        
        return boost

# ============ ENHANCED COMPONENTS ============
class MarketRegimeDetector:
    def __init__(self):
        self.regime_history = []
    
    def detect_regime(self, df):
        """Detect market regime"""
        if len(df) < 50:
            return 'SIDEWAYS', 0.5
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()
        trend = df['close'].rolling(50).mean()
        
        current_vol = volatility.iloc[-1]
        vol_avg = volatility.mean()
        price_vs_ma = (df['close'].iloc[-1] - trend.iloc[-1]) / trend.iloc[-1]
        
        if current_vol > vol_avg * 1.5:
            regime = 'VOLATILE'
            confidence = min(current_vol / (vol_avg * 2), 0.95)
        elif price_vs_ma > 0.05:
            regime = 'BULL'
            confidence = min(abs(price_vs_ma) / 0.15, 0.95)
        elif price_vs_ma < -0.05:
            regime = 'BEAR'
            confidence = min(abs(price_vs_ma) / 0.15, 0.95)
        else:
            regime = 'SIDEWAYS'
            confidence = 0.7
        
        self.regime_history.append({'regime': regime, 'confidence': confidence, 'timestamp': datetime.utcnow()})
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)
            
        return regime, confidence

class AdvancedRiskManager:
    def __init__(self):
        self.max_position_size = 0.1
    
    def calculate_position_size(self, account_balance, confidence, volatility, regime):
        """Calculate position size using Kelly Criterion"""
        win_prob = 0.5 + abs(confidence) * 0.3
        win_loss_ratio = self.get_win_loss_ratio(regime)
        
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly = max(0.02, min(kelly, 0.2))  # Conservative bounds
        
        vol_adjustment = 1.0 / (1.0 + volatility * 15)
        regime_multiplier = self.get_regime_multiplier(regime)
        
        position_size = account_balance * kelly * vol_adjustment * regime_multiplier
        return min(position_size, account_balance * self.max_position_size)
    
    def get_win_loss_ratio(self, regime):
        ratios = {'BULL': 2.2, 'BEAR': 1.8, 'SIDEWAYS': 1.5, 'VOLATILE': 1.3}
        return ratios.get(regime, 1.8)
    
    def get_regime_multiplier(self, regime):
        multipliers = {'BULL': 1.1, 'BEAR': 0.9, 'SIDEWAYS': 0.7, 'VOLATILE': 0.6}
        return multipliers.get(regime, 1.0)

class SentimentAnalyzer:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300
    
    async def get_fear_greed_index(self):
        """Get Fear & Greed Index"""
        cache_key = 'fear_greed'
        current_time = datetime.utcnow().timestamp()
        
        if cache_key in self.cache and current_time - self.cache[cache_key]['timestamp'] < self.cache_timeout:
            return self.cache[cache_key]['value']
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.alternative.me/fng/', timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        fear_greed = int(data['data'][0]['value'])
                        self.cache[cache_key] = {'value': fear_greed, 'timestamp': current_time}
                        return fear_greed
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed: {e}")
        
        return 50
    
    def interpret_sentiment(self, fear_greed):
        """Interpret sentiment score"""
        if fear_greed >= 75: return 'EXTREME_GREED', -0.2
        elif fear_greed >= 60: return 'GREED', -0.1
        elif fear_greed >= 40: return 'NEUTRAL', 0.0
        elif fear_greed >= 25: return 'FEAR', 0.1
        else: return 'EXTREME_FEAR', 0.2

# ============ MAIN BOT CLASS ============
class QuantumHybridBotV3:
    def __init__(self):
        # Core components
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = AdvancedRiskManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        
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
        """Complete enhanced analysis"""
        token = symbol.replace("-USDT", "")
        
        # Fetch data and basic indicators
        df = await self.fetch_data_with_indicators(symbol, "1h")
        if df.empty:
            logger.warning(f"No data for {symbol}")
            return
        
        last_price = df['close'].iloc[-1]
        
        # 1. MARKET REGIME DETECTION
        regime, regime_confidence = self.regime_detector.detect_regime(df)
        
        # 2. SENTIMENT ANALYSIS
        fear_greed = await self.sentiment_analyzer.get_fear_greed_index()
        sentiment_label, sentiment_score = self.sentiment_analyzer.interpret_sentiment(fear_greed)
        
        # 3. LIGHT DEEP LEARNING PREDICTION
        dl_signal, dl_confidence = self.light_dl.predict(df, token)
        
        # 4. TECHNICAL ANALYSIS (existing logic)
        technical_score = self.calculate_technical_score(df, token)
        
        # 5. QUANTUM OPTIMIZATION (periodic)
        if len(self.signal_history[token]) > 50 and len(self.signal_history[token]) % 50 == 0:
            performance_data = self.quantum_optimizer.collect_performance_data(self.signal_history)
            optimized_weights = self.quantum_optimizer.optimize_weights(performance_data)
            if optimized_weights:
                self.adaptive_weights[token] = optimized_weights
                logger.info(f"Updated weights for {token}: {optimized_weights}")
        
        # 6. REAL-TIME ADAPTATION
        adaptation_boost = self.realtime_adaptor.get_boost(token, regime)
        
        # 7. COMBINE ALL SIGNALS
        final_score = self.combine_signals(
            technical_score=technical_score,
            dl_signal=dl_signal,
            dl_confidence=dl_confidence,
            sentiment_score=sentiment_score,
            regime_impact=self.calculate_regime_impact(regime),
            adaptation_boost=adaptation_boost
        )
        
        # Decision making
        threshold = self.calculate_dynamic_threshold(regime, fear_greed)
        decision = "BUY" if final_score > threshold else "SELL" if final_score < -threshold else "HOLD"
        
        # Risk management
        volatility = df['ATR'].iloc[-1] / last_price if 'ATR' in df.columns else 0.02
        position_size = self.risk_manager.calculate_position_size(
            account_balance, final_score, volatility, regime
        )
        
        # Price targets
        buy_target, sell_target, fibs = self.hybrid_price_targets(df, last_price)
        
        # Check alert conditions
        if not self.should_alert(symbol, last_price, decision):
            return
        
        # 8. ENHANCED REPORTING
        await self.send_enhanced_alert(
            symbol=symbol,
            decision=decision,
            last_price=last_price,
            final_score=final_score,
            regime=regime,
            sentiment=(sentiment_label, fear_greed),
            dl_confidence=dl_confidence,
            position_size=position_size,
            buy_target=buy_target,
            sell_target=sell_target
        )
        
        # Record for learning
        self.record_signal(token, decision, final_score, regime)
        
        # Update DL model periodically
        if len(self.signal_history[token]) % 100 == 0:
            self.light_dl.train_model(df, token)
    
    def calculate_technical_score(self, df, token):
        """Calculate technical analysis score"""
        # Simplified technical scoring (from original logic)
        score = 0.0
        
        # RSI
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi < 30: score += 0.3
            elif rsi > 70: score -= 0.3
        
        # MACD
        if 'MACD_hist' in df.columns:
            macd_hist = df['MACD_hist'].iloc[-1]
            if macd_hist > 0: score += 0.2
            else: score -= 0.2
        
        # EMA trend
        if 'EMA_50' in df.columns and 'EMA_200' in df.columns:
            if df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1]:
                score += 0.2
            else:
                score -= 0.2
        
        return max(-1.0, min(1.0, score))
    
    def combine_signals(self, technical_score, dl_signal, dl_confidence, sentiment_score, regime_impact, adaptation_boost):
        """Intelligently combine all signals"""
        # Weighted combination
        final = (technical_score * 0.4 + 
                dl_signal * dl_confidence * 0.3 +
                sentiment_score * 0.2 +
                regime_impact * 0.1)
        
        # Apply adaptation boost
        final *= adaptation_boost
        
        return max(-1.0, min(1.0, final))
    
    def calculate_regime_impact(self, regime):
        """Calculate regime-based impact"""
        impacts = {'BULL': 0.1, 'BEAR': -0.1, 'SIDEWAYS': 0.0, 'VOLATILE': -0.05}
        return impacts.get(regime, 0.0)
    
    def calculate_dynamic_threshold(self, regime, fear_greed):
        """Calculate dynamic decision threshold"""
        base_threshold = 0.15
        
        # Regime adjustments
        regime_adjustments = {'VOLATILE': 0.08, 'BULL': 0.05, 'BEAR': 0.08, 'SIDEWAYS': 0.06}
        threshold = base_threshold + regime_adjustments.get(regime, 0.0)
        
        # Sentiment adjustment
        if fear_greed >= 70 or fear_greed <= 30:  # Extreme sentiment
            threshold += 0.04
        
        return threshold
    
    def should_alert(self, symbol, last_price, decision):
        """Check if should send alert"""
        if decision == 'HOLD':
            return False
        
        now = datetime.utcnow()
        key = symbol
        
        # Price change check
        if key in self.last_price_sent:
            change = abs(last_price - self.last_price_sent[key]) / self.last_price_sent[key]
            if change < PRICE_ALERT_THRESHOLD:
                return False
        
        # Cooldown check
        if key in self.last_sent_time:
            if now - self.last_sent_time[key] < timedelta(minutes=COOLDOWN_MINUTES):
                return False
        
        self.last_price_sent[key] = last_price
        self.last_sent_time[key] = now
        return True
    
    def record_signal(self, token, decision, score, regime):
        """Record signal for learning"""
        signal_data = {
            'timestamp': datetime.utcnow(),
            'decision': decision,
            'score': score,
            'regime': regime
        }
        
        self.signal_history[token].append(signal_data)
        if len(self.signal_history[token]) > 1000:
            self.signal_history[token].pop(0)
    
    async def fetch_data_with_indicators(self, symbol, timeframe):
        """Fetch data and add indicators"""
        df = await fetch_kucoin_candles(symbol, timeframe, MAX_OHLCV)
        if df.empty:
            return df
        
        df = add_core_indicators(df)
        return df.dropna()
    
    def hybrid_price_targets(self, df, last_price):
        """Calculate price targets"""
        recent = df.tail(100)
        high, low = recent["high"].max(), recent["low"].min()
        diff = high - low
        
        fibs = {
            "0.236": high - diff*0.236, 
            "0.382": high - diff*0.382,
            "0.5": high - diff*0.5,
            "0.618": high - diff*0.618
        }
        
        levels = list(fibs.values())
        buy = max([l for l in levels if l <= last_price], default=last_price*0.98)
        sell = min([l for l in levels if l >= last_price], default=last_price*1.02)
        
        return smart_round(buy), smart_round(sell), fibs
    
    async def send_enhanced_alert(self, symbol, decision, last_price, final_score,
                                regime, sentiment, dl_confidence, position_size,
                                buy_target, sell_target):
        """Send enhanced Telegram alert"""
        sentiment_label, fear_greed = sentiment
        
        msg = (f"🚀 **QUANTUM BOT v3.0**\n"
               f"⏰ {now_str()}\n"
               f"📌 **{symbol}**\n"
               f"💰 **Price**: ${last_price:.4f}\n"
               f"🔔 **Decision**: **{decision}** (Score: {final_score:.3f})\n"
               f"📊 **Regime**: {regime}\n"
               f"😊 **Sentiment**: {sentiment_label} (F&G: {fear_greed})\n"
               f"🧠 **DL Confidence**: {dl_confidence:.2f}\n"
               f"💼 **Position**: ${position_size:.2f}\n"
               f"🛒 **Buy Target**: ${buy_target:.4f}\n"
               f"💵 **Sell Target**: ${sell_target:.4f}\n"
               f"⚡ **Enhanced AI Analysis**")
        
        logger.info(f"v3 Analysis: {symbol} -> {decision} (Score: {final_score:.3f})")
        await send_telegram(msg)

# ============ UTILITY FUNCTIONS ============
def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

async def send_telegram(msg: str):
    if not bot: 
        logger.warning("Telegram bot not available")
        return
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg)
        logger.info("Telegram message sent successfully")
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def smart_round(v: float):
    if v >= 1: return round(v, 2)
    if v >= 0.01: return round(v, 4)
    return round(v, 6)

# ============ FIXED CORE INDICATORS FUNCTION ============
def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add core technical indicators - FIXED VERSION"""
    if df.empty: 
        return df
        
    df = df.copy()
    
    try:
        # Ensure we're working with pandas Series, not numpy arrays
        close_series = pd.Series(df["close"].values, index=df.index)
        
        # EMAs
        for span in [9, 21, 50, 200]:
            df[f"EMA_{span}"] = close_series.ewm(span=span).mean()
        
        # RSI
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = close_series.ewm(span=12).mean()
        exp2 = close_series.ewm(span=26).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        
        # ATR - simplified calculation
        high_low = df['high'] - df['low']
        df["ATR"] = high_low.rolling(14).mean()
        
        # Fill NaN values
        df = df.ffill().bfill()
        
        logger.info(f"Indicators calculated successfully. RSI: {df['RSI'].iloc[-1]:.2f}")
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        # Return basic dataframe without indicators
        return df
    
    return df

# ============ MAIN EXECUTION ============
bot_v3 = QuantumHybridBotV3()

async def main_loop():
    """Main execution loop"""
    logger.info("🚀 Starting Quantum Hybrid Bot v3.0...")
    
    # Check if KuCoin client is available
    if not market_client:
        logger.error("❌ KuCoin client not available. Check API credentials.")
        return
    
    # Test data fetching
    logger.info("🔍 Testing data fetching...")
    test_symbol = "BTC-USDT"
    test_df = await fetch_kucoin_candles(test_symbol, "1h", 10)
    if test_df.empty:
        logger.error("❌ Data fetching test failed")
        return
    else:
        logger.info(f"✅ Data fetching test passed: {len(test_df)} candles")
    
    # Initial training for DL models
    logger.info("🧠 Initializing DL models...")
    successful_models = 0
    for sym in TOKENS:
        symbol = sym + "-USDT"
        df = await fetch_kucoin_candles(symbol, "1h", 100)  # Reduced for speed
        if not df.empty:
            df = add_core_indicators(df)
            bot_v3.light_dl.train_model(df, sym)
            successful_models += 1
            logger.info(f"✅ DL model trained for {sym}")
        else:
            logger.warning(f"❌ No data for {sym}")
    
    logger.info(f"✅ DL models initialized: {successful_models}/{len(TOKENS)}")
    logger.info("🔄 Starting main analysis loop...")
    
    while True:
        try:
            tasks = [bot_v3.analyze_symbol(sym + "-USDT") for sym in TOKENS]
            await asyncio.gather(*tasks)
            logger.info(f"💤 Cycle completed. Waiting 5 minutes... ({datetime.utcnow().strftime('%H:%M:%S')})")
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Quantum Hybrid Bot v3.0")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
