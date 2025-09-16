# -*- coding: utf-8 -*-
"""
Production-Ready Hybrid Crypto Bot
- Multi-Timeframe Weighted Voting: 1h, 4h, 1d
- Adaptive Weighted Voting + Fib/Harmonics + ATR targets
- Dynamic thresholds based on ATR
- ML-based prediction (RandomForest, optional)
- Historic success evaluation per weight category
- Telegram alerts with confidence and cooldown
"""

# [Imports and config identical to previous version]

TIMEFRAME_WEIGHTS = {"1h": 0.2, "4h": 0.3, "1d": 0.5}  # relative influence on final score
# --- Test mode flag ---
TEST_TELEGRAM_MODE = True  # Ако е True, секогаш ќе праќа Telegram порака

# ================= WEIGHTED VOTING (MULTI-TIMEFRAME) =================
def multi_timeframe_voting(dfs: dict, token: str) -> tuple[str, float]:
    total_score = 0
    for tf, df in dfs.items():
        decision, score = weighted_voting_signals(df, token)
        weight = TIMEFRAME_WEIGHTS.get(tf, 0.33)
        total_score += score * weight
    # Final decision based on combined weighted score
    last_close = dfs["1d"]["close"].iloc[-1]
    atr = dfs["1d"]["ATR"].iloc[-1] if "ATR" in dfs["1d"].columns else 0.01
    dynamic_threshold = max(0.2, atr/last_close)
    final_decision = "BUY" if total_score>dynamic_threshold else "SELL" if total_score<-dynamic_threshold else "HOLD"
    return final_decision, total_score

# ================= ANALYZE SYMBOL =================
async def analyze_symbol(symbol: str):
    token = symbol.replace("-USDT","")
    dfs = {}
    for tf in TIMEFRAME_WEIGHTS.keys():
        df = await fetch_kucoin_candles(symbol, tf, MAX_OHLCV)
        if df.empty: return
        dfs[tf] = add_indicators(df).dropna()
    last_price = dfs["1d"]["close"].iloc[-1]

    decision, score = multi_timeframe_voting(dfs, token)
    buy, sell, fibs = hybrid_price_targets(dfs["1d"], last_price)

    key = symbol
    now = datetime.utcnow()
    
    # === Test mode override ===
    send_alert = TEST_TELEGRAM_MODE or (
        abs(last_price - last_price_sent.get(key,last_price))/max(last_price_sent.get(key,last_price),1e-9) >= PRICE_ALERT_THRESHOLD
        and (key not in last_sent_time or now - last_sent_time[key] >= timedelta(minutes=COOLDOWN_MINUTES))
    )
    if not send_alert:
        return

    last_price_sent[key] = last_price
    last_sent_time[key] = now

    msg = (f"Strategy: Multi-Timeframe Hybrid Bot\n⏰ {now_str()}\n📊 {symbol}\n"
           f"💰 Last Price: {last_price}\n✅ Decision: {decision} (Score: {round(score,2)})\n"
           f"🛒 Buy: {buy} | 💵 Sell: {sell}\n📊 Fib Levels: {fibs}\n⚖️ Adaptive Weights: {adaptive_weights[token]}")
    logger.info(msg)
    asyncio.create_task(send_telegram(msg))
    update_adaptive_weights(token, decision, last_price)

# [Main loop identical, continuous_monitor calls analyze_symbol]
