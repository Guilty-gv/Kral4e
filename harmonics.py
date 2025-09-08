# harmonics.py
# -*- coding: utf-8 -*-
"""
Harmonic patterns detector (Gartley, Bat, Butterfly, Crab)
- find_swings: локални свингови (high/low) со configurable window
- detect_harmonics: проверка на последните N свинг-поинти и враќа list од откриени patterns
- Враќа list на tuples: (pattern_name, side, info_dict)
  e.g. ("GARTLEY", "BUY", {"XA":..., "AB/XA":..., ...})
- Настройки: tolerance и custom ratio ranges за полесно тјунирање.
NOTE: Harmonic правила се approx; треба backtest/визуелна верификација.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

# ---------- Utilities / Defaults ----------
DEFAULT_TOL = 0.05  # ±5% tolerance baseline

# Define ratio ranges per pattern (relative to XA, AB, BC, etc.)
# Each entry: (min, max) as fraction
PATTERN_RANGES = {
    # Gartley (classic approximate ranges)
    "GARTLEY": {
        "AB_XA": (0.57, 0.67),        # ~0.618
        "BC_AB": (0.382, 0.886),
        "CD_XA": (0.778, 0.886),     # CD ~ 0.786 of XA (approx)
    },
    # Bat
    "BAT": {
        "AB_XA": (0.382, 0.5),
        "BC_AB": (0.382, 0.886),
        "CD_XA": (0.886, 0.886),     # CD often ~0.886 of XA
    },
    # Butterfly
    "BUTTERFLY": {
        "AB_XA": (0.67, 0.82),       # AB ~0.786 often
        "BC_AB": (0.382, 0.886),
        "CD_XA": (1.27, 1.618),      # CD is extension beyond X (1.27-1.618)
    },
    # Crab
    "CRAB": {
        "AB_XA": (0.382, 0.618),
        "BC_AB": (0.382, 0.886),
        "CD_XA": (2.0, 3.618),       # deep extension, often ~2.24-3.618
    }
}

# ---------- Swing Finder ----------
def find_swings(df: pd.DataFrame, price_col: str = "close", window: int = 3, lookback: int = 500) -> List[Tuple[str,int,float]]:
    """
    Find local swing highs/lows.
    Returns list of tuples: (type: "H" or "L", index (int), price (float))
    - window: number of candles on each side to consider for local extremum
    - lookback: max number of rows from end to consider (saves compute)
    """
    prices = df[price_col].values
    n = len(prices)
    start = max(0, n - lookback)
    swings = []
    for i in range(start + window, n - window):
        seg = prices[i-window:i+window+1]
        p = prices[i]
        if p == seg.max() and (seg.argmax() == window):
            swings.append(("H", i, float(p)))
        elif p == seg.min() and (seg.argmin() == window):
            swings.append(("L", i, float(p)))
    # compress consecutive same-type swings (keep more extreme)
    cleaned = []
    for s in swings:
        if not cleaned:
            cleaned.append(s)
        else:
            if cleaned[-1][0] != s[0]:
                cleaned.append(s)
            else:
                # same type, keep the more extreme (higher for H, lower for L)
                last = cleaned[-1]
                if s[0] == "H" and s[2] > last[2]:
                    cleaned[-1] = s
                if s[0] == "L" and s[2] < last[2]:
                    cleaned[-1] = s
    return cleaned

# ---------- Ratio helpers ----------
def ratio(a: float, b: float) -> float:
    """Safe ratio a/b (abs), avoid div zero."""
    if b == 0:
        return float('inf')
    return abs(a / b)

def in_range(value: float, min_v: float, max_v: float, tol: float = 0.0) -> bool:
    """Check if value in [min_v*(1-tol), max_v*(1+tol)]"""
    low = min_v * (1 - tol)
    high = max_v * (1 + tol)
    return low <= value <= high

# ---------- Pattern Checks ----------
def check_gartley(X, A, B, C, D, tol=DEFAULT_TOL):
    XA = A - X
    AB = B - A
    BC = C - B
    CD = D - C
    if XA == 0 or AB == 0 or BC == 0:
        return False, {}
    ab_xa = abs(AB) / abs(XA)
    bc_ab = abs(BC) / abs(AB)
    cd_xa = abs(CD) / abs(XA)
    cfg = PATTERN_RANGES["GARTLEY"]
    ok = in_range(ab_xa, cfg["AB_XA"][0], cfg["AB_XA"][1], tol) and \
         in_range(bc_ab, cfg["BC_AB"][0], cfg["BC_AB"][1], tol) and \
         in_range(cd_xa, cfg["CD_XA"][0], cfg["CD_XA"][1], tol)
    return ok, {"AB/XA":ab_xa, "BC/AB":bc_ab, "CD/XA":cd_xa}

def check_bat(X, A, B, C, D, tol=DEFAULT_TOL):
    XA = A - X; AB = B - A; BC = C - B; CD = D - C
    if XA == 0 or AB == 0 or BC == 0:
        return False, {}
    ab_xa = abs(AB)/abs(XA); bc_ab = abs(BC)/abs(AB); cd_xa = abs(CD)/abs(XA)
    cfg = PATTERN_RANGES["BAT"]
    ok = in_range(ab_xa, cfg["AB_XA"][0], cfg["AB_XA"][1], tol) and \
         in_range(bc_ab, cfg["BC_AB"][0], cfg["BC_AB"][1], tol) and \
         in_range(cd_xa, cfg["CD_XA"][0], cfg["CD_XA"][1], tol)
    return ok, {"AB/XA":ab_xa, "BC/AB":bc_ab, "CD/XA":cd_xa}

def check_butterfly(X, A, B, C, D, tol=DEFAULT_TOL):
    XA = A - X; AB = B - A; BC = C - B; CD = D - C
    if XA == 0 or AB == 0 or BC == 0:
        return False, {}
    ab_xa = abs(AB)/abs(XA); bc_ab = abs(BC)/abs(AB); cd_xa = abs(CD)/abs(XA)
    cfg = PATTERN_RANGES["BUTTERFLY"]
    ok = in_range(ab_xa, cfg["AB_XA"][0], cfg["AB_XA"][1], tol) and \
         in_range(bc_ab, cfg["BC_AB"][0], cfg["BC_AB"][1], tol) and \
         in_range(cd_xa, cfg["CD_XA"][0], cfg["CD_XA"][1], tol)
    return ok, {"AB/XA":ab_xa, "BC/AB":bc_ab, "CD/XA":cd_xa}

def check_crab(X, A, B, C, D, tol=DEFAULT_TOL):
    XA = A - X; AB = B - A; BC = C - B; CD = D - C
    if XA == 0 or AB == 0 or BC == 0:
        return False, {}
    ab_xa = abs(AB)/abs(XA); bc_ab = abs(BC)/abs(AB); cd_xa = abs(CD)/abs(XA)
    cfg = PATTERN_RANGES["CRAB"]
    ok = in_range(ab_xa, cfg["AB_XA"][0], cfg["AB_XA"][1], tol) and \
         in_range(bc_ab, cfg["BC_AB"][0], cfg["BC_AB"][1], tol) and \
         in_range(cd_xa, cfg["CD_XA"][0], cfg["CD_XA"][1], tol)
    return ok, {"AB/XA":ab_xa, "BC/AB":bc_ab, "CD/XA":cd_xa}

# ---------- Main detection ----------
def detect_harmonics(df: pd.DataFrame, price_col: str = "close", window: int = 3,
                      lookback: int = 500, min_swings: int = 5, tol: float = DEFAULT_TOL,
                      only_last: bool = True) -> List[Tuple[str,str,Dict[str,Any]]]:
    """
    Detect harmonic patterns in df.
    - Returns list of (pattern_name, side ["BUY"|"SELL"], info)
    - only_last: if True, only check most recent contiguous 5-swings sequences;
                 if False, scan all sequences in swings list.
    """
    out = []
    swings = find_swings(df, price_col=price_col, window=window, lookback=lookback)
    if len(swings) < min_swings:
        return out

    # build list of price-only for convenience
    # swings is list of tuples (type, idx, price)
    L = len(swings)
    indices = range(0, L - 4)
    if only_last:
        indices = [L - 5]

    for i in indices:
        seq = swings[i:i+5]
        # ensure indexes increasing in time
        idxs = [s[1] for s in seq]
        if sorted(idxs) != idxs:
            continue
        types = [s[0] for s in seq]
        # Typical harmonic sequences alternate H/L/H/L/H or L/H/L/H/L depending on start
        # We can accept either, but for direction we deduce from X->A (up or down)
        prices = [s[2] for s in seq]
        X, A, B, C, D = prices

        # Determine side: if X -> A is down (A < X) then pattern is bullish (expect buy at D)
        side = "BUY" if A < X else "SELL"

        # For checks, use oriented absolute differences
        ok, info = check_gartley(X, A, B, C, D, tol=tol)
        if ok:
            out.append(("GARTLEY", side, info))
            # if only_last, return early for speed
            if only_last:
                return out

        ok, info = check_bat(X, A, B, C, D, tol=tol)
        if ok:
            out.append(("BAT", side, info))
            if only_last:
                return out

        ok, info = check_butterfly(X, A, B, C, D, tol=tol)
        if ok:
            out.append(("BUTTERFLY", side, info))
            if only_last:
                return out

        ok, info = check_crab(X, A, B, C, D, tol=tol)
        if ok:
            out.append(("CRAB", side, info))
            if only_last:
                return out

    return out
