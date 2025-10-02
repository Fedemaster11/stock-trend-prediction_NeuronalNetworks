#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os # Added os import for clean code structure

IN_PATH  = "data/processed/features_enriched_schema.parquet"
OUT_PATH = "data/processed/features_enriched_schema_mb.parquet"

MARGIN = 0.005     # 0.5%
HORIZONS = [5, 10] # genera y_mb_5d y y_mb_10d

# -----------------------------
# Helper Functions
# -----------------------------
def rsi(series, period=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(series, span):
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    """Calculates Moving Average Convergence Divergence (MACD)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rolling_zscore(x, win=20):
    """Calculates Rolling Z-Score."""
    mu = x.rolling(win).mean()
    sd = x.rolling(win).std()
    return (x - mu) / (sd + 1e-9)

# -----------------------------
# Main Processing
# -----------------------------
print("[INFO] Loading", IN_PATH)
df = pd.read_parquet(IN_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Initial clean and sort
df = df.dropna(subset=["ticker","date","adj_close","volume"]).sort_values(["ticker","date"]).reset_index(drop=True)

# Ensure ret_1d exists (needed for Z-score)
if "ret_1d" not in df.columns:
    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()

# --- 1. Margin Band Labels ---
for h in HORIZONS:
    # Forward return over h days
    fwd = df.groupby("ticker")["adj_close"].shift(-h) / df["adj_close"] - 1.0
    df[f"ret_fwd_{h}d"] = fwd
    
    # Create the label: 1=UP, 0=DOWN, -1=NEUTRAL
    up   = (fwd >= MARGIN).astype(int)
    down = (fwd <= -MARGIN).astype(int)
    lbl  = up.copy()
    lbl[(up == 0) & (down == 0)] = -1  # Mark the neutral band
    df[f"y_mb_{h}d"] = lbl

# --- 2. Lags and Multi-Day Returns ---
for k in range(1, 6):
    df[f"ret_1d_lag{k}"] = df.groupby("ticker")["ret_1d"].shift(k)
df["ret_5d"]      = df.groupby("ticker")["adj_close"].pct_change(5)
df["ret_5d_lag1"] = df.groupby("ticker")["ret_5d"].shift(1)

# --- 3. Z-scores (FIXED: using .transform()) ---
df["close_z20"]  = df.groupby("ticker")["adj_close"].transform(lambda s: rolling_zscore(s, 20))
df["vol_z20"]    = df.groupby("ticker")["volume"].transform(lambda s: rolling_zscore(s, 20))
df["ret1d_z20"]  = df.groupby("ticker")["ret_1d"].transform(lambda s: rolling_zscore(s, 20))

# --- 4. RSI & MACD (FIXED: using .transform() and index reset) ---
df["rsi14"] = df.groupby("ticker")["adj_close"].transform(lambda s: rsi(s, 14))

# MACD is complex because macd() returns 3 series. We use apply() then reset the index.
tmp = df.groupby("ticker")["adj_close"].apply(
    lambda s: pd.DataFrame({"macd": macd(s)[0], "macd_signal": macd(s)[1], "macd_hist": macd(s)[2]})
)
# Reset the level=0 (ticker) of the MultiIndex result and drop it, forcing alignment.
df[["macd","macd_signal","macd_hist"]] = tmp.reset_index(level=0, drop=True)[["macd","macd_signal","macd_hist"]]


# --- Final Cleanup and Save ---
# Drop any rows that may now have NaNs due to rolling windows/lagged features
df = df.dropna().reset_index(drop=True)

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print(f"[INFO] Saved {OUT_PATH} | rows: {len(df)} | cols: {len(df.columns)}")