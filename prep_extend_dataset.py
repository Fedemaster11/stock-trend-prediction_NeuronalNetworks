# prep_extend_dataset.py
# Extend data/processed/merged_dataset.csv with new tickers (e.g., TSLA, GOOGL)
# - Robust yfinance fetch (handles MultiIndex / weird headers)
# - Same engineered features + label
# - Sentiment placeholders = 0
# - Append + dedupe (ticker, date)

import os
import argparse
from typing import Optional, List
import numpy as np
import pandas as pd
import yfinance as yf

PROCESSED_PATH = "data/processed/merged_dataset.csv"

# ----------------------------
# Helpers
# ----------------------------
def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, trim, replace spaces with underscores for all columns."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def fetch_prices(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    # Force columns grouped by column (not ticker), avoid MultiIndex
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No price data returned for {ticker}. Check symbol or dates.")

    # If MultiIndex slipped through (older yfinance), collapse to last level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df.reset_index()  # bring Date out of index
    df = _normalize_headers(df)

    # Map possible variants to canonical names
    col_map = {}
    cols = set(df.columns)

    # Date
    if "date" in cols:
        col_map["date"] = "date"
    elif "datetime" in cols:
        df = df.rename(columns={"datetime": "date"})
        col_map["date"] = "date"

    # Price columns (handle 'adj_close', 'adjclose', 'adjusted_close', etc.)
    name_variants = {
        "open": {"open"},
        "high": {"high"},
        "low": {"low"},
        "close": {"close"},
        "adj_close": {"adj_close", "adjclose", "adjusted_close", "adj._close"},
        "volume": {"volume"},
    }

    for canon, variants in name_variants.items():
        found = next((v for v in variants if v in cols), None)
        if found:
            col_map[found] = canon

    df = df.rename(columns=col_map)
    cols = set(df.columns)

    # Fallback: if adj_close missing, use close
    if "adj_close" not in cols and "close" in cols:
        df["adj_close"] = df["close"]

    need = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{ticker}: still missing columns after normalization: {sorted(miss)} | got={list(df.columns)}")

    # Cast numeric
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic drop of rows missing essentials
    df = df.dropna(subset=["date", "open", "high", "low", "close", "adj_close", "volume"]).copy()

    # Add identifiers
    df["ticker"] = ticker
    df["stock_name"] = ticker

    return df[["date", "open", "high", "low", "close", "adj_close", "volume", "stock_name", "ticker"]]

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    # Returns & technicals
    df["ret_1d"]     = df["adj_close"].pct_change()
    df["log_ret_1d"] = np.log1p(df["ret_1d"])
    df["ma_5"]       = df["adj_close"].rolling(5, min_periods=5).mean()
    df["ma_20"]      = df["adj_close"].rolling(20, min_periods=20).mean()
    df["vol_20"]     = df["ret_1d"].rolling(20, min_periods=20).std()

    # Next-day label (avoid alignment issues with .values)
    adj_next = df["adj_close"].shift(-1)
    df["adj_close_t1"] = adj_next
    df["y_up_next"]    = (adj_next.values > df["adj_close"].values).astype(int)

    # Sentiment placeholders (so "Price + Sentiment" runs)
    df["mean_sent"] = 0.0
    df["pos_share"] = 0.0
    df["neg_share"] = 0.0
    df["n_tweets"]  = 0

    # Drop rows where indicators/shift are NA
    need = ["ret_1d","log_ret_1d","ma_5","ma_20","vol_20","adj_close_t1","y_up_next"]
    df = df.dropna(subset=need).copy()
    return df

def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[info] {path} not found. Will create it.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return _normalize_headers(df)

# ----------------------------
# Main
# ----------------------------
def main(tickers: List[str], start: str, end: Optional[str]):
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    base = load_existing(PROCESSED_PATH)

    blocks = []
    for tk in tickers:
        print(f"[+] Fetching & featurizing {tk} ...")
        px = fetch_prices(tk, start, end)
        fx = add_price_features(px)
        fx = _normalize_headers(fx)
        blocks.append(fx)

    new_df = pd.concat(blocks, ignore_index=True)
    new_df["date"] = pd.to_datetime(new_df["date"])

    if base.empty:
        merged = new_df
    else:
        merged = pd.concat([base, new_df], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])

    merged = merged.sort_values(["ticker", "date"])
    merged = merged.drop_duplicates(subset=["ticker", "date"], keep="last")

    merged.to_csv(PROCESSED_PATH, index=False)
    print(f"[done] Saved {len(merged):,} rows to {PROCESSED_PATH}")
    print(merged.groupby("ticker")["date"].agg(["min", "max", "count"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="TSLA,GOOGL", help="Comma-separated tickers to add")
    ap.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default today")
    args = ap.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    main(tickers, args.start, args.end)
