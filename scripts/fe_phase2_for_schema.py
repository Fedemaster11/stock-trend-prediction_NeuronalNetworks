#!/usr/bin/env python3
"""
Phase 2 Feature Engineering (schema-aware).

Input CSV schema (as provided):
  date, open, high, low, close, adj_close, volume, stock_name, ticker,
  log_ret_1d, ret_1d, ma_5, ma_20, vol_20, adj_close_t1, y_up_next,
  mean_sent, n_tweets, pos_share, neg_share

What this script ADDS (it DOES NOT recompute your existing features):
  - Sentiment intensity & polarity:
      sent_strength = mean_sent * n_tweets
      pos_minus_neg = pos_share - neg_share
  - Rolling sentiment means (past-only, no leakage):
      mean_sent_r{3,5,10}, sent_strength_r{3,5,10},
      pos_minus_neg_r{3,5,10}, n_tweets_r{3,5,10}
  - Lagged (t-1) sentiment fields:
      *_lag1 for mean_sent, n_tweets, pos_share, neg_share,
      sent_strength, pos_minus_neg
  - Forward returns/labels over horizons H ∈ {1,5,10} (by default):
      fwd_ret_{H}d  (based on adj_close)
      y_up_{H}d     = 1{fwd_ret_{H}d > 0}
    Note: If y_up_next exists, we alias it to y_up_1d (for consistency).

Outputs:
  - features_enriched_schema.parquet
  - features_enriched_schema.csv

Usage:
  python fe_phase2_for_schema.py \
      --input merged_dataset_with_sent.csv \
      --out-prefix features_enriched_schema \
      --horizons 1 5 10
"""

import argparse
from typing import List
import numpy as np
import pandas as pd

REQ = [
    "date","open","high","low","close","adj_close","volume",
    "stock_name","ticker","log_ret_1d","ret_1d","ma_5","ma_20","vol_20",
    "adj_close_t1","y_up_next",
    "mean_sent","n_tweets","pos_share","neg_share"
]

def _ensure_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # dates
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    # numerics
    num_cols = [
        "open","high","low","close","adj_close","volume",
        "log_ret_1d","ret_1d","ma_5","ma_20","vol_20","adj_close_t1",
        "mean_sent","n_tweets","pos_share","neg_share"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # clean
    df = df.dropna(subset=["date","ticker","adj_close"]).drop_duplicates(subset=["date","ticker"])
    return df.sort_values(["ticker","date"])

def add_sentiment_core(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic composites
    df["sent_strength"] = df["mean_sent"] * df["n_tweets"]
    df["pos_minus_neg"] = df["pos_share"] - df["neg_share"]
    return df

def add_sentiment_rolls_and_lags(df: pd.DataFrame, by=["ticker"]) -> pd.DataFrame:
    df = df.copy()

    # Rolling means over past windows; shift(1) to avoid leakage
    roll_cols = ["mean_sent", "sent_strength", "pos_minus_neg", "n_tweets"]
    for w in (3, 5, 10):
        for col in roll_cols:
            rname = f"{col}_r{w}"
            df[rname] = (
                df.groupby(by)[col]
                  .transform(lambda s: s.rolling(window=w, min_periods=w).mean())
            )
            df[rname] = df.groupby(by)[rname].shift(1)  # only info up to t-1

    # Plain lags (t-1)
    for col in ["mean_sent","n_tweets","pos_share","neg_share","sent_strength","pos_minus_neg"]:
        df[f"{col}_lag1"] = df.groupby(by)[col].shift(1)

    return df

def add_forward_returns_and_labels(df: pd.DataFrame, horizons: List[int], by=["ticker"]) -> pd.DataFrame:
    """
    Forward returns computed from adj_close to be split/Dividend-safe.
    fwd_ret_{h}d = adj_close_{t+h}/adj_close_t - 1
    y_up_{h}d    = 1{fwd_ret_{h}d > 0}
    """
    df = df.copy()
    for h in horizons:
        df[f"fwd_ret_{h}d"] = df.groupby(by)["adj_close"].transform(lambda s: s.shift(-h) / s - 1.0)
        df[f"y_up_{h}d"] = (df[f"fwd_ret_{h}d"] > 0.0).astype("Int8")
    # drop tail rows missing forward returns
    df = df.dropna(subset=[f"fwd_ret_{h}d" for h in horizons])
    return df

def alias_y_up_1d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep your original y_up_next and create a consistent alias y_up_1d if not present.
    """
    df = df.copy()
    if "y_up_1d" not in df.columns and "y_up_next" in df.columns:
        # Coerce to 0/1 Int8
        tmp = pd.to_numeric(df["y_up_next"], errors="coerce")
        df["y_up_1d"] = (tmp > 0).astype("Int8") if tmp.notna().any() else df["y_up_next"].astype("Int8")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to merged CSV with your schema")
    ap.add_argument("--out-prefix", default="features_enriched_schema", help="Output file prefix")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1,5,10], help="Forward horizons in days")
    args = ap.parse_args()

    # Load & validate
    df = pd.read_csv(args.input)
    _ensure_columns(df, REQ)
    df = _coerce_types(df)

    # Respect existing features; only ADD new ones
    df = add_sentiment_core(df)
    df = add_sentiment_rolls_and_lags(df, by=["ticker"])
    df = alias_y_up_1d(df)  # keep y_up_next; also provide y_up_1d
    df = add_forward_returns_and_labels(df, horizons=args.horizons, by=["ticker"])

    # Drop rows that have NaNs stemming from rolling warmups/lagging
    new_candidates = [
        "sent_strength","pos_minus_neg",
        "mean_sent_r3","mean_sent_r5","mean_sent_r10",
        "sent_strength_r3","sent_strength_r5","sent_strength_r10",
        "pos_minus_neg_r3","pos_minus_neg_r5","pos_minus_neg_r10",
        "n_tweets_r3","n_tweets_r5","n_tweets_r10",
        "mean_sent_lag1","n_tweets_lag1","pos_share_lag1","neg_share_lag1",
        "sent_strength_lag1","pos_minus_neg_lag1"
    ] + [f"fwd_ret_{h}d" for h in args.horizons] + [f"y_up_{h}d" for h in args.horizons]

    # Only drop if the columns exist (robustness if some windows/horizons are customized)
    present = [c for c in new_candidates if c in df.columns]
    if present:
        df = df.dropna(subset=present)

    # Save
    out_parquet = f"{args.out_prefix}.parquet"
    out_csv     = f"{args.out_prefix}.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    # Console summary
    added_cols = [c for c in df.columns if c.endswith(("_lag1")) or "_r" in c or c.startswith(("sent_strength","pos_minus_neg","fwd_ret_","y_up_"))]
    print(f"Wrote {len(df):,} rows:\n  - {out_parquet}\n  - {out_csv}")
    print("New columns (sample):", sorted(added_cols)[:20])

if __name__ == "__main__":
    main()
