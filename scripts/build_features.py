#!/usr/bin/env python3
"""
Feature engineering for price + sentiment.

Inputs:
  - CSV with at least: date, ticker, open, high, low, close, volume,
    mean_sent, pos_share, neg_share, n_tweets

Outputs:
  - features_enriched.parquet
  - features_enriched.csv

Usage:
  python build_features.py --input merged_dataset_with_sent.csv --output-prefix features_enriched \
      --horizons 1 5 10
"""

import argparse
import sys
from typing import List
import pandas as pd
import numpy as np

# ------------------------ helpers ------------------------

def _ensure_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def add_price_core(df: pd.DataFrame, by=["ticker"]) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(by + ["date"], inplace=True)

    # Basic returns
    df["ret_1d"] = df.groupby(by)["close"].pct_change()
    df["log_ret_1d"] = np.log1p(df["ret_1d"])

    # Rolling price features (past-only, no leakage vs future target)
    for w in (5, 10, 20):
        df[f"ma_close_{w}"] = df.groupby(by)["close"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        df[f"vol_ret_{w}"] = df.groupby(by)["ret_1d"].transform(lambda s: s.rolling(w, min_periods=w).std())
        # Optionally shift by 1 to use info up to t-1 only:
        df[f"ma_close_{w}"] = df.groupby(by)[f"ma_close_{w}"].shift(1)
        df[f"vol_ret_{w}"]  = df.groupby(by)[f"vol_ret_{w}"].shift(1)

    return df

def add_sentiment_features(df: pd.DataFrame, by=["ticker"]) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(by + ["date"], inplace=True)

    # Intensity & polarity balance
    df["sent_strength"] = df["mean_sent"] * df["n_tweets"]
    df["pos_minus_neg"] = df["pos_share"] - df["neg_share"]

    # Rolling windows (centered on past including today), then shift by 1 to avoid peeking when predicting t+1..t+h
    roll_cols = ["mean_sent", "sent_strength", "pos_minus_neg", "n_tweets"]
    for w in (3, 5, 10):
        for col in roll_cols:
            rname = f"{col}_r{w}"
            df[rname] = df.groupby(by)[col].transform(lambda s: s.rolling(w, min_periods=w).mean())
            df[rname] = df.groupby(by)[rname].shift(1)  # past-only

    # Plain lags (t-1) for key sentiment fields
    lag_cols = ["mean_sent", "pos_share", "neg_share", "n_tweets", "sent_strength", "pos_minus_neg"]
    for col in lag_cols:
        df[f"{col}_lag1"] = df.groupby(by)[col].shift(1)

    return df

def add_forward_returns_and_labels(df: pd.DataFrame, horizons: List[int], by=["ticker"]) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(by + ["date"], inplace=True)

    for h in horizons:
        # forward return over h days (exclusive end): (close_{t+h} / close_t - 1)
        df[f"fwd_ret_{h}d"] = df.groupby(by)["close"].transform(lambda s: s.shift(-h) / s - 1.0)
        # binary label: positive forward return
        df[f"y_up_{h}d"] = (df[f"fwd_ret_{h}d"] > 0.0).astype("Int8")

    # Drop rows where any horizon’s forward return is NaN (tail)
    fwd_cols = [f"fwd_ret_{h}d" for h in horizons]
    df = df.dropna(subset=fwd_cols)
    return df

def clean_and_types(df: pd.DataFrame) -> pd.DataFrame:
    # Deduplicate and types
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"]).drop_duplicates(subset=["date", "ticker"])
    df = df.sort_values(["ticker", "date"])
    # coerce numeric columns if needed
    numeric_guess = ["open","high","low","close","volume","mean_sent","pos_share","neg_share","n_tweets"]
    for c in numeric_guess:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to merged_dataset_with_sent.csv")
    ap.add_argument("--output-prefix", default="features_enriched", help="Output file prefix")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10], help="Forward horizons in days")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = clean_and_types(df)

    required = ["date","ticker","open","high","low","close","volume",
                "mean_sent","pos_share","neg_share","n_tweets"]
    _ensure_columns(df, required)

    df = add_price_core(df)
    df = add_sentiment_features(df)
    df = add_forward_returns_and_labels(df, horizons=args.horizons)

    # Drop rows with any newly created NaNs (from rolling/lagging warmups)
    new_feats = [c for c in df.columns if c not in required + ["ret_1d","log_ret_1d"]]
    df = df.dropna(subset=new_feats)

    # Save
    out_parquet = f"{args.output-prefix}.parquet"
    out_csv     = f"{args.output-prefix}.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    # Small print
    print(f"Wrote {len(df):,} rows to:")
    print(f"  - {out_parquet}")
    print(f"  - {out_csv}")
    print("\nColumns added (samples):")
    show = [c for c in df.columns if c.startswith(("ma_close_", "vol_ret_", "sent_strength",
                                                    "pos_minus_neg", "mean_sent_r", "n_tweets_r",
                                                    "pos_minus_neg_r", "fwd_ret_", "y_up_", "lag1"))]
    print(", ".join(show[:20]), "...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        sys.exit(1)
