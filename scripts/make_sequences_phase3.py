#!/usr/bin/env python3
"""
Make rolling sequences for sequence models (LSTM/BiLSTM/1D CNN).

Input:
  - Enriched parquet/csv from Phase 2 (e.g., features_enriched_schema.parquet)
  - If you created margin-band labels and extra features, use the new file
    (e.g., features_enriched_schema_mb.parquet)

Output:
  - NPZ with X_train/val/test, y_train/val/test, feature_names, tickers

Notes:
  - Per-ticker temporal split (70/15/15).
  - Standardize features on TRAIN only (per ticker).
  - Windows without leakage: features up to t-1 -> label at t.
  - Supports horizons: 1, 5, 10 (choose with --horizon).
  - NEW:
      * --tickers MSFT TSLA (optional) to restrict tickers
      * --label {sign, marginband}; marginband drops neutral rows (-1)
      * extra features (lags, z-scores, RSI, MACD) auto-included if present
"""

import argparse
import os
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

RANDOM_STATE = 42

# --------------------------
# Base price features
# --------------------------
PRICE_COLS_BASE = [
    "ret_1d","log_ret_1d","ma_5","ma_20","vol_20","adj_close_t1",
    "open","high","low","close","adj_close","volume"
]

# --------------------------
# Extra technical features (added in your Phase-2 MB script)
# --------------------------
EXTRA_COLS = [
    "ret_1d_lag1","ret_1d_lag2","ret_1d_lag3","ret_1d_lag4","ret_1d_lag5",
    "ret_5d_lag1",
    "close_z20","vol_z20","ret1d_z20",
    "rsi14","macd","macd_signal","macd_hist",
]

# --------------------------
# Sentiment-related prefixes (only if --use-sentiment)
# --------------------------
SENT_PREFIXES = (
    "mean_sent","n_tweets","pos_share","neg_share",
    "sent_strength","pos_minus_neg",
    "mean_sent_r","n_tweets_r","sent_strength_r","pos_minus_neg_r"
)

def pick_features(df, use_sentiment: bool):
    cols = [c for c in PRICE_COLS_BASE if c in df.columns]
    # optional sentiment features
    if use_sentiment:
        for c in df.columns:
            if c.startswith(SENT_PREFIXES) or c.endswith("_lag1"):
                cols.append(c)
    # extra technical features
    cols.extend([c for c in EXTRA_COLS if c in df.columns])
    # de-dup keeping order & existing only
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    return cols

def temporal_split_idx(n, test_frac=0.15, val_frac=0.15):
    n_test = int(np.floor(test_frac * n))
    n_val = int(np.floor(val_frac * (n - n_test)))
    n_train = n - n_val - n_test
    return (
        np.arange(0, n_train),
        np.arange(n_train, n_train + n_val),
        np.arange(n_train + n_val, n),
    )

def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def build_windows(arr2d, labels, seq_len):
    """
    arr2d: (T, F), labels: (T,)
    returns: X: (N, seq_len, F), y: (N,)
    Uses windows ending at t-1 to predict label at t.
    """
    X, y = [], []
    for t in range(seq_len, len(arr2d)):
        X.append(arr2d[t - seq_len : t, :])
        y.append(labels[t])
    if not X:
        return (
            np.zeros((0, seq_len, arr2d.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=int),
        )
    return np.stack(X), np.array(y, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/features_enriched_schema.parquet",
                    help="Path to enriched dataset (parquet/csv). For margin-band use the *_mb.parquet file.")
    ap.add_argument("--out", default="data/processed/seq_MSFT_GOOGL_TSLA_h{h}_L{L}.npz",
                    help="Output NPZ path (supports {h} for horizon and {L} for seq_len)")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10],
                    help="Target horizon in days")
    ap.add_argument("--seq-len", type=int, default=30, dest="seq_len",
                    help="Sequence length (days) for rolling windows")
    ap.add_argument("--use-sentiment", action="store_true",
                    help="Include sentiment features (mainly useful for MSFT)")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Optional list of tickers to include (default: MSFT, GOOGL, TSLA)")
    ap.add_argument("--label", default="sign", choices=["sign", "marginband"],
                    help="Use classic sign label or margin-band label (drops neutral rows).")
    args = ap.parse_args()

    # --------------------------
    # Load dataframe
    # --------------------------
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    else:
        df = pd.read_parquet(args.input)
        if "date" in df.columns and not is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        else:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
            except Exception:
                pass

    df = df.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    # --------------------------
    # Which tickers to keep
    # --------------------------
    default_keep = ["MSFT", "GOOGL", "TSLA"]
    keep = args.tickers if args.tickers is not None else default_keep
    df = df[df["ticker"].isin(keep)].copy()
    print(f"[INFO] Using tickers: {keep}")

    # --------------------------
    # Choose label column
    # --------------------------
    if args.label == "marginband":
        y_col = f"y_mb_{args.horizon}d"
    else:
        # classic sign label (backward compatibility)
        y_col = f"y_up_{args.horizon}d" if f"y_up_{args.horizon}d" in df.columns else "y_up_next"

    if y_col not in df.columns:
        raise ValueError(f"Label column not found: {y_col}")

    # If margin-band: drop neutral rows (-1)
    if args.label == "marginband":
        before = len(df)
        df = df[df[y_col].isin([0, 1])].copy()
        print(f"[INFO] Dropped neutrals for {y_col}: {before - len(df)} rows")

    # --------------------------
    # Feature columns
    # --------------------------
    feat_cols = pick_features(df, use_sentiment=args.use_sentiment)
    if len(feat_cols) == 0:
        raise ValueError("No feature columns found. Check your input file and feature lists.")

    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []
    X_te_list, y_te_list = [], []

    # --------------------------
    # Per-ticker processing
    # --------------------------
    for ticker in keep:
        dft = df[df["ticker"] == ticker].dropna(subset=feat_cols + [y_col]).copy()
        if len(dft) < (args.seq_len + 60):  # minimum rows to ensure windows per split
            print(f"[WARN] Skipping {ticker}: not enough rows ({len(dft)})")
            continue
        dft = dft.sort_values("date").reset_index(drop=True)

        feats = dft[feat_cols].values.astype(np.float32)
        labels = dft[y_col].astype(int).values

        # temporal split indices on rows, then window within each split
        tr_idx, va_idx, te_idx = temporal_split_idx(len(dft), test_frac=0.15, val_frac=0.15)

        # fit scaler on train only (per ticker)
        mu, sd = standardize_fit(feats[tr_idx])
        feats_tr = standardize_apply(feats[tr_idx], mu, sd)
        feats_va = standardize_apply(feats[va_idx], mu, sd)
        feats_te = standardize_apply(feats[te_idx], mu, sd)

        # build windows within each split (avoid leakage)
        Xtr, ytr = build_windows(feats_tr, labels[tr_idx], args.seq_len)
        Xva, yva = build_windows(feats_va, labels[va_idx], args.seq_len)
        Xte, yte = build_windows(feats_te, labels[te_idx], args.seq_len)

        # stash
        if len(ytr):
            X_tr_list.append(Xtr); y_tr_list.append(ytr)
        if len(yva):
            X_va_list.append(Xva); y_va_list.append(yva)
        if len(yte):
            X_te_list.append(Xte); y_te_list.append(yte)

    # --------------------------
    # Concatenate splits across tickers
    # --------------------------
    def cat_or_empty(lst, axis=0, feat_dim=len(feat_cols)):
        return np.concatenate(lst, axis=axis) if len(lst) > 0 else np.zeros((0, args.seq_len, feat_dim), dtype=np.float32)

    X_train = cat_or_empty(X_tr_list)
    y_train = np.concatenate(y_tr_list) if len(y_tr_list) > 0 else np.zeros((0,), dtype=int)
    X_val   = cat_or_empty(X_va_list)
    y_val   = np.concatenate(y_va_list) if len(y_va_list) > 0 else np.zeros((0,), dtype=int)
    X_test  = cat_or_empty(X_te_list)
    y_test  = np.concatenate(y_te_list) if len(y_te_list) > 0 else np.zeros((0,), dtype=int)

    # --------------------------
    # Save
    # --------------------------
    out_path = args.out.format(h=args.horizon, L=args.seq_len)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        feature_names=np.array(feat_cols, dtype=object),
        tickers=np.array(keep, dtype=object),
    )

    print("Saved:", os.path.abspath(out_path))
    print("Shapes:",
          "X_train", X_train.shape, "y_train", y_train.shape,
          "X_val",   X_val.shape,   "y_val",   y_val.shape,
          "X_test",  X_test.shape,  "y_test",  y_test.shape)
    print("Features:", len(feat_cols))
    print("Tickers used:", keep)
    print("Label column:", y_col)
    print("Seq len:", args.seq_len, "| Horizon:", args.horizon, "| Label mode:", args.label)

if __name__ == "__main__":
    main()
    