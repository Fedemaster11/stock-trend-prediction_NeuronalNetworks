# scripts/add_tickers_simple.py
# Minimal, robust appender for TSLA/GOOGL into data/processed/merged_dataset.csv
# Handles yfinance MultiIndex where ('Open','TSLA') or ('TSLA','Open').

import os, argparse
import numpy as np
import pandas as pd
import yfinance as yf

PROCESSED = "data/processed/merged_dataset.csv"
os.makedirs("data/processed", exist_ok=True)

FIELDS_CANON = {"open","high","low","close","adj_close","volume"}
FIELDS_RAW   = {"open","high","low","close","adj close","adj_close","volume"}

def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _collapse_multiindex(df, ticker):
    """Return a DataFrame with single-level columns that are field names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)
    # Try to choose the level that looks like fields
    lvl0_norm = [str(c).strip().lower().replace(" ", "_") for c in lvl0]
    lvl1_norm = [str(c).strip().lower().replace(" ", "_") for c in lvl1]

    set0 = set(lvl0_norm)
    set1 = set(lvl1_norm)

    # If last level is the ticker (e.g., ('Open','TSLA')), use the first level (fields)
    if len(set(lvl1_norm)) == 1 and list(set(lvl1_norm))[0] == ticker.lower():
        df.columns = lvl0
        return df

    # If first level is the ticker (e.g., ('TSLA','Open')), use the second level (fields)
    if len(set(lvl0_norm)) == 1 and list(set(lvl0_norm))[0] == ticker.lower():
        df.columns = lvl1
        return df

    # Otherwise, pick the level that contains field-like names
    score0 = len(set0 & FIELDS_RAW)
    score1 = len(set1 & FIELDS_RAW)
    df.columns = lvl0 if score0 >= score1 else lvl1
    return df

def fetch_prices(ticker, start):
    df = yf.download(
        ticker,
        start=start,
        interval="1d",
        auto_adjust=False,
        progress=False,
        # group_by can behave differently across versions; handle both cases below
        # group_by="column",
        threads=False,
    )
    if df is None or df.empty:
        raise SystemExit(f"[ERR] No price data for {ticker}. Try a different symbol/date.")

    # Collapse MultiIndex to field names
    if isinstance(df.columns, pd.MultiIndex):
        df = _collapse_multiindex(df, ticker)

    df = df.reset_index()  # bring Date out of index
    df = norm_cols(df)

    # normalize keys
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime":"date"})
    if "adj_close" not in df.columns and "adjclose" in df.columns:
        df = df.rename(columns={"adjclose":"adj_close"})
    if "adj_close" not in df.columns and "adj close" in df.columns:
        df = df.rename(columns={"adj close":"adj_close"})
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    need = {"date","open","high","low","close","adj_close","volume"}
    miss = need - set(df.columns)
    if miss:
        # Show what we actually got to help debugging
        print("[dbg] columns after normalization:", list(df.columns))
        raise SystemExit(f"[ERR] {ticker}: missing cols: {sorted(miss)}")

    # numeric cast + drop essential NAs
    for c in ["open","high","low","close","adj_close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(need)).copy()

    df["ticker"] = ticker.upper()
    df["stock_name"] = ticker.upper()
    return df[["date","open","high","low","close","adj_close","volume","stock_name","ticker"]]

def add_features(px):
    df = px.sort_values("date").reset_index(drop=True)
    df["ret_1d"]     = df["adj_close"].pct_change()
    df["log_ret_1d"] = np.log1p(df["ret_1d"])
    df["ma_5"]       = df["adj_close"].rolling(5,  min_periods=5).mean()
    df["ma_20"]      = df["adj_close"].rolling(20, min_periods=20).mean()
    df["vol_20"]     = df["ret_1d"].rolling(20, min_periods=20).std()
    df["adj_close_t1"] = df["adj_close"].shift(-1)
    df["y_up_next"]    = (df["adj_close_t1"].values > df["adj_close"].values).astype(int)
    # sentiment placeholders
    df["mean_sent"] = 0.0
    df["pos_share"] = 0.0
    df["neg_share"] = 0.0
    df["n_tweets"]  = 0
    need = ["ret_1d","log_ret_1d","ma_5","ma_20","vol_20","adj_close_t1","y_up_next"]
    return df.dropna(subset=need).copy()

def safe_merge(new_df):
    # Load or init base
    if os.path.exists(PROCESSED):
        base = pd.read_csv(PROCESSED)
        base = norm_cols(base)
    else:
        base = pd.DataFrame()

    new_df = norm_cols(new_df)
    if not base.empty:
        merged = pd.concat([base, new_df], ignore_index=True)
    else:
        merged = new_df

    must = ['date','open','high','low','close','adj_close','volume','stock_name','ticker',
            'log_ret_1d','ret_1d','ma_5','ma_20','vol_20','adj_close_t1','y_up_next',
            'mean_sent','n_tweets','pos_share','neg_share']
    for c in must:
        if c not in merged.columns:
            merged[c] = 0.0
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.sort_values(["ticker","date"]).drop_duplicates(subset=["ticker","date"], keep="last")
    merged.to_csv(PROCESSED, index=False)
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="TSLA,GOOGL", help="Comma-separated tickers")
    ap.add_argument("--start",   default="2015-01-01", help="Start date YYYY-MM-DD")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    all_new = []
    for tk in tickers:
        print(f"[+] Fetching {tk} ...")
        px = fetch_prices(tk, args.start)
        fx = add_features(px)
        print(f"    {tk}: {len(fx)} rows | {fx['date'].min()} → {fx['date'].max()}")
        all_new.append(fx)

    new_df = pd.concat(all_new, ignore_index=True)
    merged = safe_merge(new_df)
    print(f"[OK] merged rows: {len(merged)}")
    print(merged.groupby("ticker")["date"].agg(["min","max","count"]))

if __name__ == "__main__":
    main()
