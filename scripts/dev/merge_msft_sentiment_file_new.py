# scripts/merge_msft_sentiment_file_new.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# INPUTS (edit if your paths differ)
PROC_IN  = Path("data/processed/merged_dataset.csv")
SENT_IN  = Path("data/interim/sentiment_daily_msft.csv")   # your file
PROC_OUT = Path("data/processed/merged_dataset_with_msft_sent.csv")

SENT_COLS_STD = ["mean_sent", "pos_share", "neg_share", "n_tweets"]

def norm_cols(df):
    # normalize case-insensitive column names for date/ticker/sentiment
    cols = {c.lower(): c for c in df.columns}
    def get(name_opts):
        for opt in name_opts:
            if opt in cols: return cols[opt]
        return None

    date_c = get(["date"])
    tick_c = get(["ticker","symbol"])
    if date_c is None or tick_c is None:
        raise ValueError("Need date and ticker columns in sentiment file (any case).")

    # rename to standard
    ren = {date_c: "date", tick_c: "ticker"}
    # try to map sentiment columns
    for want in SENT_COLS_STD:
        c = get([want])
        if c is not None:
            ren[c] = want
    df = df.rename(columns=ren)

    # ensure all SENT_COLS_STD exist
    for c in SENT_COLS_STD:
        if c not in df.columns:
            df[c] = 0.0

    # types and formatting
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in SENT_COLS_STD:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df[["ticker","date"] + SENT_COLS_STD]

def coalesce(old, new):
    return np.where((pd.isna(old)) | (old == 0), new, old)

def main():
    if not PROC_IN.exists():
        raise FileNotFoundError(PROC_IN)
    if not SENT_IN.exists():
        raise FileNotFoundError(SENT_IN)

    dfp = pd.read_csv(PROC_IN)
    dfs = pd.read_csv(SENT_IN)

    # normalize both
    dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce").dt.date
    dfp["ticker"] = dfp["ticker"].astype(str).str.upper()
    # make sure sentiment columns exist in processed
    for c in SENT_COLS_STD:
        if c not in dfp.columns:
            dfp[c] = 0.0

    dfs = norm_cols(dfs)

    # keep only MSFT from the sentiment file (safety)
    dfs = dfs[dfs["ticker"] == "MSFT"].copy()
    if dfs.empty:
        raise ValueError("No MSFT rows in sentiment file.")

    # merge (left join keeps all processed rows; sentiment fills where dates match)
    df = dfp.merge(dfs, on=["ticker","date"], how="left", suffixes=("", "_new"))

    # coalesce per sentiment column
    for c in SENT_COLS_STD:
        newc = c + "_new"
        df[c] = coalesce(dfp[c], df[newc].fillna(0))
        if newc in df.columns:
            df.drop(columns=[newc], inplace=True)

    # finish
    df.to_csv(PROC_OUT, index=False)
    print(f"[OK] wrote {PROC_OUT} with shape {df.shape}")

if __name__ == "__main__":
    main()
