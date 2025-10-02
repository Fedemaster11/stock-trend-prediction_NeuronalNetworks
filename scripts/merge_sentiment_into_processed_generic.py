import argparse
import pandas as pd
import numpy as np
from pathlib import Path

SENT_COLS = ["mean_sent", "n_tweets", "pos_share", "neg_share"]

def coalesce(old, new):
    # keep old if it's non-zero/non-null; otherwise take new
    return np.where((pd.isna(old)) | (old == 0), new, old)

def normalize_sent_df(dfs: pd.DataFrame, inject_ticker: str | None = None) -> pd.DataFrame:
    # case-insensitive mapper
    lowmap = {c.lower(): c for c in dfs.columns}

    def pick(*names):
        for n in names:
            if n in lowmap:
                return lowmap[n]
        return None

    # ticker/date columns
    tick_c = pick("ticker", "symbol")
    date_c = pick("date", "created_at", "timestamp", "time", "datetime")

    if tick_c is None:
        if inject_ticker is None:
            raise ValueError("Sentiment file needs a ticker column or pass --inject-ticker.")
        dfs["ticker"] = inject_ticker
        tick_c = "ticker"

    if date_c is None:
        raise ValueError("Sentiment file needs a date column (Date/created_at/timestamp/time/datetime).")

    # rename to standard schema
    ren = {tick_c: "ticker", date_c: "date"}
    for want in SENT_COLS:
        c = pick(want)
        if c is not None:
            ren[c] = want
    dfs = dfs.rename(columns=ren)

    # ensure all sentiment columns exist
    for c in SENT_COLS:
        if c not in dfs.columns:
            dfs[c] = 0.0

    # types
    dfs["ticker"] = dfs["ticker"].astype(str).str.upper()
    dfs["date"]   = pd.to_datetime(dfs["date"], errors="coerce").dt.date
    for c in SENT_COLS:
        dfs[c] = pd.to_numeric(dfs[c], errors="coerce").fillna(0.0)

    return dfs[["ticker", "date"] + SENT_COLS]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, help="Path to processed merged_dataset.csv")
    ap.add_argument("--sentiment", required=True, help="Path to sentiment daily CSV (MSFT in your case)")
    ap.add_argument("--out", required=True, help="Output CSV (write a new file to be safe)")
    ap.add_argument("--tickers", default="", help="Optional comma list to keep from sentiment (e.g., MSFT,TSLA)")
    ap.add_argument("--inject-ticker", default=None, help="If sentiment file has no ticker col, inject this value (e.g., MSFT)")
    args = ap.parse_args()

    proc_path = Path(args.processed)
    sent_path = Path(args.sentiment)
    out_path  = Path(args.out)

    if not proc_path.exists():
        raise FileNotFoundError(proc_path)
    if not sent_path.exists():
        raise FileNotFoundError(sent_path)

    # Load
    dfp = pd.read_csv(proc_path)
    dfs = pd.read_csv(sent_path)

    # Normalize
    dfp["ticker"] = dfp["ticker"].astype(str).str.upper()
    dfp["date"]   = pd.to_datetime(dfp["date"], errors="coerce").dt.date
    for c in SENT_COLS:
        if c not in dfp.columns:
            dfp[c] = 0.0

    dfs = normalize_sent_df(dfs, inject_ticker=args.inject_ticker)

    # Optional filter of sentiment tickers
    if args.tickers:
        keep = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        dfs = dfs[dfs["ticker"].isin(keep)]

    # Safety stats before merge
    before = {
        t: (dfp.loc[dfp["ticker"] == t, "mean_sent"] != 0).sum()
        for t in (dfs["ticker"].unique())
    }

    # Merge
    df = dfp.merge(dfs, on=["ticker", "date"], how="left", suffixes=("", "_new"))

    # Coalesce sentiment cols
    for c in SENT_COLS:
        newc = c + "_new"
        if newc in df.columns:
            df[c] = coalesce(df[c], df[newc].fillna(0))
            df.drop(columns=[newc], inplace=True)

    # Save
    df.to_csv(out_path, index=False)

    after = {
        t: (df.loc[df["ticker"] == t, "mean_sent"] != 0).sum()
        for t in (dfs["ticker"].unique())
    }

    # Report
    print(f"[OK] Wrote {out_path} shape={df.shape}")
    for t in after:
        print(f"[{t}] mean_sent non-zeros: {before.get(t,0)} -> {after[t]}")

if __name__ == "__main__":
    main()
