# scripts/build_sentiment_daily.py
# Aggregate tweet-level files -> daily sentiment per ticker/date
# Tries to auto-detect common column names.

import os, glob, argparse, re
import pandas as pd
import numpy as np

def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def detect_col(cols, *candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

def extract_cashtags(text, tickers_upper):
    if not isinstance(text, str):
        return []
    tags = set(m.group(1).upper() for m in re.finditer(r'\$([A-Za-z\.]+)', text))
    return list(tags & set(tickers_upper))

def load_all(folder=None, files=None):
    paths = []
    if folder:
        paths += glob.glob(os.path.join(folder, "*.csv"))
    if files:
        for f in files:
            if "*" in f or "?" in f:
                paths += glob.glob(f)
            else:
                paths.append(f)
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("[ERR] No input CSVs found.")
    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p, low_memory=False)
            d["_src"] = os.path.basename(p)
            dfs.append(d)
        except Exception as e:
            print(f"[warn] Could not read {p}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/raw/tweets", help="Folder with tweet CSVs")
    ap.add_argument("--files", nargs="*", help="Specific CSVs (optional)")
    ap.add_argument("--tickers", default="MSFT,TSLA,GOOGL", help="Tickers to keep")
    ap.add_argument("--out", default="data/interim/sentiment_daily.csv", help="Output CSV")
    # optional column hints
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--ticker-col", default=None)
    ap.add_argument("--score-col", default=None)   # e.g. 'compound' or 'sentiment' numeric in [-1,1]
    ap.add_argument("--label-col", default=None)   # e.g. 'label' in {positive,neutral,negative}
    ap.add_argument("--pos-th", type=float, default=0.05)
    ap.add_argument("--neg-th", type=float, default=-0.05)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    tickset = set(tickers)

    df = load_all(folder=args.folder, files=args.files)
    if df.empty:
        raise SystemExit("[ERR] No tweets loaded.")
    df = norm_cols(df)
    cols = df.columns

    # --- detect date ---
    date_col = args.date_col or detect_col(cols, "date", "created_at", "timestamp")
    if not date_col:
        raise SystemExit("[ERR] Could not detect a date column. Use --date-col.")
    dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["date"] = dt.dt.tz_localize(None) if hasattr(dt, "dt") else pd.to_datetime(df[date_col], errors="coerce")
    df["date"] = df["date"].dt.date

    # --- detect ticker ---
    ticker_col = args.ticker_col or detect_col(cols, "ticker", "symbol", "stock", "cashtag")
    if ticker_col:
        df["ticker"] = df[ticker_col].astype(str).str.upper().str.strip()
    else:
        text_col = detect_col(cols, "text", "content", "body")
        if not text_col:
            raise SystemExit("[ERR] No ticker column and no text column to parse $TICKER from.")
        # derive from $cashtags
        found = df[text_col].apply(lambda x: extract_cashtags(x, tickers))
        df = df.loc[found.apply(len) > 0].copy()
        df["ticker"] = found.apply(lambda lst: lst[0])  # take first match

    df = df[df["ticker"].isin(tickset)].copy()

    # --- detect sentiment signal ---
    label_col = args.label_col or detect_col(cols, "label", "sentiment_label")
    score_col = args.score_col or detect_col(cols, "compound", "sentiment", "polarity", "score")

    # build a numeric score in [-1,1]
    score = None
    if score_col and pd.api.types.is_numeric_dtype(df[score_col]):
        score = df[score_col].astype(float).clip(-1,1)
    elif label_col:
        lab = df[label_col].astype(str).str.lower().str.strip()
        mapping = {"positive": 1.0, "pos": 1.0, "negative": -1.0, "neg": -1.0, "neutral": 0.0, "neu": 0.0}
        score = lab.map(mapping)
    else:
        raise SystemExit("[ERR] Could not detect numeric score or label. Use --score-col or --label-col.")

    df["_score"] = score
    df = df.dropna(subset=["date", "ticker", "_score"])

    # daily agg per (ticker,date)
    def daily_agg(g):
        s = g["_score"].astype(float)
        n = len(s)
        pos_share = float((s > args.pos_th).mean()) if n else 0.0
        neg_share = float((s < args.neg_th).mean()) if n else 0.0
        mean_sent = float(s.mean()) if n else 0.0
        return pd.Series({
            "mean_sent": mean_sent,
            "pos_share": pos_share,
            "neg_share": neg_share,
            "n_tweets": n
        })

    out = (df.groupby(["ticker","date"], as_index=False)
             .apply(daily_agg)
             .reset_index(drop=True))

    # ensure types
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out[["ticker","date","mean_sent","pos_share","neg_share","n_tweets"]]
    out.to_csv(args.out, index=False)
    print(f"[OK] wrote {len(out):,} rows -> {args.out}")
    print(out.head(5))
    print("\nTickers:", out["ticker"].unique())
    by = out.groupby("ticker")["date"].agg(["min","max","count"])
    print("\nDate ranges:\n", by)

if __name__ == "__main__":
    main()
