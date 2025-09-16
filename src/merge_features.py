#!/usr/bin/env python3
import argparse, os, glob
import pandas as pd, numpy as np

PRICE_COLS = ["Open","High","Low","Close","Adj Close","Volume"]

def to_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in PRICE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def engineer_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    # make sure numeric
    df = to_numeric_cols(df)
    # basic features
    df["log_ret_1d"] = np.log(df["Adj Close"]).diff()
    df["ret_1d"] = df["Adj Close"].pct_change()
    df["ma_5"] = df["Adj Close"].rolling(5).mean()
    df["ma_20"] = df["Adj Close"].rolling(20).mean()
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    # next-day target
    df["adj_close_t1"] = df["Adj Close"].shift(-1)
    df["y_up_next"] = (df["adj_close_t1"] > df["Adj Close"]).astype(int)
    return df

def load_prices(price_dir: str) -> pd.DataFrame:
    paths = glob.glob(os.path.join(price_dir, "*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {price_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # normalize Date & Ticker
        if "Date" not in df.columns:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        if "Ticker" not in df.columns:
            ticker = os.path.basename(p).split("_")[0].replace("-", ".")
            df["Ticker"] = ticker

        # parse date & coerce numerics
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = to_numeric_cols(df)

        # drop rows with missing critical fields
        df = df.dropna(subset=["Date","Adj Close"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out

def merge_with_sentiment(price_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.date
    sent_df["Date"]  = pd.to_datetime(sent_df["Date"]).dt.date
    merged = price_df.merge(sent_df, how="left", on=["Date","Ticker"])

    # forward-fill sentiment per ticker, then neutral-fill
    for col in ["mean_sent","pos_share","neg_share"]:
        if col in merged.columns:
            merged[col] = (merged.groupby("Ticker")[col].ffill()).fillna(0.0)
        else:
            merged[col] = 0.0
    merged["n_tweets"] = merged.get("n_tweets", 0).fillna(0)

    return merged

def main(price_dir: str, sentiment_csv: str, outfile: str):
    prices = load_prices(price_dir)
    prices = engineer_price_features(prices)

    if os.path.exists(sentiment_csv):
        sent = pd.read_csv(sentiment_csv)
        merged = merge_with_sentiment(prices, sent)
    else:
        merged = prices.copy()
        merged["mean_sent"] = 0.0
        merged["pos_share"] = 0.0
        merged["neg_share"] = 0.0
        merged["n_tweets"] = 0.0

    # drop rows where rolling features are NaN (warmup period)
    merged = merged.dropna(subset=["ma_5","ma_20","vol_20","log_ret_1d","ret_1d","y_up_next"])

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    merged.to_csv(outfile, index=False)
    print(f"[merge_features] saved {len(merged)} rows -> {outfile}")
    print("[merge_features] columns:", list(merged.columns))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge OHLCV features with daily sentiment to produce modeling dataset.")
    p.add_argument("--price_dir", default="data/raw/prices")
    p.add_argument("--sentiment_csv", default="data/processed/sentiment_daily.csv")
    p.add_argument("--outfile", default="data/processed/merged_dataset.csv")
    a = p.parse_args()
    main(a.price_dir, a.sentiment_csv, a.outfile)
