# scripts/build_sentiment_from_interim_msft_new.py
import pandas as pd
from pathlib import Path

SRC = Path("data/interim/tweet_scored.csv")
OUT = Path("data/interim/sentiment_daily_msft.csv")  # NEW FILE

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source: {SRC}")

    df = pd.read_csv(SRC)

    # 1) Date column: pick the first that exists
    date_col = next((c for c in ["date","created_at","timestamp","time","datetime"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            "No date column found. Expected one of: date, created_at, timestamp, time, datetime."
        )
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    # 2) Ticker column: if missing, inject MSFT
    ticker_col = next((c for c in ["ticker","symbol","cashtag"] if c in df.columns), None)
    if ticker_col is None:
        df["ticker"] = "MSFT"
        ticker_col = "ticker"
    df["ticker"] = df[ticker_col].astype(str).str.upper()

    # 3) Sentiment score: prefer numeric 'compound', else map {-1,0,1} from 'label'
    has_comp = "compound" in df.columns
    has_label = "label" in df.columns

    if not has_comp and not has_label:
        raise ValueError("Need a 'compound' column or a categorical 'label' in {-1,0,1}.")

    if has_comp:
        df["score"] = pd.to_numeric(df["compound"], errors="coerce")
        pos_mask = df["score"] > 0
        neg_mask = df["score"] < 0
    else:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df["score"] = df["label"].map({-1: -1.0, 0: 0.0, 1: 1.0})
        pos_mask = df["label"] == 1
        neg_mask = df["label"] == -1

    # 4) Keep only MSFT rows (in case a ticker column was present with other symbols)
    df = df[df["ticker"] == "MSFT"].copy()
    if df.empty:
        raise ValueError("No MSFT rows available after normalization/injection.")

    # 5) Aggregate to daily
    daily = (
        df.groupby(["ticker","date"])
          .agg(mean_sent=("score","mean"),
               pos_share=(pos_mask,"mean"),
               neg_share=(neg_mask,"mean"),
               n_tweets=("score","size"))
          .reset_index()
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT, index=False)
    print(f"[OK] Wrote {OUT} with {len(daily)} daily rows for MSFT")

if __name__ == "__main__":
    main()
