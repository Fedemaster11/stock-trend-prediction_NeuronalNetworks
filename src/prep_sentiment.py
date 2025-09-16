#!/usr/bin/env python3
import argparse, os, re, pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Use [\$] to match a literal dollar sign; robust against shell expansion.
TICKER_PATTERNS = {
    "PLTR": r"([\$]?PLTR|Palantir)",
    "MSFT": r"([\$]?MSFT|Microsoft)",
    # Include XETRA RHM.DE and common US OTC symbols; accept $RHM and $RHM.DE
    "RHM.DE": r"([\$]?RHM(?:\.DE)?|Rheinmetall|RNMBF|RNMBY)"
}

def detect_tickers(text: str):
    hits = []
    for ticker, pat in TICKER_PATTERNS.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(ticker)
    return hits

def load_kaggle_csv(path: str):
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    # Try to infer columns
    cand_text = [c for c in df.columns if c.lower() in ["text","content","tweet","clean_text","body"]]
    text_col = cand_text[0] if cand_text else df.columns[0]
    cand_time = [c for c in df.columns if any(k in c.lower() for k in ["date","time","created","timestamp"])]
    time_col = cand_time[0] if cand_time else None
    return df, text_col, time_col

def daily_sentiment(df: pd.DataFrame, text_col: str, time_col: str | None):
    sid = SentimentIntensityAnalyzer()
    records = []
    it = tqdm(enumerate(df[text_col].astype(str)), total=len(df), desc="[prep_sentiment] Scoring")
    for i, txt in it:
        tickers = detect_tickers(txt)
        if not tickers:
            continue
        score = sid.polarity_scores(txt)["compound"]
        if time_col:
            try:
                dt = pd.to_datetime(df.iloc[i][time_col]).date()
            except Exception:
                dt = pd.NaT
        else:
            dt = pd.NaT
        records.append({"date": dt, "text": txt, "score": score, "tickers": tickers})
    scored = pd.DataFrame.from_records(records)
    if scored.empty:
        return scored, pd.DataFrame()
    scored = scored.explode("tickers").dropna(subset=["date"]).copy()
    scored["date"] = pd.to_datetime(scored["date"])
    daily = (scored.groupby(["tickers", scored["date"].dt.date])
                   .agg(mean_sent=("score","mean"),
                        n_tweets=("score","size"),
                        pos_share=("score", lambda s: (s>0).mean()),
                        neg_share=("score", lambda s: (s<0).mean()))
                   .reset_index()
                   .rename(columns={"tickers":"Ticker","date":"Date"}))
    return scored, daily

def main(infile: str, out_scored: str, out_daily: str):
    df, text_col, time_col = load_kaggle_csv(infile)
    print(f"[prep_sentiment] Using columns: text='{text_col}' time='{time_col}'")
    scored, daily = daily_sentiment(df, text_col, time_col)
    os.makedirs(os.path.dirname(out_scored), exist_ok=True)
    os.makedirs(os.path.dirname(out_daily), exist_ok=True)
    if not scored.empty:
        try:
            scored.to_parquet(out_scored, index=False)
            print(f"[prep_sentiment] wrote {len(scored):,} rows -> {out_scored}")
        except Exception:
            alt = out_scored.replace(".parquet", ".csv")
            scored.to_csv(alt, index=False)
            print(f"[prep_sentiment] parquet failed; wrote CSV -> {alt}")
    else:
        print("[prep_sentiment] No per-tweet scores generated.")
    if not daily.empty:
        daily.to_csv(out_daily, index=False)
        print(f"[prep_sentiment] wrote {len(daily):,} daily rows -> {out_daily}")
    else:
        print("[prep_sentiment] No daily aggregates generated.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--out_scored", default="data/interim/tweet_scored.parquet")
    ap.add_argument("--out_daily", default="data/processed/sentiment_daily.csv")
    a = ap.parse_args()
    main(a.infile, a.out_scored, a.out_daily)
