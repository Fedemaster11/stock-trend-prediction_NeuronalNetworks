import os, numpy as np, pandas as pd

def merge_sentiment_data(merged_file_path, sentiment_file_path, output_file_path):
    if not os.path.exists(merged_file_path):
        raise FileNotFoundError(f"Main merged dataset not found: {merged_file_path}")
    if not os.path.exists(sentiment_file_path):
        raise FileNotFoundError(f"Sentiment dataset not found: {sentiment_file_path}")

    df_merged = pd.read_csv(merged_file_path)
    df_sent = pd.read_csv(sentiment_file_path)

    # normalize names/case
    df_merged.rename(columns=lambda c: c.strip(), inplace=True)
    df_sent.rename(columns=lambda c: c.strip(), inplace=True)
    if "Ticker" in df_sent.columns: df_sent.rename(columns={"Ticker":"ticker"}, inplace=True)
    if "Date" in df_sent.columns:   df_sent.rename(columns={"Date":"date"}, inplace=True)

    if "ticker" not in df_merged.columns or "date" not in df_merged.columns:
        raise ValueError("merged CSV needs columns 'ticker' and 'date'.")
    if "ticker" not in df_sent.columns or "date" not in df_sent.columns:
        raise ValueError("sentiment CSV needs columns 'Ticker/Date' or 'ticker/date'.")

    # types
    df_merged["ticker"] = df_merged["ticker"].astype(str).str.upper().str.strip()
    df_sent["ticker"]   = df_sent["ticker"].astype(str).str.upper().str.strip()
    df_merged["date"] = pd.to_datetime(df_merged["date"], errors="coerce").dt.date
    df_sent["date"]   = pd.to_datetime(df_sent["date"], errors="coerce").dt.date

    # only MSFT from sentiment
    df_sent = df_sent[df_sent["ticker"] == "MSFT"].copy()
    if df_sent.empty:
        raise ValueError("No MSFT rows found in sentiment_daily.csv")

    # ensure sentiment cols exist
    scols = ["mean_sent","n_tweets","pos_share","neg_share"]
    for c in scols:
        if c not in df_sent.columns: df_sent[c] = 0.0
        if c not in df_merged.columns: df_merged[c] = 0.0
        df_sent[c] = pd.to_numeric(df_sent[c], errors="coerce").fillna(0.0)

    # merge
    out = df_merged.merge(
        df_sent[["ticker","date"] + scols],
        on=["ticker","date"], how="left", suffixes=("", "_new")
    )

    # coalesce: keep existing if non-zero, otherwise take new
    for c in scols:
        newc = f"{c}_new"
        out[c] = np.where((out[c].fillna(0) == 0), out[newc].fillna(0), out[c])
        if newc in out.columns: out.drop(columns=[newc], inplace=True)

    out.to_csv(output_file_path, index=False)
    print(f"✅ Saved: {output_file_path}")

if __name__ == "__main__":
    merge_sentiment_data(
        merged_file_path=r"data\processed\merged_dataset.csv",
        sentiment_file_path=r"data\processed\sentiment_daily.csv",
        output_file_path=r"data\processed\merged_dataset_with_sent.csv",
    )
