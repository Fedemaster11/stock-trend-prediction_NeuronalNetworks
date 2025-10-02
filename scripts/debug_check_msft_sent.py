# scripts/debug_check_msft_sent.py (fix)
import pandas as pd
from pathlib import Path

OUT = Path(r"data\processed\merged_dataset_with_sent.csv")

df = pd.read_csv(OUT)
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
df["date"] = pd.to_datetime(df["date"])  # <-- make Timestamp

ms = df[df["ticker"]=="MSFT"].copy()
print(f"MSFT rows: {len(ms)} | mean_sent non-zeros: {(ms['mean_sent']!=0).sum()}")

mask = (ms["date"] >= "2021-09-01") & (ms["date"] <= "2021-10-31")
print(ms.loc[mask, ["date","mean_sent","n_tweets"]].head(20))
