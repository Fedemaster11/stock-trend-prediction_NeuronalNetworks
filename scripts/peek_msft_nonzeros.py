import pandas as pd
from pathlib import Path

FP = Path(r"data\processed\merged_dataset_with_sent.csv")

df = pd.read_csv(FP)
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
df["date"]   = pd.to_datetime(df["date"])

ms = df[df["ticker"]=="MSFT"].copy()

print("MSFT rows:", len(ms))
print("MSFT mean_sent non-zeros:", (ms["mean_sent"]!=0).sum())

nz = ms[ms["mean_sent"] != 0].sort_values("date")
print("\nFirst 10 non-zero rows:")
print(nz[["date","mean_sent","n_tweets"]].head(10))

print("\nLast 10 non-zero rows:")
print(nz[["date","mean_sent","n_tweets"]].tail(10))

# Optional: show around the very first non-zero date
if not nz.empty:
    first_date = nz["date"].iloc[0]
    win = ms[(ms["date"]>=first_date-pd.Timedelta(days=3)) & (ms["date"]<=first_date+pd.Timedelta(days=3))]
    print("\nWindow around first non-zero date:")
    print(win[["date","mean_sent","n_tweets"]].sort_values("date"))
