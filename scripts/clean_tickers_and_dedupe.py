import pandas as pd, os
p = "data/processed/merged_dataset.csv"
assert os.path.exists(p), f"Missing {p}"
df = pd.read_csv(p)
df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
if "ticker" not in df.columns or "date" not in df.columns:
    raise SystemExit("[ERR] 'ticker' or 'date' missing.")
df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values(["ticker","date"]).drop_duplicates(subset=["ticker","date"], keep="last")
df.to_csv(p, index=False)
print("[OK] standardized tickers & re-deduped. Counts:")
print(df.groupby("ticker")["date"].agg(["min","max","count"]))
