import pandas as pd
p = "data/processed/merged_dataset.csv"
df = pd.read_csv(p)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print("\nTicker row counts:")
print(df["ticker"].value_counts())
print("\nDate ranges:")
print(df.groupby("ticker")["date"].agg(["min","max","count"]))
