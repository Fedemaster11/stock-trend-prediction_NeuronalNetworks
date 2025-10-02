import pandas as pd

# Load your merged dataset
df = pd.read_csv("data/processed/merged_dataset.csv")

# Normalize column names: lowercase + underscores (e.g., 'Stock Name' -> 'stock_name')
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Now we can rely on 'date' and 'ticker'
if "date" not in df.columns or "ticker" not in df.columns:
    raise ValueError(f"Expected columns 'date' and 'ticker' not found. Columns present: {df.columns.tolist()}")

# Parse date
df["date"] = pd.to_datetime(df["date"], errors="raise")

# Row counts per ticker
counts = df["ticker"].value_counts()
print("📊 Row counts per ticker:")
print(counts)

# Top ticker
top_ticker = counts.idxmax()
top_count = counts.max()
print(f"\n🏆 Stock with the most data: {top_ticker} ({top_count} rows)")

# Date ranges per ticker
print("\n🗓️ Date range per ticker:")
for t in counts.index:
    dft = df[df["ticker"] == t]
    print(f"- {t}: {dft['date'].min().date()} → {dft['date'].max().date()} (rows={len(dft)})")

# Global date range
print("\n🌍 Full dataset date range:", df["date"].min().date(), "→", df["date"].max().date())
