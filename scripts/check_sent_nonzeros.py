# scripts/check_sent_nonzeros.py
import sys
import pandas as pd

DATA = "data/processed/merged_dataset.csv"
TICKERS = sys.argv[1:] or ["TSLA", "GOOGL"]

df = pd.read_csv(DATA)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

needed = {"ticker", "date"}
missing = needed - set(df.columns)
if missing:
    raise SystemExit(f"[ERR] Missing columns in merged_dataset.csv: {sorted(missing)}")

# Optional sentiment columns
sent_cols = ["mean_sent", "pos_share", "neg_share", "n_tweets"]
present = [c for c in sent_cols if c in df.columns]
if not present:
    raise SystemExit("[INFO] No sentiment columns found at all.")

print(f"\nChecking sentiment non-zeros for: {', '.join(TICKERS)}")
for tk in TICKERS:
    d = df[df["ticker"].str.upper() == tk.upper()].copy()
    if d.empty:
        print(f"\n{tk}: 0 rows in dataset.")
        continue
    print(f"\n{tk}: {len(d)} rows")
    for c in present:
        nz = (d[c] != 0).sum()
        pct = (nz / len(d)) * 100
        print(f"  {c:<10} non-zeros: {nz:>6} ({pct:5.1f}%)")
