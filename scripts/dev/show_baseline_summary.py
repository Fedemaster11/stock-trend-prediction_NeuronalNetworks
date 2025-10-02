# scripts/show_baseline_summary.py
import os, json
import pandas as pd

RESULTS = "results/baselines.csv"
DATA    = "data/processed/merged_dataset.csv"

if not os.path.exists(RESULTS):
    raise SystemExit("[ERR] No results/baselines.csv yet. Train first.")

# --- Load & normalize ---
df = pd.read_csv(RESULTS)
df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

needed = {"ticker","model","feature_set","accuracy","f1_weighted","roc_auc",
          "tn","fp","fn","tp","best_params"}
missing = needed - set(df.columns)
if missing:
    raise SystemExit(f"[ERR] results file missing columns: {sorted(missing)}")

# Keep only the latest row per (ticker, model, feature_set) based on file order
df["_row"] = range(len(df))
last = (df.sort_values("_row")
          .groupby(["ticker","model","feature_set"], as_index=False)
          .tail(1)
          .drop(columns=["_row"]))

# Focus on our 3 tickers and LogReg (adjust if you add RF/LSTM later)
focus = last[last["ticker"].isin(["MSFT","TSLA","GOOGL"])].copy()
focus = focus.sort_values(["ticker","model","feature_set"])

# Pretty print table
cols = ["ticker","model","feature_set","accuracy","f1_weighted","roc_auc","tn","fp","fn","tp","best_params"]
print("\n=== Baseline summary (latest per ticker × model × feature_set) ===")
if focus.empty:
    print("No rows for MSFT/TSLA/GOOGL found yet.")
else:
    # shorter params
    def short(x):
        try:
            d = json.loads(x)
            return ", ".join(f"{k}={v}" for k,v in d.items())
        except Exception:
            return str(x)[:120]
    focus["best_params"] = focus["best_params"].map(short)
    print(focus[cols].to_string(index=False, justify='center'))

# Optional: side-by-side compare Price-only vs Price+Sentiment per ticker
wide = (focus
        .pivot_table(index=["ticker","model"],
                     columns="feature_set",
                     values=["accuracy","f1_weighted","roc_auc"],
                     aggfunc="first"))
if not wide.empty:
    print("\n=== Side-by-side (Price-only vs Price + Sentiment) ===")
    # reorder cols if present
    wide = wide.reindex(columns=pd.MultiIndex.from_product(
        [["accuracy","f1_weighted","roc_auc"], ["Price-only","Price + Sentiment"]]
    ), fill_value=float("nan"))
    print(wide.round(4))

# Also show dataset date ranges (sanity)
if os.path.exists(DATA):
    d = pd.read_csv(DATA)
    d.columns = [c.strip().lower().replace(" ","_") for c in d.columns]
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        print("\n=== Processed dataset date ranges ===")
        print(d.groupby("ticker")["date"].agg(["min","max","count"]))
