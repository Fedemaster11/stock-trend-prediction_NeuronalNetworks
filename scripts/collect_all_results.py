#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect all results (baselines + deep models) into one CSV
and produce comparison charts.

Outputs:
- results/all_models_comparison.csv
- results/final_f1_bar.png
- results/final_auc_bar.png
"""

import os, re, json, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(ROOT, "runs")
RES_DIR  = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def parse_npz_info(s: str):
    """
    Try to parse 'seq_<TICKER>_h<horizon>_L<seq>_*.npz'
    Returns dict with keys: ticker, horizon, seq_len (str if unknown).
    """
    if not s:
        return {"ticker": None, "horizon": None, "seq_len": None}
    m = re.search(r"seq_([A-Z]+)_h(\d+)_L(\d+)", s.replace("\\", "/"))
    if m:
        return {
            "ticker": m.group(1),
            "horizon": int(m.group(2)),
            "seq_len": int(m.group(3)),
        }
    # fallback: infer from run_dir name like 'cnn1d_MSFT_h10_L60_MB'
    m2 = re.search(r"(?:lstm|bilstm|cnn1d)_([A-Z]+)_h(\d+)_L(\d+)", s.replace("\\", "/"))
    if m2:
        return {
            "ticker": m2.group(1),
            "horizon": int(m2.group(2)),
            "seq_len": int(m2.group(3)),
        }
    return {"ticker": None, "horizon": None, "seq_len": None}

def try_read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        # print(f"Could not read {path}: {e}")
        return None

def add_row(rows, model, family, run_dir, src, d, npz_hint=None):
    """
    Normalize and append one row to rows list.
    src = 'sweep' or 'final'
    d = dict from sweep_summary.json or final_summary.json
    """
    # Preferred place to parse ticker/horizon/seq_len is npz from sweep.
    # Then fallback to run_dir.
    npz_path = None
    if src == "sweep":
        npz_path = d.get("npz_path")
    info = parse_npz_info(npz_path or run_dir)

    if src == "sweep":
        # sweep_summary.json schema (our enhanced one)
        # { "metric":"f1", "best_threshold":..., 
        #   "val":{"f1":...,"precision":...,"recall":...,"auc":...},
        #   "test":{"f1":...,"precision":...,"recall":...,"auc":...}, ...}
        val = d.get("val", {}) or {}
        test = d.get("test", {}) or {}
        rows.append({
            "model": model,                # e.g., 'lstm', 'bilstm', 'cnn1d'
            "family": family,              # 'deep'
            "run_dir": run_dir,
            "ticker": info["ticker"],
            "horizon": info["horizon"],
            "seq_len": info["seq_len"],
            "source": "sweep",
            "best_threshold": d.get("best_threshold"),
            "val_f1": val.get("f1"),
            "val_precision": val.get("precision"),
            "val_recall": val.get("recall"),
            "val_auc": val.get("auc"),
            "test_f1": test.get("f1"),
            "test_precision": test.get("precision"),
            "test_recall": test.get("recall"),
            "test_auc": test.get("auc"),
        })
    else:
        # final_summary.json schema from trainers
        # {
        #   "final_metrics": {
        #     "train": {...}, "val": {...}, "test": {...}
        #   }, "args": {...}, ...
        # }
        fm = (d or {}).get("final_metrics", {})
        val = fm.get("val", {}) or {}
        test = fm.get("test", {}) or {}
        # no threshold here (default 0.5)
        rows.append({
            "model": model,
            "family": family,
            "run_dir": run_dir,
            "ticker": info["ticker"],
            "horizon": info["horizon"],
            "seq_len": info["seq_len"],
            "source": "final",
            "best_threshold": np.nan,
            "val_f1": val.get("f1"),
            "val_precision": val.get("precision"),
            "val_recall": val.get("recall"),
            "val_auc": val.get("auc"),
            "test_f1": test.get("f1"),
            "test_precision": test.get("precision"),
            "test_recall": test.get("recall"),
            "test_auc": test.get("auc"),
        })

# -------------------------------------------------------
# 1) Deep runs (LSTM / BiLSTM / CNN1D)
# -------------------------------------------------------
rows = []

for family_model_glob, model_name in [
    (os.path.join(RUNS_DIR, "lstm_*"),   "lstm"),
    (os.path.join(RUNS_DIR, "bilstm_*"), "bilstm"),
    (os.path.join(RUNS_DIR, "cnn1d_*"),  "cnn1d"),
]:
    for run_dir in glob.glob(family_model_glob):
        if not os.path.isdir(run_dir):
            continue

        # prefer sweep_summary.json if present (threshold-optimized)
        sweep_path = os.path.join(run_dir, "sweep_summary.json")
        final_path = os.path.join(run_dir, "final_summary.json")

        d_sweep = try_read_json(sweep_path)
        d_final = try_read_json(final_path)

        if d_sweep:
            add_row(rows, model_name, "deep", run_dir, "sweep", d_sweep)
        elif d_final:
            add_row(rows, model_name, "deep", run_dir, "final", d_final)
        else:
            # print(f"Skipping (no summaries): {run_dir}")
            pass

df_deep = pd.DataFrame(rows)

# -------------------------------------------------------
# 2) Baselines (results/baselines.csv)
# -------------------------------------------------------
baseline_csv = os.path.join(RES_DIR, "baselines.csv")
df_base = pd.DataFrame()
if os.path.exists(baseline_csv):
    try:
        dfb = pd.read_csv(baseline_csv)
        # Normalize column names we care about (they exist in your file)
        keep = ["ticker", "model", "feature_set", "f1", "precision", "recall", "auc"]
        present = [c for c in keep if c in dfb.columns]
        dfb = dfb[present].copy()
        dfb = dfb.rename(columns={
            "f1": "test_f1", "precision": "test_precision",
            "recall": "test_recall", "auc": "test_auc"
        })
        dfb["family"] = "baseline"
        dfb["source"] = "cv_or_holdout"
        dfb["run_dir"] = np.nan
        dfb["best_threshold"] = np.nan
        # Add consistent columns:
        dfb["horizon"] = np.nan
        dfb["seq_len"] = np.nan
        df_base = dfb
    except Exception as e:
        print(f"Could not read baselines.csv: {e}")

# -------------------------------------------------------
# 3) Merge & Save
# -------------------------------------------------------
if len(df_base) > 0:
    df_all = pd.concat([df_deep, df_base], ignore_index=True, sort=False)
else:
    df_all = df_deep.copy()

# Tidy display order
col_order = [
    "family","model","ticker","horizon","seq_len",
    "source","best_threshold",
    "val_f1","val_precision","val_recall","val_auc",
    "test_f1","test_precision","test_recall","test_auc",
    "feature_set","run_dir"
]
for c in col_order:
    if c not in df_all.columns:
        df_all[c] = np.nan
df_all = df_all[col_order]

out_csv = os.path.join(RES_DIR, "all_models_comparison.csv")
df_all.to_csv(out_csv, index=False)
print(f"[OK] Wrote {out_csv} with {len(df_all)} rows.")

# -------------------------------------------------------
# 4) Plots (Test F1 / Test AUC by Ticker × Model)
# -------------------------------------------------------
# Only plot where we have test metrics
plot_df = df_all[df_all["test_f1"].notna()].copy()

def _safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

plot_df["test_f1"]  = plot_df["test_f1"].map(_safe_float)
plot_df["test_auc"] = plot_df["test_auc"].map(_safe_float)

# Prefer threshold-optimized rows when both 'final' and 'sweep' exist for same run/model/ticker
# Heuristic: keep 'sweep' first; if duplicates, drop others.
plot_df["source_rank"] = plot_df["source"].map({"sweep":0, "final":1, "cv_or_holdout":2}).fillna(3)
plot_df.sort_values(["ticker","model","source_rank"], inplace=True)
plot_df = plot_df.drop_duplicates(subset=["ticker","model"], keep="first")

# Bar: Test F1
fig1, ax1 = plt.subplots(figsize=(8,4.5))
# order columns by family/model for consistent look
plot_df["label"] = plot_df.apply(
    lambda r: f"{r['model']}" if pd.isna(r.get("feature_set")) else f"{r['model']} ({r['feature_set']})",
    axis=1
)
# Keep a readable pivot (Ticker x ModelLabel)
piv_f1 = plot_df.pivot_table(index="ticker", columns="label", values="test_f1", aggfunc="mean")
piv_f1.plot(kind="bar", ax=ax1)
ax1.set_title("Test F1 by Ticker × Model")
ax1.set_ylabel("F1")
ax1.set_xlabel("")
ax1.grid(axis="y", alpha=0.3)
plt.tight_layout()
f1_path = os.path.join(RES_DIR, "final_f1_bar.png")
plt.savefig(f1_path, dpi=150)
plt.close(fig1)
print(f"[OK] Wrote {f1_path}")

# Bar: Test AUC
fig2, ax2 = plt.subplots(figsize=(8,4.5))
piv_auc = plot_df.pivot_table(index="ticker", columns="label", values="test_auc", aggfunc="mean")
piv_auc.plot(kind="bar", ax=ax2)
ax2.set_title("Test AUC by Ticker × Model")
ax2.set_ylabel("AUC")
ax2.set_xlabel("")
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
auc_path = os.path.join(RES_DIR, "final_auc_bar.png")
plt.savefig(auc_path, dpi=150)
plt.close(fig2)
print(f"[OK] Wrote {auc_path}")