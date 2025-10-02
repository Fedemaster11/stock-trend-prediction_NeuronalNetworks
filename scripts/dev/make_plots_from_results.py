#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Config / Inputs
# --------------------------
RESULTS_DIR = "results"
RUNS_DIR = "runs"
FINAL_CSV = os.path.join(RESULTS_DIR, "final_comparison.csv")
OUT_DIR = RESULTS_DIR

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# Helpers
# --------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def load_final_comparison(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}. Run your collection step first.")
    df = pd.read_csv(path)
    # Normalize column names we expect: ticker, model, f1, precision, recall, auc
    cols = {c.lower(): c for c in df.columns}
    # Try flexible mapping
    rename_map = {}
    for want in ["ticker", "model", "f1", "precision", "recall", "auc"]:
        # if exact lower exists
        if want in cols:
            rename_map[cols[want]] = want
        else:
            # try partial
            for c in df.columns:
                if c.lower() == want:
                    rename_map[c] = want
                    break
    df = df.rename(columns=rename_map)
    # Keep only needed cols
    keep = [c for c in ["ticker", "model", "f1", "precision", "recall", "auc"] if c in df.columns]
    df = df[keep].copy()
    # Coerce numerics
    for m in ["f1", "precision", "recall", "auc"]:
        if m in df.columns:
            df[m] = df[m].apply(safe_float)
    # Standardize model names a bit
    if "model" in df.columns:
        df["model"] = df["model"].str.upper().str.replace("1D", "1D", regex=False)
        df["model"] = df["model"].str.replace("CNN1D", "CNN1D").str.replace("BILSTM", "BiLSTM").str.replace("LSTM", "LSTM")
    return df

def load_threshold_rows(runs_dir):
    rows = []
    for run in glob.glob(os.path.join(runs_dir, "*")):
        if not os.path.isdir(run):
            continue
        sweep_path = os.path.join(run, "sweep_summary.json")
        if not os.path.exists(sweep_path):
            continue
        try:
            with open(sweep_path, "r") as f:
                js = json.load(f)
            # Expect keys like: ticker, model_family, threshold, f1, precision, recall (values for TEST at t*)
            # Our earlier sweeps printed val/test lines; some scripts also save JSON. Be robust:
            row = {
                "run_dir": run,
                "ticker": js.get("ticker", None),
                "model": js.get("model_family", None),
                "threshold": safe_float(js.get("threshold", np.nan)),
                "test_f1": safe_float(js.get("test_f1", np.nan)),
                "test_precision": safe_float(js.get("test_precision", np.nan)),
                "test_recall": safe_float(js.get("test_recall", np.nan)),
            }
            # Try to infer ticker/model from folder name if missing
            base = os.path.basename(run)
            if row["ticker"] is None:
                for t in ["MSFT", "TSLA", "GOOGL"]:
                    if t.lower() in base.lower():
                        row["ticker"] = t
                        break
            if row["model"] is None:
                if "cnn1d" in base.lower():
                    row["model"] = "CNN1D"
                elif "bilstm" in base.lower():
                    row["model"] = "BiLSTM"
                elif "lstm" in base.lower():
                    row["model"] = "LSTM"
            rows.append(row)
        except Exception:
            # Ignore malformed JSON
            pass
    return pd.DataFrame(rows)

def grouped_bar(ax, df, metric, title):
    # df: rows are (ticker, model, metric)
    tickers = df["ticker"].unique().tolist()
    models = df["model"].unique().tolist()
    models.sort(key=lambda x: {"LSTM":0, "BiLSTM":1, "CNN1D":2}.get(x, 99))
    x = np.arange(len(tickers))
    width = 0.22
    for i, m in enumerate(models):
        y = []
        for t in tickers:
            v = df.loc[(df["ticker"]==t) & (df["model"]==m), metric]
            y.append(v.values[0] if len(v)>0 else np.nan)
        ax.bar(x + i*width - width, y, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

# --------------------------
# 1) Load data
# --------------------------
final_df = load_final_comparison(FINAL_CSV)

# --------------------------
# 2) Save basic consolidated CSVs (optional)
# --------------------------
final_df.sort_values(["ticker","model"]).to_csv(os.path.join(OUT_DIR, "final_comparison_sorted.csv"), index=False)

# --------------------------
# 3) Charts from final_comparison.csv
# --------------------------
plt.figure(figsize=(8,5))
grouped_bar(plt.gca(), final_df, "f1", "Test F1 by Ticker × Model")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "final_f1_bar.png"), dpi=160)
plt.close()

plt.figure(figsize=(8,5))
grouped_bar(plt.gca(), final_df, "precision", "Test Precision by Ticker × Model")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "final_precision_bar.png"), dpi=160)
plt.close()

plt.figure(figsize=(8,5))
grouped_bar(plt.gca(), final_df, "recall", "Test Recall by Ticker × Model")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "final_recall_bar.png"), dpi=160)
plt.close()

if "auc" in final_df.columns and final_df["auc"].notna().any():
    plt.figure(figsize=(8,5))
    grouped_bar(plt.gca(), final_df, "auc", "Test AUC by Ticker × Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "final_auc_bar.png"), dpi=160)
    plt.close()

# --------------------------
# 4) Threshold table from sweeps
# --------------------------
th_df = load_threshold_rows(RUNS_DIR)
if len(th_df) > 0:
    # Clean model names
    th_df["model"] = th_df["model"].fillna("").str.replace("cnn1d","CNN1D").str.replace("bilstm","BiLSTM").str.replace("lstm","LSTM")
    # Save table
    th_df.sort_values(["ticker","model"]).to_csv(os.path.join(OUT_DIR, "threshold_table.csv"), index=False)

    # Plot thresholds (by ticker × model)
    # Keep rows with ticker/model/threshold available
    th_plot = th_df.dropna(subset=["ticker","model","threshold"]).copy()
    if len(th_plot) > 0:
        # Pivot for grouped bars
        tickers = sorted(th_plot["ticker"].dropna().unique().tolist())
        models = sorted(th_plot["model"].dropna().unique().tolist(), key=lambda x: {"LSTM":0,"BiLSTM":1,"CNN1D":2}.get(x, 99))
        x = np.arange(len(tickers))
        width = 0.22
        plt.figure(figsize=(8,5))
        for i, m in enumerate(models):
            y = []
            for t in tickers:
                rows = th_plot[(th_plot["ticker"]==t) & (th_plot["model"]==m)]
                if len(rows) == 0:
                    y.append(np.nan)
                else:
                    y.append(rows["threshold"].values[0])
            plt.bar(x + i*width - width, y, width, label=m)
        plt.xticks(x, tickers)
        plt.ylabel("Threshold (t*)")
        plt.title("Chosen Decision Threshold by Ticker × Model")
        plt.legend(frameon=False)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "thresholds_by_ticker.png"), dpi=160)
        plt.close()
else:
    # No sweep JSONs found; just note it.
    with open(os.path.join(OUT_DIR, "threshold_table.csv"), "w") as f:
        f.write("ticker,model,threshold,test_f1,test_precision,test_recall,run_dir\n")