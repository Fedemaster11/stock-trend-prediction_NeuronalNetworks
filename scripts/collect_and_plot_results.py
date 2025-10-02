#!/usr/bin/env python
import os, re, json, argparse
import pandas as pd
import matplotlib.pyplot as plt

def infer_ticker(npz_basename: str) -> str:
    m = re.search(r"seq_([A-Z\.]+)_h(\d+)_L(\d+)_MB\.npz", npz_basename)
    if not m:
        return "UNKNOWN"
    return m.group(1)

def infer_h_L(npz_basename: str):
    m = re.search(r"_h(\d+)_L(\d+)_MB\.npz", npz_basename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs", help="Directory with run subfolders")
    ap.add_argument("--out-csv", default="results/final_comparison.csv")
    ap.add_argument("--out-png", default="results/final_f1_bar.png")
    args = ap.parse_args()

    rows = []
    for sub in os.listdir(args.runs_dir):
        run_dir = os.path.join(args.runs_dir, sub)
        if not os.path.isdir(run_dir):
            continue
        # accept both families
        if not (sub.lower().startswith("lstm_") or sub.lower().startswith("cnn1d")):
            continue
        ss = os.path.join(run_dir, "sweep_summary.json")
        if not os.path.exists(ss):
            continue
        try:
            with open(ss, "r") as f:
                d = json.load(f)
        except Exception:
            continue

        ticker = infer_ticker(d.get("npz_basename", ""))
        h, L = infer_h_L(d.get("npz_basename", ""))
        rows.append({
            "model": d.get("model_family", "unknown"),
            "run_dir": run_dir,
            "ticker": ticker,
            "horizon": h,
            "seq_len": L,
            "val_auc": d.get("val", {}).get("auc"),
            "val_f1": d.get("val", {}).get("f1"),
            "val_precision": d.get("val", {}).get("precision"),
            "val_recall": d.get("val", {}).get("recall"),
            "test_auc": d.get("test", {}).get("auc"),
            "test_f1": d.get("test", {}).get("f1"),
            "test_precision": d.get("test", {}).get("precision"),
            "test_recall": d.get("test", {}).get("recall"),
            "best_threshold": d.get("val", {}).get("best_threshold"),
        })

    if not rows:
        print("[WARN] No sweep_summary.json files found. Run sweeps with --save first.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[SAVED] {args.out_csv}")
    print(df.sort_values(["ticker","model"]))

    # Plot: F1 on TEST by ticker & model
    piv = df.pivot_table(index="ticker", columns="model", values="test_f1", aggfunc="max")
    ax = piv.plot(kind="bar")
    ax.set_ylabel("Test F1")
    ax.set_title("Test F1 by Ticker and Model")
    ax.legend(title="Model")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, dpi=150)
    print(f"[SAVED] {args.out_png}")

if __name__ == "__main__":
    main()
