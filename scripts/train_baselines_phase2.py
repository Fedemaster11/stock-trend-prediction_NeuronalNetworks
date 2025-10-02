#!/usr/bin/env python3
"""
Phase 2: Train baselines (LogReg, RF) with temporal splits.
Compares Price-only vs Price+Sentiment for horizons (1,5,10)d.

Inputs:
  features_enriched_schema.csv OR .parquet from fe_phase2_for_schema.py

Outputs:
  results_phase2.csv
"""

import argparse, warnings, os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

RANDOM_STATE = 42

PRICE_COLS_BASE = [
    "ret_1d","log_ret_1d","ma_5","ma_20","vol_20","adj_close_t1",
    # you can include raw OHLC/volume too if you like:
    "open","high","low","close","adj_close","volume"
]
# any column that *starts with* one of these will be considered sentiment-ish
SENT_PREFIXES = (
    "mean_sent","n_tweets","pos_share","neg_share",
    "sent_strength","pos_minus_neg",
    "mean_sent_r","n_tweets_r","sent_strength_r","pos_minus_neg_r",
    "_lag1"  # will match columns that end with lag1 when combined with startswith logic below
)

def pick_features(df: pd.DataFrame, with_sentiment: bool) -> list:
    cols = []
    for c in df.columns:
        if c in PRICE_COLS_BASE:
            cols.append(c)
    if with_sentiment:
        for c in df.columns:
            if (c.startswith(SENT_PREFIXES) or c.endswith("_lag1")):
                cols.append(c)
    # keep unique, stable
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    return cols

def temporal_split_one_ticker(dft: pd.DataFrame, test_frac=0.15, val_frac=0.15):
    dft = dft.sort_values("date").reset_index(drop=True)
    n = len(dft)
    n_test = int(np.floor(test_frac * n))
    n_val  = int(np.floor(val_frac  * (n - n_test)))
    n_train = n - n_val - n_test
    return dft.iloc[:n_train], dft.iloc[n_train:n_train+n_val], dft.iloc[n_train+n_val:]

def fit_eval_model(model, Xtr, ytr, Xva, yva, Xte, yte):
    model.fit(Xtr, ytr)
    p_va = model.predict_proba(Xva)[:,1]
    # tune threshold on val by F1
    thresholds = np.linspace(0.2, 0.8, 25)
    f1s = []
    for t in thresholds:
        f1s.append(f1_score(yva, (p_va>=t).astype(int), zero_division=0))
    best_t = float(thresholds[int(np.argmax(f1s))])

    p_te = model.predict_proba(Xte)[:,1]
    yhat = (p_te >= best_t).astype(int)

    return {
        "acc": accuracy_score(yte, yhat),
        "f1":  f1_score(yte, yhat, zero_division=0),
        "prec": precision_score(yte, yhat, zero_division=0),
        "rec": recall_score(yte, yhat, zero_division=0),
        "roc": roc_auc_score(yte, p_te) if len(np.unique(yte))>1 else np.nan,
        "thr": best_t
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="features_enriched_schema.parquet", help="Path to enriched dataset (.parquet or .csv)")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1,5,10], help="Horizons in days")
    args = ap.parse_args()

    # load
    # load
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input, parse_dates=["date"])
    # make dates naive (drop tz) for simplicity
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    else:
        df = pd.read_parquet(args.input)
    # force datetime and drop timezone to avoid dtype issues downstream
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
            
    # sanity: keep needed columns
    need = {"date","ticker"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing required columns {need - set(df.columns)}")

    results = []
    for ticker in sorted(df["ticker"].dropna().unique()):
        dft = df[df["ticker"]==ticker].dropna(subset=["date"]).copy()
        if len(dft) < 300:
            continue

        # temporal split once per ticker; reuse for all horizons and feature sets
        train, val, test = temporal_split_one_ticker(dft, test_frac=0.15, val_frac=0.15)

        for h in args.horizons:
            y_col = f"y_up_{h}d"
            if y_col not in dft.columns:
                # fallback to y_up_next for h==1 if alias not present
                if h==1 and "y_up_next" in dft.columns:
                    y_col = "y_up_next"
                else:
                    continue

            for with_sent in [False, True]:
                feat_cols = pick_features(dft, with_sentiment=with_sent)
                # drop rows that have NA in features or label
                tr = train.dropna(subset=feat_cols+[y_col]).copy()
                va =  val.dropna(subset=feat_cols+[y_col]).copy()
                te = test.dropna(subset=feat_cols+[y_col]).copy()
                if min(len(tr), len(va), len(te)) < 50:
                    continue

                # data
                Xtr, ytr = tr[feat_cols].values, tr[y_col].astype(int).values
                Xva, yva = va[feat_cols].values, va[y_col].astype(int).values
                Xte, yte = te[feat_cols].values, te[y_col].astype(int).values

                # Logistic Regression (scaled)
                scaler = StandardScaler()
                Xtr_lr = scaler.fit_transform(Xtr)
                Xva_lr = scaler.transform(Xva)
                Xte_lr = scaler.transform(Xte)
                lr = LogisticRegression(max_iter=500, solver="lbfgs", random_state=RANDOM_STATE)
                res_lr = fit_eval_model(lr, Xtr_lr, ytr, Xva_lr, yva, Xte_lr, yte)
                results.append({
                    "ticker": ticker, "horizon_d": h,
                    "feature_set": "price+sent" if with_sent else "price",
                    "model": "logreg", **res_lr,
                    "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                    "features_used": len(feat_cols)
                })

                # Random Forest (no scaling)
                rf = RandomForestClassifier(
                    n_estimators=400, max_depth=None,
                    min_samples_split=4, min_samples_leaf=1,
                    random_state=RANDOM_STATE, n_jobs=-1
                )
                res_rf = fit_eval_model(rf, Xtr, ytr, Xva, yva, Xte, yte)
                results.append({
                    "ticker": ticker, "horizon_d": h,
                    "feature_set": "price+sent" if with_sent else "price",
                    "model": "rf", **res_rf,
                    "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                    "features_used": len(feat_cols)
                })

    out = pd.DataFrame(results).sort_values(["ticker","horizon_d","feature_set","model"])
    out.to_csv("results_phase2.csv", index=False)
    print(out.head(20))
    print("\nSaved → results_phase2.csv")

if __name__ == "__main__":
    main()
