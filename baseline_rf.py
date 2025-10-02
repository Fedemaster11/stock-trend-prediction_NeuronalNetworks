import warnings; warnings.filterwarnings("ignore")

import os, re, json, argparse, numpy as np, pandas as pd, optuna
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report, balanced_accuracy_score
)
from sklearn.inspection import permutation_importance

DATA_PATH   = "data/processed/merged_dataset.csv"
N_TRIALS    = 50
RANDOM_SEED = 42

# ---------------------------
# utils
# ---------------------------
def log_result(ticker, model, featureset, acc, f1w, prec, rec, roc, cm, best_params):
    os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)
    row = pd.DataFrame([{
        "ticker": ticker, "model": model, "feature_set": featureset,
        "accuracy": acc, "f1_weighted": f1w, "precision_weighted": prec,
        "recall_weighted": rec, "roc_auc": roc,
        "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        "best_params": json.dumps(best_params)
    }])
    out = "results/baselines.csv"
    row.to_csv(out, mode="a", header=not os.path.exists(out), index=False)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # RSI + MACD
    try:
        import pandas_ta as ta
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"]).rename(columns=str.lower)
        line_cols = [c for c in macd.columns if c.startswith("macd_")]
        hist_cols = [c for c in macd.columns if c.startswith("macdh")]
        if not line_cols or not hist_cols: raise RuntimeError(list(macd.columns))
        df["macd_line"] = macd[line_cols[0]].values
        df["macd_hist"] = macd[hist_cols[0]].values
    except Exception as e:
        delta = df["close"].diff()
        up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = up / (down + 1e-12)
        df["rsi"] = 100 - (100 / (1 + rs))
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_line"] = macd_line
        df["macd_hist"] = macd_line - signal
        print(f"[create_features] Fallback RSI/MACD due to: {e}")

    # sentiment enrichments (mask by has_tweets to avoid smearing zeros)
    df["has_tweets"] = (df["n_tweets"] > 0).astype(int)
    df["pos_minus_neg"] = df["pos_share"] - df["neg_share"]

    sent = df["mean_sent"].where(df["has_tweets"] == 1)
    df["mean_sent_r3"]  = sent.rolling(3,  min_periods=1).mean().fillna(0.0)
    df["mean_sent_r5"]  = sent.rolling(5,  min_periods=1).mean().fillna(0.0)
    df["mean_sent_r10"] = sent.rolling(10, min_periods=1).mean().fillna(0.0)

    pmn = df["pos_minus_neg"].where(df["has_tweets"] == 1)
    df["pos_minus_neg_r5"] = pmn.rolling(5, min_periods=1).mean().fillna(0.0)

    # volume-weighted sentiment strength (diminishing returns with pow)
    df["sent_strength"] = df["mean_sent"] * (df["n_tweets"].clip(lower=0) + 1).pow(0.3)

    return df.dropna().reset_index(drop=True)

def load_dataframe(path: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required = {
        "date","ticker","open","high","low","close","adj_close","volume",
        "ma_5","ma_20","vol_20","log_ret_1d","ret_1d","y_up_next",
        "mean_sent","pos_share","neg_share","n_tweets"
    }
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
    df = df[df["ticker"] == ticker].copy()
    if df.empty: raise ValueError(f"No rows for ticker '{ticker}'")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return create_features(df)

def pick_threshold(y_val, p_val):
    best_t, best_m = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_hat = (p_val >= t).astype(int)
        if y_hat.min() == y_hat.max():  # skip all-one/zero
            continue
        m = balanced_accuracy_score(y_val, y_hat)
        if m > best_m:
            best_t, best_m = t, m
    if best_m < 0:
        best_t = 0.5
        best_m = balanced_accuracy_score(y_val, (p_val>=0.5).astype(int))
    return float(best_t), float(best_m)

# ---------------------------
# optuna objective
# ---------------------------
def make_objective(Xtr, Xv, ytr, yv):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(5, 26))),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt","log2", 0.4, 0.6, 0.8, 1.0]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**params)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xv)[:, 1]
        try:
            return roc_auc_score(yv, p)
        except Exception:
            return 0.5
    return objective

# ---------------------------
# train/eval
# ---------------------------
def train_eval(ticker, label, df, price_feats, sent_feats):
    feats = list(price_feats) + (sent_feats if "Sentiment" in label else [])
    target = "y_up_next"

    X = df[feats].copy()
    y = df[target].astype(int).copy()

    n = len(df)
    tr = int(0.70 * n); va = int(0.85 * n)
    Xtr, Xv, Xte = X.iloc[:tr], X.iloc[tr:va], X.iloc[va:]
    ytr, yv, yte = y.iloc[:tr], y.iloc[tr:va], y.iloc[va:]

    print(f"Dataset (ticker={ticker} · {label}) → train: {len(Xtr)}, val: {len(Xv)}, test: {len(Xte)}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(make_objective(Xtr, Xv, ytr, yv), n_trials=N_TRIALS)
    best = study.best_params
    print(f"Best params ({label}): {best}")

    # threshold from train-only fit
    clf_tr = RandomForestClassifier(**{**best, "random_state": RANDOM_SEED, "n_jobs": -1})
    clf_tr.fit(Xtr, ytr)
    p_val = clf_tr.predict_proba(Xv)[:, 1]
    t_star, _ = pick_threshold(yv, p_val)

    # final fit on train+val
    clf = RandomForestClassifier(**{**best, "random_state": RANDOM_SEED, "n_jobs": -1})
    clf.fit(pd.concat([Xtr, Xv]), pd.concat([ytr, yv]))

    p_test = clf.predict_proba(Xte)[:, 1]
    y_pred = (p_test >= t_star).astype(int)

    acc  = accuracy_score(yte, y_pred)
    f1w  = f1_score(yte, y_pred, average="weighted")
    prec = precision_score(yte, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(yte, y_pred, average="weighted", zero_division=0)
    roc  = roc_auc_score(yte, p_test)
    cm   = confusion_matrix(yte, y_pred)

    print(f"\n===== FINAL RESULTS on Test Set ({ticker} · {label}) =====")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1-weighted: {f1w:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"ROC AUC:     {roc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(yte, y_pred, zero_division=0))

    # permutation importance (quick, guarded)
    try:
        imp = permutation_importance(clf, Xte, yte, n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)
        order = np.argsort(imp.importances_mean)[::-1][:12]
        print("\nTop-12 permutation importances:")
        for idx in order:
            print(f"  {X.columns[idx]:<20s}  {imp.importances_mean[idx]:.6f}")
    except Exception as e:
        print(f"[perm_importance] skipped due to: {e}")

    log_result(ticker, "RandomForest", label, acc, f1w, prec, rec, roc, cm, {**best, "threshold": round(t_star,3)})

    safe_tk  = re.sub(r"[^a-z0-9]+", "", ticker.lower())
    safe_lbl  = re.sub(r"[^a-z0-9]+", "_", label.lower())
    dump(clf, f"models/rf_{safe_tk}_{safe_lbl}.joblib")

# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="MSFT,TSLA,GOOGL", help="Comma-separated tickers")
    ap.add_argument("--trials", type=int, default=N_TRIALS, help="Optuna trials")
    ap.add_argument("--min-date", default=None, help="e.g. 2021-09-01")  # New argument
    args = ap.parse_args()

    price_feats = ["open","high","low","close","adj_close","volume",
                   "ma_5","ma_20","vol_20","log_ret_1d","ret_1d",
                   "rsi","macd_line","macd_hist"]
    sent_feats  = ["mean_sent","pos_share","neg_share","n_tweets",
                   "has_tweets","mean_sent_r3","mean_sent_r5","mean_sent_r10",
                   "pos_minus_neg","pos_minus_neg_r5","sent_strength"]

    for tk in [t.strip() for t in args.tickers.split(",") if t.strip()]:
        print(f"\n========== TICKER: {tk} ==========")
        try:
            df = load_dataframe(DATA_PATH, tk)
        except Exception as e:
            print(e); continue

        run_suffix = ""
        if args.min_date:
            cut = pd.to_datetime(args.min_date)
            df = df[df["date"] >= cut].reset_index(drop=True)
            run_suffix = f" (post{cut.date()})"
            if len(df) < 200:
                print(f"Warning: Dataset for {tk} after {cut.date()} is too small ({len(df)} rows). Skipping.")
                continue

        # Price-only
        train_eval(tk, "Price-only" + run_suffix, df, price_feats, [])

        # Price + Sentiment
        train_eval(tk, "Price + Sentiment" + run_suffix, df, price_feats, sent_feats)

if __name__ == "__main__":
    main()