# baseline_logreg_msft.py
# Phase 3 Baseline: Logistic Regression on MSFT with AUC tuning + balanced threshold

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import optuna
from joblib import dump
import argparse


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_auc_score, balanced_accuracy_score
)

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/processed/merged_dataset.csv"   # point to your file (e.g., merged_dataset_with_sent.csv)
TICKER = "MSFT"
N_TRIALS = 50
RANDOM_SEED = 42

# -----------------------------
# Logging helper
# -----------------------------
def log_result(ticker, model, featureset, acc, f1w, prec, rec, roc, cm, best_params):
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    row = pd.DataFrame([{
        "ticker": ticker, "model": model, "feature_set": featureset,
        "accuracy": acc, "f1_weighted": f1w, "precision_weighted": prec,
        "recall_weighted": rec, "roc_auc": roc,
        "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        "best_params": json.dumps(best_params)
    }])
    out = "results/baselines.csv"
    row.to_csv(out, mode="a", header=not os.path.exists(out), index=False)

# -----------------------------
# Feature engineering
# -----------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build RSI(14) and MACD(12,26,9) features."""
    try:
        import pandas_ta as ta
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"]).rename(columns=str.lower)
        line_cols = [c for c in macd.columns if c.startswith("macd_")]
        hist_cols = [c for c in macd.columns if c.startswith("macdh")]
        if not line_cols or not hist_cols:
            raise RuntimeError(f"Unexpected MACD cols: {list(macd.columns)}")
        df["macd_line"] = macd[line_cols[0]].values
        df["macd_hist"] = macd[hist_cols[0]].values
    except Exception as e:
        # Fallback: manual RSI + MACD
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
        print(f"[create_features] Fallback RSI/MACD used due to: {e}")

    return df.dropna().reset_index(drop=True)

# -----------------------------
# Load & prepare data
# -----------------------------
def load_msft_dataframe(path: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required_base = {
        "date", "ticker", "open", "high", "low", "close", "adj_close",
        "volume", "ma_5", "ma_20", "vol_20", "log_ret_1d", "ret_1d",
        "y_up_next", "mean_sent", "pos_share", "neg_share", "n_tweets"
    }
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df = df[df["ticker"] == ticker].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = create_features(df)
    return df

# -----------------------------
# Threshold selection (validation)
# -----------------------------
def pick_threshold(y_val, p_val, metric="balanced_accuracy"):
    best_t, best_m = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_hat = (p_val >= t).astype(int)
        # skip degenerate thresholds (all-0 or all-1)
        if y_hat.min() == y_hat.max():
            continue
        if metric == "f1_macro":
            m = f1_score(y_val, y_hat, average="macro", zero_division=0)
        elif metric == "balanced_accuracy":
            m = balanced_accuracy_score(y_val, y_hat)
        else:
            m = accuracy_score(y_val, y_hat)
        if m > best_m:
            best_t, best_m = t, m
    if best_m < 0:
        # fallback to 0.5 if everything degenerated
        best_t = 0.5
        best_m = balanced_accuracy_score(y_val, (p_val >= 0.5).astype(int))
    return float(best_t), float(best_m)

# -----------------------------
# Optuna objective (AUC on val)
# -----------------------------
def objective(trial, X_train, X_val, y_train, y_val):
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)  # tamed range
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xv  = scaler.transform(X_val)

    model = LogisticRegression(
        C=C, solver=solver, penalty="l2",
        max_iter=1000, random_state=RANDOM_SEED,
        class_weight=class_weight
    )
    model.fit(Xtr, y_train)
    p_val = model.predict_proba(Xv)[:, 1]
    return roc_auc_score(y_val, p_val)

# -----------------------------
# Train & evaluate best model
# -----------------------------
def train_and_eval_best_model(
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test,
    price_features, sentiment_features, label: str
):
    feats = list(price_features)
    if "Sentiment" in label:
        feats = feats + sentiment_features

    Xtr, Xv, Xte = X_train_df[feats], X_val_df[feats], X_test_df[feats]

    # ---- Hyperparameter search (AUC on val) ----
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(lambda trial: objective(trial, Xtr, Xv, y_train, y_val),
                   n_trials=N_TRIALS)
    best_params = study.best_params
    print(f"\nBest hyperparameters for {label}: {best_params}")

    # ---- Threshold on val (train-only model) ----
    scaler_tr = StandardScaler().fit(Xtr)
    Xtr_sc = scaler_tr.transform(Xtr)
    Xv_sc  = scaler_tr.transform(Xv)

    tmp = LogisticRegression(
        C=best_params["C"], solver=best_params["solver"], penalty="l2",
        max_iter=1000, random_state=RANDOM_SEED, class_weight=best_params["class_weight"]
    ).fit(Xtr_sc, y_train)
    p_val = tmp.predict_proba(Xv_sc)[:, 1]
    t_star, val_balacc = pick_threshold(y_val, p_val, metric="balanced_accuracy")
    n0, n1 = (p_val < t_star).sum(), (p_val >= t_star).sum()
    print(f"[val] threshold={t_star:.3f} · preds 0={n0}, 1={n1} · balanced_acc={val_balacc:.3f}")

    # ---- Final fit on TRAIN+VAL, evaluate on TEST with t* ----
    scaler = StandardScaler().fit(pd.concat([Xtr, Xv], axis=0))
    Xtrv_sc = scaler.transform(pd.concat([Xtr, Xv], axis=0))
    Xte_sc  = scaler.transform(Xte)
    y_trv   = pd.concat([y_train, y_val], axis=0)

    best = LogisticRegression(
        C=best_params["C"], solver=best_params["solver"], penalty="l2",
        max_iter=1000, random_state=RANDOM_SEED, class_weight=best_params["class_weight"]
    )
    best.fit(Xtrv_sc, y_trv)

    # Predict with tuned threshold
    p_test = best.predict_proba(Xte_sc)[:, 1]
    y_pred = (p_test >= t_star).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    roc = roc_auc_score(y_test, p_test)  # AUC uses probabilities
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== FINAL RESULTS on Test Set ({label}) =====")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1-weighted: {f1w:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"ROC AUC:     {roc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    best_params_logged = {**best_params, "threshold": round(t_star, 3)}
    log_result(TICKER, "LogReg", label, acc, f1w, prec, rec, roc, cm, best_params_logged)

    safe_label = label.replace(" ", "_").lower()
    dump(best,   f"models/logreg_{TICKER.lower()}_{safe_label}.joblib")
    dump(scaler, f"models/scaler_{TICKER.lower()}_{safe_label}.joblib")

# -----------------------------
# Main
# -----------------------------
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-date", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    # Load MSFT dataset
    df = load_msft_dataframe(DATA_PATH, TICKER)

    # Apply min-date filter if provided
    if args.min_date is not None:
        df = df[df["date"] >= pd.to_datetime(args.min_date)].reset_index(drop=True)
        print(f"Filtered dataset from {args.min_date}: {len(df)} rows left")

    target = "y_up_next"
    price_features = [
        "open", "high", "low", "close", "adj_close",
        "volume", "ma_5", "ma_20", "vol_20",
        "log_ret_1d", "ret_1d", "rsi", "macd_line", "macd_hist"
    ]
    sentiment_features = ["mean_sent", "pos_share", "neg_share", "n_tweets"]

    n = len(df)
    train_end = int(0.7 * n)
    val_end   = int(0.85 * n)

    X_full_df = df[price_features + sentiment_features].copy()
    y = df[target].copy()

    X_train_df = X_full_df.iloc[:train_end]
    X_val_df   = X_full_df.iloc[train_end:val_end]
    X_test_df  = X_full_df.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val   = y.iloc[train_end:val_end]
    y_test  = y.iloc[val_end:]

    print(f"Dataset (ticker={TICKER}) sizes → "
          f"train: {len(X_train_df)}, val: {len(X_val_df)}, test: {len(X_test_df)}")

    # Run baselines
    train_and_eval_best_model(
        X_train_df, X_val_df, X_test_df, y_train, y_val, y_test,
        price_features, sentiment_features, label="Price-only"
    )
    train_and_eval_best_model(
        X_train_df, X_val_df, X_test_df, y_train, y_val, y_test,
        price_features, sentiment_features, label="Price + Sentiment"
    )


if __name__ == "__main__":
    main()
