import warnings
warnings.filterwarnings("ignore")

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import optuna
from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_auc_score
)

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/processed/merged_dataset.csv"
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
    """
    Build RSI(14) and MACD(12,26,9) features.
    Uses pandas_ta if available; fallback to pure-pandas if needed.
    """
    try:
        import pandas_ta as ta
        # RSI(14)
        df["rsi"] = ta.rsi(df["close"], length=14)

        # MACD(12,26,9)
        macd = ta.macd(df["close"])
        macd = macd.rename(columns=str.lower)
        macd_line_cols = [c for c in macd.columns if c.startswith("macd_")]
        macd_hist_cols = [c for c in macd.columns if c.startswith("macdh")]
        if not macd_line_cols or not macd_hist_cols:
            raise RuntimeError(f"Unexpected MACD columns: {list(macd.columns)}")
        df["macd_line"] = macd[macd_line_cols[0]].values
        df["macd_hist"] = macd[macd_hist_cols[0]].values

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

        print(f"[create_features] Used fallback RSI/MACD due to: {e}")

    return df.dropna().reset_index(drop=True)

# -----------------------------
# Load & prepare data
# -----------------------------
def load_dataframe(path: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required_base = {
        "date", "ticker", "open", "high", "low", "close", "adj_close",
        "volume", "ma_5", "ma_20", "vol_20", "log_ret_1d", "ret_1d",
        "y_up_next", "mean_sent", "pos_share", "neg_share"
    }
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df = df[df["ticker"] == ticker].copy()
    if df.empty:
        raise ValueError(f"No rows found for ticker '{ticker}' in {path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = create_features(df)
    return df

# -----------------------------
# Optuna objective
# -----------------------------
def objective(trial, X_train, X_val, y_train, y_val):
    C = trial.suggest_float("C", 1e-5, 1e2, log=True)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    # Fix: Check for empty data before fitting scaler
    if X_train.empty:
        return 0 # Return a score of 0 to fail the trial gracefully

    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)

    model = LogisticRegression(
        C=C, solver=solver, penalty="l2",
        max_iter=1000, random_state=RANDOM_SEED,
        class_weight=class_weight
    )
    model.fit(X_train_sc, y_train)
    y_val_pred = model.predict(X_val_sc)
    return f1_score(y_val, y_val_pred, average="weighted")

# -----------------------------
# Train & evaluate best model
# -----------------------------
def train_and_eval_best_model(
    ticker_sym: str,
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test,
    price_features, sentiment_features, label: str,
    n_trials: int
):
    feats = list(price_features)
    if "Sentiment" in label:
        feats = feats + sentiment_features

    Xtr, Xv, Xte = X_train_df[feats], X_val_df[feats], X_test_df[feats]
    
    # Fix: New check for empty splits before starting optimization
    if Xtr.empty or Xv.empty or Xte.empty:
        print(f"Skipping {ticker_sym} · {label}: one or more splits are empty.")
        return

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(lambda trial: objective(trial, Xtr, Xv, y_train, y_val),
                   n_trials=n_trials)

    print(f"\nBest hyperparameters for {ticker_sym} · {label}: {study.best_params}")

    scaler = StandardScaler().fit(pd.concat([Xtr, Xv], axis=0))
    Xtrv_sc = scaler.transform(pd.concat([Xtr, Xv], axis=0))
    Xte_sc = scaler.transform(Xte)
    y_trv = pd.concat([y_train, y_val], axis=0)

    best = LogisticRegression(
        C=study.best_params["C"],
        solver=study.best_params["solver"],
        penalty="l2",
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight=study.best_params["class_weight"]
    )
    best.fit(Xtrv_sc, y_trv)

    y_pred = best.predict(Xte_sc)
    y_prob = best.predict_proba(Xte_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== FINAL RESULTS on Test Set ({ticker_sym} · {label}) =====")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1-weighted: {f1w:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"ROC AUC:     {roc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    log_result(ticker_sym, "LogReg", label, acc, f1w, prec, rec, roc, cm, study.best_params)

    safe_ticker = re.sub(r"[^a-z0-9]+", "", ticker_sym.lower())
    safe_label  = re.sub(r"[^a-z0-9]+", "_", label.lower())

    dump(best,   f"models/logreg_{safe_ticker}_{safe_label}.joblib")
    dump(scaler, f"models/scaler_{safe_ticker}_{safe_label}.joblib")

# -----------------------------
# Main
# -----------------------------
def main():
    np.random.seed(RANDOM_SEED)

    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="MSFT,PLTR,RHM.DE",
                    help="Comma-separated tickers, e.g. MSFT,PLTR,RHM.DE")
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help="Optuna trials per feature set")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    n_trials = int(args.trials)

    target = "y_up_next"
    price_features = [
        "open", "high", "low", "close", "adj_close",
        "volume", "ma_5", "ma_20", "vol_20",
        "log_ret_1d", "ret_1d", "rsi", "macd_line", "macd_hist"
    ]
    sentiment_features = ["mean_sent", "pos_share", "neg_share"]

    for tk in tickers:
        print(f"\n========== TICKER: {tk} ==========")
        try:
            df = load_dataframe(DATA_PATH, tk)
        except ValueError as e:
            print(e)
            continue

        required_cols = set([target] + price_features + sentiment_features)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns after feature engineering for {tk}: {sorted(missing)}")

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

        print(f"Dataset (ticker={tk}) → train: {len(X_train_df)}, val: {len(X_val_df)}, test: {len(X_test_df)}")

        train_and_eval_best_model(
            tk, X_train_df, X_val_df, X_test_df, y_train, y_val, y_test,
            price_features, sentiment_features, label="Price-only", n_trials=n_trials
        )
        train_and_eval_best_model(
            tk, X_train_df, X_val_df, X_test_df, y_train, y_val, y_test,
            price_features, sentiment_features, label="Price + Sentiment", n_trials=n_trials
        )

if __name__ == "__main__":
    main()
    