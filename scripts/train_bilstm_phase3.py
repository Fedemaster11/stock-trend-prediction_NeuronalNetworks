#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BiLSTM sequence classifier for stock movement prediction (Phase 3C).

- Uses the same NPZ format produced by make_sequences_phase3.py
- Early-stops on validation F1
- Saves best checkpoint + per-epoch CSV + final_summary.json
- Default is bidirectional=True

Example (PowerShell, one line):
.\.venv\Scripts\python.exe .\scripts\train_bilstm_phase3.py --npz .\data\processed\seq_MSFT_h10_L60_MB.npz --epochs 60 --batch 128 --hidden 48 --layers 1 --dropout 0.5 --lr 2e-4 --weight-decay 1e-4 --patience 10
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Reproducibility & utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def safe_roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def to_device(batch, device):
    return tuple(t.to(device) for t in batch)


# --------------
# Model (BiLSTM)
# --------------
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 48,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
        fc_hidden: int = 0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
            batch_first=True,
        )

        out_dim = hidden_size * self.num_directions

        if fc_hidden and fc_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(out_dim, fc_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden, 1),
            )
        else:
            self.head = nn.Linear(out_dim, 1)

    def forward(self, x):
        # x: (B, L, F)
        out, _ = self.lstm(x)          # (B, L, out_dim)
        last = out[:, -1, :]           # (B, out_dim)
        logits = self.head(last).squeeze(-1)  # (B,)
        return logits


# ----------------------------
# Train / eval loops
# ----------------------------
def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses = []
    y_true_all, y_prob_all = [], []

    for batch in loader:
        Xb, yb = to_device(batch, device)
        logits = model(Xb)
        loss = criterion(logits, yb)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(loss.item())

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            y_true_all.append(yb.detach().cpu().numpy())
            y_prob_all.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all) if len(y_true_all) else np.array([])
    y_prob = np.concatenate(y_prob_all) if len(y_prob_all) else np.array([])
    y_pred = (y_prob >= 0.5).astype(int) if y_prob.size else np.array([])

    if y_true.size:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        auc = safe_roc_auc(y_true, y_prob)
    else:
        acc = prec = rec = f1 = auc = float("nan")

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
    }


def compute_pos_weight(y_train: np.ndarray):
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    if pos == 0:
        return 1.0
    return max(neg / max(pos, 1), 1.0)


# --------------
# Main
# --------------
def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM on sequence NPZ.")
    # Data / IO
    parser.add_argument("--npz", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: ./runs/bilstm_<timestamp>)")

    # Model
    parser.add_argument("--hidden", type=int, default=48, help="LSTM hidden size")
    parser.add_argument("--layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument("--fc-hidden", type=int, default=0, help="Optional FC hidden size after LSTM (0 disables)")

    # Training
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42)

    # Device
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    args = parser.parse_args()
    set_seed(args.seed)

    # Out dir
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join("runs", f"bilstm_{timestamp}")
    ensure_out_dir(args.out_dir)

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    # Load data
    print(f"[INFO] Loading: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.float32)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.float32)

    assert X_train.ndim == 3, "Expected X_train shape (N, L, F)"
    seq_len   = X_train.shape[1]
    input_dim = X_train.shape[2]
    print(f"[INFO] Shapes: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
    print(f"[INFO] Sequence length: {seq_len}, Features: {input_dim}")

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, drop_last=False)

    # Model
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
        bidirectional=True,          # <— BiLSTM fixed
        dropout=args.dropout,
        fc_hidden=args.fc_hidden,
    ).to(device)

    # Loss (weighted)
    pos_weight_val = compute_pos_weight(y_train)
    print(f"[INFO] Class balance (train): pos={(y_train==1).sum()}, neg={(y_train==0).sum()}, pos_weight={pos_weight_val:.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(args.patience // 2, 2))

    # Train loop with early stopping (val F1)
    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics   = run_epoch(model, val_loader,   device, criterion, optimizer=None)

        scheduler.step(val_metrics["f1"])

        row = {"epoch": epoch,
               **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()},
               "lr": optimizer.param_groups[0]["lr"]}
        log_rows.append(row)

        print(f"[E{epoch:03d}] train: loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f} auc={train_metrics['auc']:.4f} | "
              f"val: loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping on val F1
        if val_metrics["f1"] > best_val_f1 + 1e-6:
            best_val_f1 = val_metrics["f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
                "input_dim": input_dim,
                "seq_len": seq_len,
            }
            torch.save(best_state, os.path.join(args.out_dir, "best_model.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break

        # rolling CSV log
        pd.DataFrame(log_rows).to_csv(os.path.join(args.out_dir, "metrics_epoch_log.csv"), index=False)

    # Load best before final eval
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    # Final metrics (fixed 0.5 just for logging; use sweep for best t*)
    train_final = run_epoch(model, train_loader, device, criterion, optimizer=None)
    val_final   = run_epoch(model, val_loader,   device, criterion, optimizer=None)
    test_final  = run_epoch(model, test_loader,  device, criterion, optimizer=None)

    # Confusion matrix on test at 0.5
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            Xb, yb = to_device(batch, device)
            logits = model(Xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob_all.append(probs)
            y_true_all.append(yb.detach().cpu().numpy())
    y_true = np.concatenate(y_true_all) if len(y_true_all) else np.array([])
    y_pred = (np.concatenate(y_prob_all) >= 0.5).astype(int) if len(y_prob_all) else np.array([])
    cm = confusion_matrix(y_true, y_pred).tolist() if y_true.size else [[0,0],[0,0]]

    summary = {
        "args": vars(args),
        "shapes": {"X_train": list(X_train.shape), "X_val": list(X_val.shape), "X_test": list(X_test.shape)},
        "final_metrics": {"train": train_final, "val": val_final, "test": test_final},
        "confusion_matrix_test": cm,
        "best_val_f1": best_val_f1,
    }

    with open(os.path.join(args.out_dir, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SUMMARY]")
    print(json.dumps(summary["final_metrics"], indent=2))
    print(f"\nSaved best model and logs to: {args.out_dir}")


if __name__ == "__main__":
    main()
