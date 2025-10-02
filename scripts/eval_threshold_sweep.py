#!/usr/bin/env python
import argparse, os, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

# ===== Models (must match trainers) =====
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1,
                 bidirectional=False, dropout=0.0, fc_hidden=0):
        super().__init__()
        self.num_dirs = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=bidirectional,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True
        )
        out_dim = hidden_size * self.num_dirs
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
        out, _ = self.lstm(x)      # (B, L, out_dim)
        last = out[:, -1, :]       # (B, out_dim)
        return self.head(last).squeeze(-1)

class CNN1D(nn.Module):
    def __init__(self, in_feat, seq_len, channels=32, k=5, dropout=0.5, fc_hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_feat, channels, kernel_size=k, padding=k//2)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, padding=k//2)
        self.bn2   = nn.BatchNorm1d(channels)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.drop  = nn.Dropout(dropout)
        if fc_hidden and fc_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(channels, fc_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(fc_hidden, 1)
            )
        else:
            self.head = nn.Linear(channels, 1)

    def forward(self, x):           # x: (B, L, F)
        x = x.transpose(1, 2)       # -> (B, F, L)
        x = self.drop(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(x).squeeze(-1)    # (B, C)
        return self.head(x).squeeze(-1)

# ===== Utilities =====
def safe_auc(y, p):
    try: return roc_auc_score(y, p)
    except: return float("nan")

def prf1(y, yhat):
    tp = ((y==1)&(yhat==1)).sum()
    fp = ((y==0)&(yhat==1)).sum()
    fn = ((y==1)&(yhat==0)).sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    Xtr, ytr = data["X_train"].astype(np.float32), data["y_train"].astype(int)
    Xva, yva = data["X_val"].astype(np.float32),   data["y_val"].astype(int)
    Xte, yte = data["X_test"].astype(np.float32),  data["y_test"].astype(int)
    return (Xtr,ytr),(Xva,yva),(Xte,yte)

def predict_probs(model, X, batch=1024, device="cpu"):
    ps = []
    model.to(device).eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            logits = model(xb)
            ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(ps) if ps else np.array([])

def load_run(run_dir):
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    sd   = ckpt["model_state_dict"]
    input_dim = ckpt.get("input_dim")
    seq_len   = ckpt.get("seq_len")

    # Determine family robustly
    name = os.path.basename(run_dir).lower()
    family = args.get("model_family")  # optional if you set it in trainers
    if family is None:
        if "cnn1d" in name:
            family = "cnn1d"
        elif args.get("bidir", False) or "bilstm" in name:
            family = "bilstm"
        else:
            family = "lstm"

    if family == "cnn1d":
        model = CNN1D(
            in_feat=input_dim, seq_len=seq_len,
            channels=args.get("channels", 32),
            k=args.get("kernel", 5),
            dropout=args.get("dropout", 0.5),
            fc_hidden=args.get("fc_hidden", 64),
        )
    else:
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_size=args.get("hidden", 64),
            num_layers=args.get("layers", 1),
            bidirectional=args.get("bidir", family=="bilstm"),
            dropout=args.get("dropout", 0.0),
            fc_hidden=args.get("fc_hidden", 0),
        )

    # Load exact weights
    model.load_state_dict(sd)
    return model, args, family, input_dim, seq_len

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="NPZ used for training")
    ap.add_argument("--run-dir", required=True, help="Folder containing best_model.pt")
    ap.add_argument("--metric", default="f1", choices=["f1","recall","precision"])
    ap.add_argument("--tmin", type=float, default=0.05)
    ap.add_argument("--tmax", type=float, default=0.95)
    ap.add_argument("--nsteps", type=int, default=181)
    ap.add_argument("--precision-min", type=float, default=0.0,
                    help="Optional precision floor on VAL when choosing t*")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    args = ap.parse_args()

    model, margs, family, input_dim, seq_len = load_run(args.run_dir)
    (_, _), (Xv, yv), (Xt, yt) = load_npz(args.npz)

    pv = predict_probs(model, Xv, device=args.device)
    pt = predict_probs(model, Xt, device=args.device)

    print(f"[VAL] AUC={safe_auc(yv, pv):.3f}")
    print(f"[TEST] AUC={safe_auc(yt, pt):.3f}")

    ts = np.linspace(args.tmin, args.tmax, args.nsteps)
    best = None
    for t in ts:
        yhat_v = (pv >= t).astype(int)
        p,r,f = prf1(yv, yhat_v)
        if p < args.precision_min:
            continue
        score = {"f1":f,"recall":r,"precision":p}[args.metric]
        if (best is None) or (score > best[0]):
            best = (score, t, p, r, f)

    if best is None:
        print("[WARN] No threshold met constraints; relaxing precision floor.")
        for t in ts:
            yhat_v = (pv >= t).astype(int)
            p,r,f = prf1(yv, yhat_v)
            score = {"f1":f,"recall":r,"precision":p}[args.metric]
            if (best is None) or (score > best[0]):
                best = (score, t, p, r, f)

    _, t_best, p_best, r_best, f1_best = best
    yhat_t = (pt >= t_best).astype(int)
    pT, rT, f1T = prf1(yt, yhat_t)

    print(f"[VAL] best threshold={t_best:.3f} | F1={f1_best:.3f} P={p_best:.3f} R={r_best:.3f}")
    print(f"[TEST] @t*          | F1={f1T:.3f} P={pT:.3f} R={rT:.3f}")

if __name__ == "__main__":
    main()
