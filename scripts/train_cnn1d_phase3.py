#!/usr/bin/env python3
import argparse, os, json, numpy as np, torch, torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd

def safe_auc(y, p):
    try: return roc_auc_score(y, p)
    except: return float("nan")

class CNN1D(nn.Module):
    def __init__(self, in_feat, seq_len, channels=32, k=5, dropout=0.5, fc_hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_feat, channels, kernel_size=k, padding=k//2)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, padding=k//2)
        self.bn2   = nn.BatchNorm1d(channels)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.drop  = nn.Dropout(dropout)
        if fc_hidden>0:
            self.head = nn.Sequential(nn.Linear(channels, fc_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_hidden, 1))
        else:
            self.head = nn.Linear(channels, 1)

    def forward(self, x):           # x: (B, L, F)
        x = x.transpose(1,2)        # -> (B, F, L)
        x = self.drop(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.head(x).squeeze(-1)

def run_epoch(m, loader, device, crit, opt=None):
    train = opt is not None
    m.train() if train else m.eval()
    losses=[]; ys=[]; ps=[]
    for xb,yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = m(xb)
        loss   = crit(logits, yb)
        if train:
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        with torch.no_grad():
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            ps.append(prob); ys.append(yb.detach().cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    yhat = (p>=0.5).astype(int)
    acc, (prec,rec,f1,_) = accuracy_score(y,yhat), precision_recall_fscore_support(y,yhat,average="binary",zero_division=0)
    return {"loss":float(np.mean(losses)),"acc":float(acc),"precision":float(prec),"recall":float(rec),"f1":float(f1),"auc":float(safe_auc(y,p))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--out-dir")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=32); ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.5); ap.add_argument("--fc-hidden", type=int, default=64)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")
    ap.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs)") 
    args=ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join("runs", f"cnn1d_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.device=="auto":
        device=torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu"))
    else: device=torch.device(args.device)
    print("[INFO] device:", device)

    d=np.load(args.npz, allow_pickle=True)
    Xtr=d["X_train"].astype("float32"); ytr=d["y_train"].astype("float32")
    Xva=d["X_val"].astype("float32");   yva=d["y_val"].astype("float32")
    Xte=d["X_test"].astype("float32");  yte=d["y_test"].astype("float32")

    tr=TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va=TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    te=TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    L=len(Xtr[0]); F=Xtr.shape[2]

    m=CNN1D(in_feat=F, seq_len=L, channels=args.channels, k=args.kernel, dropout=args.dropout, fc_hidden=args.fc_hidden).to(device)

    # class weighting like LSTM
    pos = float((ytr==1).sum()); neg=float((ytr==0).sum()); w = 1.0 if pos==0 or neg==0 else np.clip(neg/pos, 0.5, 5.0)
    if args.pos_weight is not None: w=float(args.pos_weight)
    crit = nn.BCEWithLogitsLoss() if args.no_class_weight else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], dtype=torch.float32, device=device))
    print(f"[INFO] pos_weight={w:.3f}")

    opt=torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    tr_loader=DataLoader(tr, batch_size=args.batch, shuffle=True)
    va_loader=DataLoader(va, batch_size=args.batch)
    te_loader=DataLoader(te, batch_size=args.batch)

    best=-1; best_state=None; rows=[]
    for e in range(1, args.epochs+1):
        trm=run_epoch(m,tr_loader,device,crit,opt); vam=run_epoch(m,va_loader,device,crit,None)
        sch.step(vam["f1"]); rows.append({"epoch":e, **{f"train_{k}":v for k,v in trm.items()}, **{f"val_{k}":v for k,v in vam.items()}, "lr":opt.param_groups[0]["lr"]})
        print(f"[E{e:03d}] train f1={trm['f1']:.3f} auc={trm['auc']:.3f} | val f1={vam['f1']:.3f} auc={vam['auc']:.3f} | lr={opt.param_groups[0]['lr']:.2e}")
        if vam["f1"]>best+1e-6:
            best=vam["f1"]; best_state={"model_state_dict":m.state_dict(),"args":vars(args),"input_dim":F,"seq_len":L}
            torch.save(best_state, os.path.join(args.out_dir,"best_model.pt"))
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir,"metrics_epoch_log.csv"), index=False)

    # final summary
    if best_state: m.load_state_dict(best_state["model_state_dict"])
    train=run_epoch(m,tr_loader,device,crit,None); val=run_epoch(m,va_loader,device,crit,None); test=run_epoch(m,te_loader,device,crit,None)
    summ={"train":train,"val":val,"test":test}
    with open(os.path.join(args.out_dir,"final_summary.json"),"w") as f: json.dump({"final_metrics":summ,"args":vars(args)}, f, indent=2)
    print("\n[SUMMARY]\n", json.dumps(summ, indent=2))
    print("Saved to:", args.out_dir)

if __name__=="__main__": main()
