import argparse, os, json, numpy as np, pandas as pd, torch, copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime, timedelta

def make_daily(df):
    g = df.groupby(df['created_at'].dt.date).agg(
        solved=('is_correct', 'count'),
        correct=('is_correct', 'sum'),
        avg_time=('response_time_sec', 'mean')
    ).reset_index().rename(columns={'created_at':'date'})
    g['daily_acc_rate'] = (g['correct'] / g['solved']).clip(0,1)
    g['date'] = pd.to_datetime(g['date'])

    if g['avg_time'].notna().any():
        rt_med = float(np.nanmedian(g['avg_time'].values.astype(float)))
        if not np.isfinite(rt_med):
            rt_med = 10.0
    else:
        rt_med = 10.0
    g['daily_avg_time'] = g['avg_time'].fillna(rt_med)

    return g[['date','daily_acc_rate','daily_avg_time']]

def build_sequences(daily, window):
    if daily.empty:
        return [], []
    start = daily['date'].min()
    end   = daily['date'].max()
    cal = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})
    cal['weekday'] = cal['date'].dt.weekday
    feat = cal.merge(daily, how='left', on='date')

    acc = feat['daily_acc_rate'].fillna(0.0).astype(np.float32).clip(0,1).values
   
    if feat['daily_avg_time'].notna().any():
        rt_med = float(np.nanmedian(feat['daily_avg_time'].values.astype(float)))
        if not np.isfinite(rt_med):
            rt_med = 10.0
    else:
        rt_med = 10.0
    rt = feat['daily_avg_time'].fillna(rt_med).astype(np.float32).values
    rt = (np.log1p(rt) / np.log1p(600.0)).clip(0,1).astype(np.float32)

    wds = feat['weekday'].values.astype(int)

    X_list, y_list = [], []
 
    for t in range(window, len(feat)):
        lab_idx = t
        if np.isnan(feat['daily_acc_rate'].iloc[lab_idx]):
            continue

        win = slice(t-window, t)

        wd_oh = np.zeros((window, 7), dtype=np.float32)
        for i, w in enumerate(wds[win]):
            if 0 <= int(w) < 7:
                wd_oh[i, int(w)] = 1.0

        X = np.concatenate([
            acc[win].reshape(window,1),      
            rt[win].reshape(window,1),       
            wd_oh                             
        ], axis=1)                            
        y = np.float32(feat['daily_acc_rate'].iloc[lab_idx])
        X_list.append(X); y_list.append(y)

    return X_list, y_list

class LSTMReg(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=1, bidir=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.bidir = bidir
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last = hn[-1] if not self.bidir else torch.cat([hn[-2], hn[-1]], dim=1)
        return self.head(last).squeeze(1)

class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)   # (N, W, F)
        self.y = torch.tensor(np.array(y).reshape(-1), dtype=torch.float32)  # (N,)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--window", type=int, default=45)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--bidir", action="store_true")
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    need = {"patient_id","is_correct","created_at","response_time_sec"}
    if not need.issubset(df.columns):
        raise ValueError(f"csv must contain {need}")
    df['created_at'] = pd.to_datetime(df['created_at'], errors="coerce")
    df = df.dropna(subset=['created_at']).sort_values(['patient_id','created_at']).reset_index(drop=True)
    df['is_correct'] = df['is_correct'].astype(int).clip(0,1)

    X_all, y_all = [], []
    for pid, g in df.groupby('patient_id', sort=False):
        daily = make_daily(g)
        Xs, ys = build_sequences(daily, args.window)
        if Xs:
            X_all.extend(Xs); y_all.extend(ys)

    if not X_all:
        raise RuntimeError("No training samples; check data density.")

    ds = SeqDS(X_all, y_all)
    N = len(ds)
    n_val = max(1, int(N * args.val_split))
    n_tr  = N - n_val
    tr_set, va_set = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed))
    tr_loader = DataLoader(tr_set, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMReg(input_dim=9, hidden=args.hidden, layers=args.layers, bidir=args.bidir).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    def run(loader, train=False):
        tot, mae = 0.0, 0.0
        model.train() if train else model.eval()
        with torch.set_grad_enabled(train):
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = crit(pred, yb)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * Xb.size(0)
                mae += torch.abs(pred - yb).sum().item()
        return tot/len(loader.dataset), mae/len(loader.dataset)

    hist = {"train_mse": [], "val_mse": [], "val_mae": []}
    best, best_state = float("inf"), None
    for ep in range(1, args.epochs+1):
        tr_mse, _ = run(tr_loader, train=True)
        va_mse, va_mae = run(va_loader, train=False)
        hist["train_mse"].append(tr_mse); hist["val_mse"].append(va_mse); hist["val_mae"].append(va_mae)
        print(f"[{ep:02d}] train_mse={tr_mse:.4f} val_mse={va_mse:.4f} val_mae={va_mae:.4f}")
        if va_mse < best:
            best, best_state = va_mse, copy.deepcopy(model.state_dict())

    ckpt = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "window": args.window,
        "input_dim": 9,        
        "hidden": args.hidden,
        "layers": args.layers,
        "bidir": args.bidir,
        "hist": hist
    }
    torch.save(ckpt, os.path.join(args.outdir, "model_daily_nextday.pth"))
    with open(os.path.join(args.outdir, "metrics_daily.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Saved model to {os.path.join(args.outdir, 'model_daily_nextday.pth')}")

if __name__ == "__main__":
    main()
