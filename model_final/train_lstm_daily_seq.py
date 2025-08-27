import argparse, os, json, numpy as np, pandas as pd, torch, copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

def mask_outlier_rows(df, k=1.5, max_time_sec=600):
    """
    환자별 response_time_sec에 대해 이상치 행 마스크(True=이상치)를 반환.
    기준(OR):
      - response_time_sec > Q3 + k*IQR
      - response_time_sec > max_time_sec
    """
    out_mask = pd.Series(False, index=df.index)
    for pid, g in df.groupby('patient_id', sort=False):
        rt = pd.to_numeric(g['response_time_sec'], errors='coerce')
        if rt.notna().any():
            q1 = np.nanpercentile(rt, 25)
            q3 = np.nanpercentile(rt, 75)
            iqr = max(q3 - q1, 0.0)
            th_iqr = q3 + k * iqr
        else:
            th_iqr = np.inf
        mask_pid = (rt > th_iqr) | (rt > max_time_sec)
        out_mask.loc[g.index] = mask_pid.fillna(False)
    return out_mask

def make_daily(df):
    g = df.groupby(df['created_at'].dt.date).agg(
        solved=('is_correct', 'count'),
        correct=('is_correct', 'sum'),
        avg_time=('response_time_sec', 'mean')
    ).reset_index().rename(columns={'created_at':'date'})
    g['date'] = pd.to_datetime(g['date'])
    g['daily_acc_rate'] = (g['correct'] / g['solved']).astype(float).clip(0, 1)
    g['daily_avg_time'] = pd.to_numeric(g['avg_time'], errors='coerce')
    return g[['date','daily_acc_rate','daily_avg_time']]

def interpolate_by_gap(series: pd.Series) -> pd.Series:
    """
    규칙: 연속 결측 길이가 n이면 그 블록 전체를
         '앞쪽 n일 + 뒤쪽 n일'의 평균으로 채움.
    - 경계에서 n일이 부족하면 가능한 범위만 사용
    - 앞/뒤 모두 없으면 남겨두고, 마지막에 전체 중앙값으로 채움
    """
    s = pd.to_numeric(series, errors='coerce').astype(float).copy()
    isn = s.isna()
    if not isn.any():
        return s

    na_idx = np.where(isn.values)[0]
    if na_idx.size == 0:
        return s

    splits = np.where(np.diff(na_idx) != 1)[0] + 1
    runs = np.split(na_idx, splits)
    valid_idx = np.where(~isn.values)[0]

    for run in runs:
        if len(run) == 0:
            continue
        n = len(run)
        start_i = int(run[0])
        end_i   = int(run[-1])

        left_candidates = valid_idx[valid_idx < start_i]
        left_take = left_candidates[-n:] if left_candidates.size > 0 else np.array([], dtype=int)

        right_candidates = valid_idx[valid_idx > end_i]
        right_take = right_candidates[:n] if right_candidates.size > 0 else np.array([], dtype=int)

        neighbors = np.concatenate([left_take, right_take], axis=0)
        if neighbors.size > 0:
            fill_val = float(np.nanmean(s.values[neighbors].astype(float)))
            s.values[run] = fill_val

    if s.isna().any():
        med = float(np.nanmedian(s.values))
        if not np.isfinite(med):
            med = 0.0
        s = s.fillna(med)
    return s

def build_sequences(daily, window, max_time_sec, iqr_k):
    """
    - 캘린더 결합으로 누락일 포함
    - 일평균시간에 대해 2차 안전 클리핑(IQR/상한 → NaN) 후 갭 기반 보간
    - 정답률도 갭 기반 보간
    - X: [acc, rt_scaled, weekday_onehot(7)] -> input_dim=9
    - y_next10: t 이후 10일의 일별 정답률
    - y_month:  t 날짜가 속한 달의 평균 정답률
    """
    if daily.empty:
        return [], [], []

    start = daily['date'].min()
    end   = daily['date'].max()
    cal = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})
    cal['weekday'] = cal['date'].dt.weekday
    feat = cal.merge(daily, how='left', on='date')

    rt = pd.to_numeric(feat['daily_avg_time'], errors='coerce')
    if rt.notna().any():
        q1 = np.nanpercentile(rt, 25)
        q3 = np.nanpercentile(rt, 75)
        iqr = max(q3 - q1, 0.0)
        th = max(q3 + iqr_k * iqr, max_time_sec)
        rt = rt.where(rt <= th, np.nan)
    feat['daily_avg_time'] = rt

    feat['daily_acc_rate'] = interpolate_by_gap(feat['daily_acc_rate']).clip(0, 1)
    feat['daily_avg_time'] = interpolate_by_gap(feat['daily_avg_time'])

    rt_scaled = (np.log1p(feat['daily_avg_time'].values.astype(float)) / np.log1p(600.0))
    rt_scaled = np.clip(rt_scaled, 0.0, 1.0).astype(np.float32)

    acc = feat['daily_acc_rate'].values.astype(np.float32)
    wds = feat['weekday'].values.astype(int)

    feat['month'] = feat['date'].dt.to_period('M')
    month_mean = feat.groupby('month')['daily_acc_rate'].mean().astype(float)

    X_list, y10_list, ym_list = [], [], []
    T = len(feat)

    for t in range(window, T - 10):
        win = slice(t - window, t)

        wd_oh = np.zeros((window, 7), dtype=np.float32)
        for i, w in enumerate(wds[win]):
            if 0 <= int(w) < 7:
                wd_oh[i, int(w)] = 1.0

        X = np.concatenate([
            acc[win].reshape(window, 1),
            rt_scaled[win].reshape(window, 1),
            wd_oh
        ], axis=1)  # (window, 9)

        y_next10 = acc[t:t+10]
        if len(y_next10) < 10:
            continue

        cur_month = pd.Period(feat['date'].iloc[t], freq='M')
        y_month = float(month_mean.get(cur_month, np.nan))
        if not np.isfinite(y_month):
            local = acc[max(0, t-15):t+15]
            y_month = float(np.nanmean(local)) if np.isfinite(np.nanmean(local)) else float(np.nanmean(acc))

        X_list.append(X)
        y10_list.append(y_next10.astype(np.float32))
        ym_list.append(np.float32(y_month))

    return X_list, y10_list, ym_list

class LSTMRegMulti(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=1, bidir=False):
        super().__init__()
        self.bidir = bidir
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.head_daily = nn.Sequential(
            nn.Linear(out_dim, 128), nn.ReLU(),
            nn.Linear(128, 10), nn.Sigmoid()
        )
        self.head_month = nn.Sequential(
            nn.Linear(out_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last = hn[-1] if not self.bidir else torch.cat([hn[-2], hn[-1]], dim=1)
        y10 = self.head_daily(last)       
        ym  = self.head_month(last).squeeze(1)  
        return y10, ym

class SeqDS(Dataset):
    def __init__(self, X, y10, ym):
        self.X   = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)        
        self.y10 = torch.tensor(np.stack(y10, axis=0), dtype=torch.float32)      
        self.ym  = torch.tensor(np.array(ym).reshape(-1), dtype=torch.float32)   
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y10[i], self.ym[i]

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
    ap.add_argument("--lr", type=float, default=1e-3)          
    ap.add_argument("--max_time_sec", type=float, default=600) 
    ap.add_argument("--iqr_k", type=float, default=1.5)         
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    need = {"patient_id","is_correct","created_at","response_time_sec"}
    if not need.issubset(df.columns):
        raise ValueError(f"csv must contain {need}")
    df['created_at'] = pd.to_datetime(df['created_at'], errors="coerce")
    df = df.dropna(subset=['created_at']).sort_values(['patient_id','created_at']).reset_index(drop=True)
    df['is_correct'] = pd.to_numeric(df['is_correct'], errors='coerce').clip(0,1)

    mask = mask_outlier_rows(df, k=args.iqr_k, max_time_sec=args.max_time_sec)
    df = df.loc[~mask].copy()

    X_all, y10_all, ym_all = [], [], []
    for pid, g in df.groupby('patient_id', sort=False):
        daily = make_daily(g)
        Xs, y10s, yms = build_sequences(
            daily,
            window=args.window,
            max_time_sec=args.max_time_sec,
            iqr_k=args.iqr_k
        )
        if Xs:
            X_all.extend(Xs); y10_all.extend(y10s); ym_all.extend(yms)

    if not X_all:
        raise RuntimeError("No training samples; check data density after outlier removal/interpolation.")

    ds = SeqDS(X_all, y10_all, ym_all)
    N = len(ds)
    n_val = max(1, int(N * args.val_split))
    n_tr  = N - n_val
    tr_set, va_set = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed))
    tr_loader = DataLoader(tr_set, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegMulti(input_dim=9, hidden=args.hidden, layers=args.layers, bidir=args.bidir).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)   # --lr 적용
    mse = nn.MSELoss()

    def run(loader, train=False):
        model.train() if train else model.eval()
        sum_mse_total, sum_mae10, sum_mae_month, n = 0.0, 0.0, 0.0, 0
        with torch.set_grad_enabled(train):
            for Xb, y10b, ymb in loader:
                Xb, y10b, ymb = Xb.to(device), y10b.to(device), ymb.to(device)
                p10, pm = model(Xb)
                loss_daily = mse(p10, y10b)
                loss_month = mse(pm,  ymb)
                loss = 0.7 * loss_daily + 0.3 * loss_month
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()

                bsz = Xb.size(0)
                sum_mse_total += loss.item() * bsz
                sum_mae10     += torch.mean(torch.abs(p10 - y10b), dim=1).sum().item()
                sum_mae_month += torch.abs(pm - ymb).sum().item()
                n += bsz
        return sum_mse_total / n, sum_mae10 / n, sum_mae_month / n

    hist = {"train_mse_total": [], "val_mse_total": [], "val_mae10": [], "val_mae_month": []}
    best, best_state = float("inf"), None

    for ep in range(1, args.epochs+1):
        tr_mse_total, _, _ = run(tr_loader, train=True)
        va_mse_total, va_mae10, va_mae_month = run(va_loader, train=False)
        hist["train_mse_total"].append(tr_mse_total)
        hist["val_mse_total"].append(va_mse_total)
        hist["val_mae10"].append(va_mae10)
        hist["val_mae_month"].append(va_mae_month)
        print(f"[{ep:02d}] train_mse_total={tr_mse_total:.4f} "
              f"val_mse_total={va_mse_total:.4f} val_mae10={va_mae10:.4f} val_mae_month={va_mae_month:.4f}")
        if va_mse_total < best:
            best, best_state = va_mse_total, copy.deepcopy(model.state_dict())

    ckpt = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "window": args.window,
        "input_dim": 9,
        "hidden": args.hidden,
        "layers": args.layers,
        "bidir": args.bidir,
        "hist": hist,
        "heads": {"daily_10": True, "month_avg": True},
        "preproc": {
            "iqr_k": args.iqr_k,
            "max_time_sec": args.max_time_sec,
            "gap_rule": "len(run)=n -> mean(last n valid before + first n valid after)",
            "rt_log_scale_base": 600.0
        }
    }
    os.path.isdir(args.outdir) or os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "model_daily_10_and_month.pth")
    torch.save(ckpt, ckpt_path)
    with open(os.path.join(args.outdir, "metrics_daily.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Saved model to {ckpt_path}")

if __name__ == "__main__":
    main()

# import argparse, os, json, numpy as np, pandas as pd, torch, copy
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# from datetime import datetime, timedelta

# def make_daily(df):
#     g = df.groupby(df['created_at'].dt.date).agg(
#         solved=('is_correct', 'count'),
#         correct=('is_correct', 'sum'),
#         avg_time=('response_time_sec', 'mean')
#     ).reset_index().rename(columns={'created_at':'date'})
#     g['daily_acc_rate'] = (g['correct'] / g['solved']).clip(0,1)
#     g['date'] = pd.to_datetime(g['date'])

#     if g['avg_time'].notna().any():
#         rt_med = float(np.nanmedian(g['avg_time'].values.astype(float)))
#         if not np.isfinite(rt_med):
#             rt_med = 10.0
#     else:
#         rt_med = 10.0
#     g['daily_avg_time'] = g['avg_time'].fillna(rt_med)

#     return g[['date','daily_acc_rate','daily_avg_time']]

# def build_sequences(daily, window):
#     if daily.empty:
#         return [], []
#     start = daily['date'].min()
#     end   = daily['date'].max()
#     cal = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})
#     cal['weekday'] = cal['date'].dt.weekday
#     feat = cal.merge(daily, how='left', on='date')

#     acc = feat['daily_acc_rate'].fillna(0.0).astype(np.float32).clip(0,1).values
   
#     if feat['daily_avg_time'].notna().any():
#         rt_med = float(np.nanmedian(feat['daily_avg_time'].values.astype(float)))
#         if not np.isfinite(rt_med):
#             rt_med = 10.0
#     else:
#         rt_med = 10.0
#     rt = feat['daily_avg_time'].fillna(rt_med).astype(np.float32).values
#     rt = (np.log1p(rt) / np.log1p(600.0)).clip(0,1).astype(np.float32)

#     wds = feat['weekday'].values.astype(int)

#     X_list, y_list = [], []
 
#     for t in range(window, len(feat)):
#         lab_idx = t
#         if np.isnan(feat['daily_acc_rate'].iloc[lab_idx]):
#             continue

#         win = slice(t-window, t)

#         wd_oh = np.zeros((window, 7), dtype=np.float32)
#         for i, w in enumerate(wds[win]):
#             if 0 <= int(w) < 7:
#                 wd_oh[i, int(w)] = 1.0

#         X = np.concatenate([
#             acc[win].reshape(window,1),      
#             rt[win].reshape(window,1),       
#             wd_oh                             
#         ], axis=1)                            
#         y = np.float32(feat['daily_acc_rate'].iloc[lab_idx])
#         X_list.append(X); y_list.append(y)

#     return X_list, y_list

# class LSTMReg(nn.Module):
#     def __init__(self, input_dim, hidden=64, layers=1, bidir=False):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
#         out_dim = hidden * (2 if bidir else 1)
#         self.head = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
#         self.bidir = bidir
#     def forward(self, x):
#         out, (hn, cn) = self.lstm(x)
#         last = hn[-1] if not self.bidir else torch.cat([hn[-2], hn[-1]], dim=1)
#         return self.head(last).squeeze(1)

# class SeqDS(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)   # (N, W, F)
#         self.y = torch.tensor(np.array(y).reshape(-1), dtype=torch.float32)  # (N,)
#     def __len__(self): return self.X.shape[0]
#     def __getitem__(self, i): return self.X[i], self.y[i]

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True)
#     ap.add_argument("--outdir", required=True)
#     ap.add_argument("--window", type=int, default=45)
#     ap.add_argument("--epochs", type=int, default=15)
#     ap.add_argument("--batch", type=int, default=128)
#     ap.add_argument("--hidden", type=int, default=64)
#     ap.add_argument("--layers", type=int, default=1)
#     ap.add_argument("--bidir", action="store_true")
#     ap.add_argument("--val_split", type=float, default=0.2)
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)
#     torch.manual_seed(args.seed); np.random.seed(args.seed)

#     df = pd.read_csv(args.csv)
#     need = {"patient_id","is_correct","created_at","response_time_sec"}
#     if not need.issubset(df.columns):
#         raise ValueError(f"csv must contain {need}")
#     df['created_at'] = pd.to_datetime(df['created_at'], errors="coerce")
#     df = df.dropna(subset=['created_at']).sort_values(['patient_id','created_at']).reset_index(drop=True)
#     df['is_correct'] = df['is_correct'].astype(int).clip(0,1)

#     X_all, y_all = [], []
#     for pid, g in df.groupby('patient_id', sort=False):
#         daily = make_daily(g)
#         Xs, ys = build_sequences(daily, args.window)
#         if Xs:
#             X_all.extend(Xs); y_all.extend(ys)

#     if not X_all:
#         raise RuntimeError("No training samples; check data density.")

#     ds = SeqDS(X_all, y_all)
#     N = len(ds)
#     n_val = max(1, int(N * args.val_split))
#     n_tr  = N - n_val
#     tr_set, va_set = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed))
#     tr_loader = DataLoader(tr_set, batch_size=args.batch, shuffle=True)
#     va_loader = DataLoader(va_set, batch_size=args.batch, shuffle=False)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LSTMReg(input_dim=9, hidden=args.hidden, layers=args.layers, bidir=args.bidir).to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#     crit = nn.MSELoss()

#     def run(loader, train=False):
#         tot, mae = 0.0, 0.0
#         model.train() if train else model.eval()
#         with torch.set_grad_enabled(train):
#             for Xb, yb in loader:
#                 Xb, yb = Xb.to(device), yb.to(device)
#                 pred = model(Xb)
#                 loss = crit(pred, yb)
#                 if train:
#                     opt.zero_grad(); loss.backward(); opt.step()
#                 tot += loss.item() * Xb.size(0)
#                 mae += torch.abs(pred - yb).sum().item()
#         return tot/len(loader.dataset), mae/len(loader.dataset)

#     hist = {"train_mse": [], "val_mse": [], "val_mae": []}
#     best, best_state = float("inf"), None
#     for ep in range(1, args.epochs+1):
#         tr_mse, _ = run(tr_loader, train=True)
#         va_mse, va_mae = run(va_loader, train=False)
#         hist["train_mse"].append(tr_mse); hist["val_mse"].append(va_mse); hist["val_mae"].append(va_mae)
#         print(f"[{ep:02d}] train_mse={tr_mse:.4f} val_mse={va_mse:.4f} val_mae={va_mae:.4f}")
#         if va_mse < best:
#             best, best_state = va_mse, copy.deepcopy(model.state_dict())

#     ckpt = {
#         "state_dict": best_state if best_state is not None else model.state_dict(),
#         "window": args.window,
#         "input_dim": 9,        
#         "hidden": args.hidden,
#         "layers": args.layers,
#         "bidir": args.bidir,
#         "hist": hist
#     }
#     torch.save(ckpt, os.path.join(args.outdir, "model_daily_nextday.pth"))
#     with open(os.path.join(args.outdir, "metrics_daily.json"), "w") as f:
#         json.dump(hist, f, indent=2)
#     print(f"Saved model to {os.path.join(args.outdir, 'model_daily_nextday.pth')}")

# if __name__ == "__main__":
#     main()
