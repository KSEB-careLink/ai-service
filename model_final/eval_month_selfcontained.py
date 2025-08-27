import os, json, argparse, numpy as np, pandas as pd, torch
import torch.nn as nn
from datetime import datetime

class LSTMRegMulti(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=1, bidir=False):
        super().__init__()
        self.bidir = bidir
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.head_month = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        last = hn[-1] if not self.bidir else torch.cat([hn[-2], hn[-1]], dim=1)
        return self.head_month(last).squeeze(1)

class LSTMRegSingle(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=1, bidir=False):
        super().__init__()
        self.bidir = bidir
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        last = hn[-1] if not self.bidir else torch.cat([hn[-2], hn[-1]], dim=1)
        return self.head(last).squeeze(1)

def interpolate_by_gap(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').astype(float).copy()
    isn = s.isna()
    if not isn.any(): return s
    na_idx = np.where(isn.values)[0]
    splits = np.where(np.diff(na_idx) != 1)[0] + 1
    runs = np.split(na_idx, splits)
    valid_idx = np.where(~isn.values)[0]
    for run in runs:
        if len(run) == 0: continue
        n = len(run); start_i = int(run[0]); end_i = int(run[-1])
        left_c = valid_idx[valid_idx < start_i]; left_take = left_c[-n:] if left_c.size > 0 else np.array([], dtype=int)
        right_c = valid_idx[valid_idx > end_i]; right_take = right_c[:n] if right_c.size > 0 else np.array([], dtype=int)
        nei = np.concatenate([left_take, right_take], 0)
        if nei.size > 0:
            fill_val = float(np.nanmean(s.values[nei].astype(float)))
            s.values[run] = fill_val
    if s.isna().any():
        med = float(np.nanmedian(s.values)); med = 0.0 if not np.isfinite(med) else med
        s = s.fillna(med)
    return s

def make_daily(df):
    g = df.groupby(df['created_at'].dt.date).agg(
        solved=('is_correct','count'),
        correct=('is_correct','sum'),
        avg_time=('response_time_sec','mean')
    ).reset_index().rename(columns={'created_at':'date'})
    g['date'] = pd.to_datetime(g['date'])
    g['daily_acc_rate'] = (g['correct']/g['solved']).astype(float).clip(0,1)
    g['daily_avg_time'] = pd.to_numeric(g['avg_time'], errors='coerce')
    return g[['date','daily_acc_rate','daily_avg_time']]

def build_feat(daily):
    if daily.empty: return None
    start, end = daily['date'].min(), daily['date'].max()
    cal = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})
    cal['weekday'] = cal['date'].dt.weekday
    feat = cal.merge(daily, how='left', on='date')
    rt = pd.to_numeric(feat['daily_avg_time'], errors='coerce')
    if rt.notna().any():
        q1, q3 = np.nanpercentile(rt, 25), np.nanpercentile(rt, 75)
        iqr = max(q3 - q1, 0.0)
        th = max(q3 + 1.5*iqr, 600.0)
        rt = rt.where(rt <= th, np.nan)
    feat['daily_avg_time'] = interpolate_by_gap(rt)
    feat['daily_acc_rate'] = interpolate_by_gap(feat['daily_acc_rate']).clip(0,1)
    rt_scaled = (np.log1p(feat['daily_avg_time'].values.astype(float)) / np.log1p(600.0))
    feat['rt_scaled'] = np.clip(rt_scaled, 0.0, 1.0).astype(np.float32)
    return feat

def predict_month_avg(model, window, feat, ym):
    M_start = datetime.strptime(ym + "-01", "%Y-%m-%d")
    days = pd.date_range(M_start, M_start + pd.offsets.MonthEnd(1), freq='D')
    acc = feat['daily_acc_rate'].values.astype(np.float32)
    rt  = feat['rt_scaled'].values.astype(np.float32)
    wds = feat['weekday'].values.astype(int)
    dates = feat['date'].values
    preds = []
    for d in days:
        idx = np.searchsorted(dates, np.datetime64(d))
        if idx < window: continue
        sl = slice(idx-window, idx)
        wd_oh = np.zeros((window,7), dtype=np.float32)
        for i, w in enumerate(wds[sl]):
            if 0 <= int(w) < 7: wd_oh[i,int(w)] = 1.0
        X = np.concatenate([acc[sl].reshape(window,1), rt[sl].reshape(window,1), wd_oh], axis=1)
        xb = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = float(model(xb).item())
        preds.append(np.clip(y, 0.0, 1.0))
    return (float(np.mean(preds)) if preds else None), len(preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--datadir", required=True, help="dir with model_daily_10_and_month.pth or model_daily_nextday.pth")
    ap.add_argument("--patient_id", required=True)
    ap.add_argument("--start_month"); ap.add_argument("--end_month")
    args = ap.parse_args()

    ckpt_new = os.path.join(args.datadir, "model_daily_10_and_month.pth")
    ckpt_old = os.path.join(args.datadir, "model_daily_nextday.pth")
    ckpt_path = ckpt_new if os.path.exists(ckpt_new) else ckpt_old
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("checkpoint not found")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    W = int(ckpt.get("window", 45)); F = int(ckpt.get("input_dim", 9))
    hidden = int(ckpt.get("hidden", 64)); layers = int(ckpt.get("layers", 1))
    bidir = bool(ckpt.get("bidir", False))

    try:
        model = LSTMRegMulti(F, hidden=hidden, layers=layers, bidir=bidir); model.load_state_dict(ckpt["state_dict"]); model.eval()
    except Exception:
        model = LSTMRegSingle(F, hidden=hidden, layers=layers, bidir=bidir); model.load_state_dict(ckpt["state_dict"]); model.eval()

    DF = pd.read_csv(args.csv)
    need = {"patient_id","is_correct","created_at","response_time_sec"}
    if not need.issubset(DF.columns):
        raise ValueError(f"csv must contain {need}")
    df = DF[DF["patient_id"]==args.patient_id].copy()
    if df.empty: raise RuntimeError("no rows for this patient_id")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors='coerce').clip(0,1)

    daily = make_daily(df)
    feat = build_feat(daily)
    if feat is None or len(feat) < W: raise RuntimeError("insufficient history")

    first_m = df["created_at"].dt.to_period("M").min().strftime("%Y-%m")
    last_m  = df["created_at"].dt.to_period("M").max().strftime("%Y-%m")
    start_m = args.start_month or first_m
    end_m   = args.end_month   or last_m

    daily['month'] = daily['date'].dt.to_period('M')
    gt_month = daily.groupby('month')['daily_acc_rate'].mean().astype(float)

    rows = []

    ys, ms = map(int, start_m.split('-')); ye, me = map(int, end_m.split('-'))
    y, m = ys, ms
    while (y < ye) or (y == ye and m <= me):
        ym = f"{y:04d}-{m:02d}"
        pred_m, used = predict_month_avg(model, W, feat, ym)
        gt_m = gt_month.get(pd.Period(ym), np.nan)
        actual_m = float(gt_m) if np.isfinite(gt_m) else None
        if pred_m is not None and actual_m is not None:
            rows.append({"month": ym, "pred": pred_m, "actual": actual_m,
                         "mae": abs(pred_m-actual_m), "rmse": (pred_m-actual_m)**2})
        m += 1
        if m == 13: m = 1; y += 1

    if not rows: raise RuntimeError("no overlapping months between prediction and actual")
    eval_df = pd.DataFrame(rows)
    acc = (1 - eval_df["mae"].mean()) * 100.0
    mae = float(eval_df["mae"].mean())
    rmse = float(np.sqrt(eval_df["rmse"].mean()))
    print("\n[월 단위 평가 결과]")
    print(eval_df.to_string(index=False))
    print("\n요약:")
    print(f"- Accuracy (월평균): {acc:.2f}%")
    print(f"- MAE: {mae:.4f}")
    print(f"- RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
