import os, json, argparse, numpy as np, pandas as pd, torch
import torch.nn as nn
from datetime import datetime, timedelta

def make_daily(df):
    g = df.groupby(df['created_at'].dt.date).agg(
        solved=('is_correct','count'),
        correct=('is_correct','sum'),
        avg_time=('response_time_sec','mean')
    ).reset_index().rename(columns={'created_at':'date'})
    g['daily_acc_rate'] = (g['correct'] / g['solved']).clip(0,1)
    g['date'] = pd.to_datetime(g['date'])
  
    if g['avg_time'].notna().any():
        rt_med = float(np.nanmedian(g['avg_time'].values.astype(float)))
        if not np.isfinite(rt_med): rt_med = 10.0
    else:
        rt_med = 10.0
    g['daily_avg_time'] = g['avg_time'].fillna(rt_med)
    return g[['date','daily_acc_rate','daily_avg_time']]

def month_iter(start_ym, end_ym):
    ys, ms = map(int, start_ym.split('-')); ye, me = map(int, end_ym.split('-'))
    y, m = ys, ms
    while (y < ye) or (y == ye and m <= me):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13: m = 1; y += 1

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

def predict_month(model, W, daily, target_month, impute="mean3"):
    """
    반환: (preds_df, gt_df, note)
      - preds_df: 해당 월 날짜별 예측 DataFrame(date, pred)
      - gt_df:    해당 월 관측값 DataFrame(date, daily_acc_rate)
      - note:     문자열(보간 방식/스킵 사유 등)
    """
    if daily.shape[0] < W:
        return None, None, "insufficient total history"

    M_start = datetime.strptime(target_month+"-01", "%Y-%m-%d")
    days_in_month = pd.date_range(M_start, M_start + pd.offsets.MonthEnd(1), freq='D')

    wnd_end = M_start - timedelta(days=1)
    cal = pd.DataFrame({'date': pd.date_range(M_start - timedelta(days=W), wnd_end, freq='D')})
    feat0 = cal.merge(daily, how='left', on='date')

    s = feat0['daily_acc_rate'].astype(float).copy()
    if s.isna().any():
        if not impute:
            return None, None, "gap in history window"
        if impute == "ffill":
            s = s.ffill().bfill()
        else: 
            s2 = s.rolling(3, min_periods=1, center=True).mean()
            s = s.fillna(s2).ffill().bfill()
        feat0['daily_acc_rate'] = s.clip(0,1)

    if daily['daily_avg_time'].notna().any():
        rt_med = float(np.nanmedian(daily['daily_avg_time'].values.astype(float)))
        if not np.isfinite(rt_med): rt_med = 10.0
    else:
        rt_med = 10.0
    def norm_rt(x):
        return (np.log1p(np.asarray(x, dtype=np.float32)) / np.log1p(600.0)).clip(0,1)

    pred_map = {}
    preds = []
    for d in days_in_month:
        wdates = pd.date_range(d - timedelta(days=W), d - timedelta(days=1), freq='D')
        acc_win, rt_win = [], []
        wd_oh = np.zeros((W,7), dtype=np.float32)

        for i, wd in enumerate(wdates):
            row = daily[daily['date']==wd]
            if not row.empty:
                a = float(row['daily_acc_rate'].values[0])
                r = float(row['daily_avg_time'].values[0]) if not np.isnan(row['daily_avg_time'].values[0]) else rt_med
            else:
               
                a = float(pred_map.get(wd.strftime("%Y-%m-%d"), acc_win[-1] if acc_win else float(feat0['daily_acc_rate'].iloc[-1])))
                r = rt_med
            acc_win.append(a); rt_win.append(r)
            wd_oh[i, wd.weekday()] = 1.0

        X = np.concatenate([
            np.array(acc_win, dtype=np.float32).reshape(W,1),
            norm_rt(rt_win).reshape(W,1),
            wd_oh
        ], axis=1)  

        x = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            yhat = float(model(x).item())
        yhat = max(0.0, min(1.0, yhat))
        ds = d.strftime("%Y-%m-%d")
        preds.append((ds, yhat))
        pred_map[ds] = yhat  

    preds_df = pd.DataFrame(preds, columns=['date','pred'])
    gt_df = daily[(daily['date']>=days_in_month.min()) & (daily['date']<=days_in_month.max())][['date','daily_acc_rate']].copy()
    gt_df['date'] = gt_df['date'].dt.strftime("%Y-%m-%d")
    return preds_df, gt_df, f"imputed:{impute or 'none'}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="quiz_logs.csv")
    ap.add_argument("--datadir", required=True, help="dir containing model_daily_nextday.pth")
    ap.add_argument("--patient_id", required=True)
    ap.add_argument("--outdir", default="./out_user_eval", help="결과 저장 폴더")
    ap.add_argument("--start_month", required=False)  # YYYY-MM
    ap.add_argument("--end_month", required=False)    # YYYY-MM
    ap.add_argument("--impute", choices=["ffill","mean3","none"], default="mean3")
    ap.add_argument("--save_json", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    per_user_dir = os.path.join(args.outdir, "patient_summaries")
    months_json = os.paths.join(args.outdir, "months_json")
    os.makedirs(per_user_dir, exist_ok=True)
    if args.save_json:
        os.makedirs(months_json, exist_ok=True)

    ckpt = torch.load(os.path.join(args.datadir, "model_daily_nextday.pth"), map_location="cpu")
    W = int(ckpt.get("window", 45))
    F = int(ckpt.get("input_dim", 9))
    hidden = int(ckpt.get("hidden", 64))
    layers = int(ckpt.get("layers", 1))
    bidir  = bool(ckpt.get("bidir", False))
    if F != 9:
        raise RuntimeError(f"input_dim mismatch: expected 9, got {F}")
    model = LSTMReg(F, hidden=hidden, layers=layers, bidir=bidir)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    DF = pd.read_csv(args.csv)
    need = {"patient_id","is_correct","created_at","response_time_sec"}
    if not need.issubset(DF.columns):
        raise ValueError(f"csv must contain {need}")
    df = DF[DF["patient_id"]==args.patient_id].copy()
    if df.empty:
        raise RuntimeError("no rows for this patient_id")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    df["is_correct"] = df["is_correct"].astype(int).clip(0,1)

    daily = make_daily(df)

    first_month = df["created_at"].dt.to_period("M").min().strftime("%Y-%m")
    last_month  = df["created_at"].dt.to_period("M").max().strftime("%Y-%m")
    start_m = args.start_month or first_month
    end_m   = args.end_month   or last_month

    rows = []
    for ym in month_iter(start_m, end_m):
        preds_df, gt_df, note = predict_month(model, W, daily, ym, None if args.impute=="none" else args.impute)
        if preds_df is None:
            rows.append({
                "patient_id": args.patient_id, "month": ym,
                "pred_month_avg": None, "daily_mae": None,
                "monthly_mae": None, "coverage": 0.0, "n_days": 0,
                "note": f"skip: {note}"
            })
            continue

        pred_month_avg = float(preds_df["pred"].mean())

        m = preds_df.merge(gt_df, on="date", how="inner")
        if not m.empty:
            daily_mae = float((m["pred"] - m["daily_acc_rate"]).abs().mean())
            monthly_mae = float(abs(m["pred"].mean() - m["daily_acc_rate"].mean()))
            
            M_start = datetime.strptime(ym + "-01", "%Y-%m-%d")
            n_days = len(pd.date_range(M_start, M_start + pd.offsets.MonthEnd(1), freq='D'))
            coverage = len(m) / n_days
        else:
            daily_mae = None; monthly_mae = None; coverage = 0.0
            M_start = datetime.strptime(ym + "-01", "%Y-%m-%d")
            n_days = len(pd.date_range(M_start, M_start + pd.offsets.MonthEnd(1), freq='D'))

        if args.save_json:
            out = {
                "patient_id": args.patient_id, "month": ym,
                "pred_daily": [{"date": d, "acc": round(a,4)} for d,a in preds_df.values.tolist()],
                "pred_month_avg": round(pred_month_avg, 4),
                "eval": None if m.empty else {
                    "daily_mae": None if daily_mae is None else round(daily_mae,4),
                    "monthly_mae": None if monthly_mae is None else round(monthly_mae,4),
                    "coverage": round(coverage,3)
                },
                "note": note
            }
            with open(os.path.join(months_json, f"{args.patient_id}_{ym}.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        rows.append({
            "patient_id": args.patient_id, "month": ym,
            "pred_month_avg": round(pred_month_avg,4),
            "daily_mae": None if daily_mae is None else round(daily_mae,4),
            "monthly_mae": None if monthly_mae is None else round(monthly_mae,4),
            "coverage": round(coverage,3),
            "n_days": int(n_days),
            "note": note
        })

    out_csv = os.path.join(per_user_dir, f"{args.patient_id}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(json.dumps({
        "patient_id": args.patient_id,
        "start_month": start_m,
        "end_month": end_m,
        "saved_csv": out_csv,
        "json_dir": months_json if args.save_json else None
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
