# routes/predict_accuracy.py
import os, json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pymysql
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

# -----------------------------
# 0) 환경 변수
# -----------------------------
EXPECTED_DAILY_SOLVES = int(os.getenv("EXPECTED_DAILY_SOLVES", "3"))
BASELINE_ACC = float(os.getenv("BASELINE_ACC", "0.60"))
COLD_DAYS_OK = int(os.getenv("COLD_DAYS_OK", "30"))
COLD_ATTEMPTS_OK = int(os.getenv("COLD_ATTEMPTS_OK", "120"))

# -----------------------------
# 1) 모델 로드
# -----------------------------
class LSTMReg(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.head(hn[-1])

MODEL_DIR = os.getenv("MODEL_DIR", "model_final/data_daily_w45")
CKPT_PATH = os.getenv("MODEL_CKPT", os.path.join(MODEL_DIR, "model.pth"))
if not os.path.exists(CKPT_PATH):
    raise RuntimeError(f"[predict-accuracy-live] model checkpoint not found: {CKPT_PATH}")

_ckpt = torch.load(CKPT_PATH, map_location="cpu")
W       = int(_ckpt.get("window", 45))
F       = int(_ckpt.get("input_dim", 9))
_hidden = int(_ckpt.get("hidden", 64))

_model = LSTMReg(input_dim=F, hidden=_hidden)
_model.load_state_dict(_ckpt["state_dict"])
_model.eval()

# -----------------------------
# 2) DB 설정 파일 로딩
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CFG_PATH = Path(os.getenv("DB_CONFIG_PATH", BASE_DIR / "db_config.json"))

if not CFG_PATH.exists():
    raise RuntimeError(f"[predict-accuracy-live] DB 설정 파일이 없습니다: {CFG_PATH}")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = json.load(f)

required_keys = ["host", "user", "password", "database", "port"]
missing = [k for k in required_keys if k not in CFG]
if missing:
    raise RuntimeError(f"[predict-accuracy-live] db_config.json에 누락된 키: {missing}")

DB_HOST = CFG["host"]
DB_USER = CFG["user"]
DB_PASS = CFG["password"]
DB_NAME = CFG["database"]
DB_PORT = int(CFG["port"])

# -----------------------------
# 3) DB → pandas 로드
# -----------------------------
def load_logs_from_db(patient_id: str) -> pd.DataFrame:
    conn = pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS,
        database=DB_NAME, port=DB_PORT, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        sql = """
        SELECT
          patient_id,
          quiz_id,             -- ✅ ERD 기준
          selected_index,      -- ✅ 있어도 무방 (안 쓰면 무시)
          is_correct,
          response_time_sec,
          created_at
        FROM quiz_logs
        WHERE patient_id = %s
        ORDER BY created_at
        """
        df = pd.read_sql(sql, conn, params=[patient_id])
    finally:
        conn.close()
    return df

# -----------------------------
# 4) 전처리 유틸
# -----------------------------
def _make_daily(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["date"] = tmp["created_at"].dt.floor("D")
    g = tmp.groupby("date").agg(
        solved=("is_correct", "count"),
        correct=("is_correct", "sum"),
        avg_time=("response_time_sec", "mean"),
    ).reset_index()
    g["daily_acc_rate"] = (g["correct"] / g["solved"]).clip(0, 1)

    if g["avg_time"].notna().any():
        rt_med = float(np.nanmedian(g["avg_time"].values.astype(float)))
        if not np.isfinite(rt_med):
            rt_med = 10.0
    else:
        rt_med = 10.0
    g["daily_avg_time"] = g["avg_time"].fillna(rt_med)

    return g[["date", "daily_acc_rate", "daily_avg_time"]]

def _build_features(daily: pd.DataFrame, wnd_start: datetime, wnd_end: datetime, window: int) -> np.ndarray:
    days = pd.date_range(wnd_start, wnd_end, freq="D")
    base = pd.DataFrame({"date": days})
    base["weekday"] = base["date"].dt.weekday

    feat = base.merge(daily, how="left", on="date")

    if feat["daily_avg_time"].notna().any():
        rt_med = float(np.nanmedian(feat["daily_avg_time"].values.astype(float)))
        if not np.isfinite(rt_med):
            rt_med = 10.0
    else:
        rt_med = 10.0
    feat["daily_acc_rate"] = feat["daily_acc_rate"].fillna(0.0).astype(float).clip(0, 1)
    feat["daily_avg_time"] = feat["daily_avg_time"].fillna(rt_med).astype(float).clip(0, 600)
    feat["daily_avg_time"] = np.log1p(feat["daily_avg_time"]) / np.log1p(600.0)

    wd = np.zeros((len(feat), 7), dtype=np.float32)
    for i, w in enumerate(feat["weekday"].astype(int).values):
        if 0 <= w < 7:
            wd[i, w] = 1.0

    X = np.concatenate([
        feat["daily_acc_rate"].values.reshape(-1, 1).astype(np.float32),
        feat["daily_avg_time"].values.reshape(-1, 1).astype(np.float32),
        wd
    ], axis=1)

    if X.shape[0] < window:
        pad = np.zeros((window - X.shape[0], X.shape[1]), dtype=np.float32)
        X = np.vstack([pad, X])
    elif X.shape[0] > window:
        X = X[-window:]
    return X.astype(np.float32)

# -----------------------------
# 5) 예측 엔드포인트
# -----------------------------
@router.get("/predict-accuracy-live")
def predict_accuracy_live(
    patient_id: str = Query(..., description="환자 ID"),
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$", description="예측 대상 월 (YYYY-MM)")
):
    df = load_logs_from_db(patient_id)

    try:
        M_start = datetime.strptime(month + "-01", "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="month must be YYYY-MM")
    wnd_end = M_start - timedelta(days=1)
    wnd_start = wnd_end - timedelta(days=W - 1)

    if df.empty:
        daily = pd.DataFrame(columns=["date", "daily_acc_rate", "daily_avg_time"])
        X = _build_features(daily, wnd_start, wnd_end, W)
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            raw_pred = float(_model(x).item())
        raw_pred = max(0.0, min(1.0, raw_pred))
        blended = BASELINE_ACC
        return {
            "patient_id": patient_id,
            "month": month,
            "predicted_final_accuracy": round(blended, 4),
            "cold_start": True,
            "ui": {"badges": ["초기 예측 — 더 많은 풀이가 쌓일수록 정확도가 올라가요."]}
        }

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    df["is_correct"] = df["is_correct"].astype(int).clip(0, 1)

    in_wnd = df[(df["created_at"] >= wnd_start) & (df["created_at"] <= wnd_end)]
    attempts = int(in_wnd.shape[0])

    daily = _make_daily(df[df["created_at"] <= wnd_end].copy())
    X = _build_features(daily, wnd_start, wnd_end, W)
    if X.shape[1] != F:
        raise HTTPException(status_code=500, detail=f"입력 차원 불일치: X({X.shape[1]}) vs 모델({F})")

    active_days = 0
    if not daily.empty:
        mask = (daily["date"] >= pd.Timestamp(wnd_start)) & (daily["date"] <= pd.Timestamp(wnd_end))
        active_days = int(daily.loc[mask].shape[0])

    day_coverage = active_days / float(W) if W > 0 else 0.0
    expected_attempts = max(1, W * EXPECTED_DAILY_SOLVES)
    attempt_coverage = min(1.0, attempts / float(expected_attempts))
    coverage = max(day_coverage, attempt_coverage)

    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        raw_pred = float(_model(x).item())
    raw_pred = max(0.0, min(1.0, raw_pred))

    blended = coverage * raw_pred + (1.0 - coverage) * BASELINE_ACC
    cold_start_flag = not (active_days >= COLD_DAYS_OK or attempts >= COLD_ATTEMPTS_OK)

    return {
        "patient_id": patient_id,
        "month": month,
        "predicted_final_accuracy": round(blended, 4),
        "cold_start": cold_start_flag,
        "ui": {
            "badges": ["초기 예측 — 더 많은 풀이가 쌓일수록 정확도가 올라가요."] if cold_start_flag else []
        }
    }
