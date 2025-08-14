# routes/predict_accuracy.py
import os, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException, Query

# ✅ 타임존 유틸 (아래 utils_time.py 함께 사용)
from .utils_time import ensure_utc_series, kst_month_window_utc

router = APIRouter()

# -----------------------------
# 0) 환경 변수
# -----------------------------
EXPECTED_DAILY_SOLVES = int(os.getenv("EXPECTED_DAILY_SOLVES", "3"))
BASELINE_ACC = float(os.getenv("BASELINE_ACC", "0.60"))
COLD_DAYS_OK = int(os.getenv("COLD_DAYS_OK", "30"))
COLD_ATTEMPTS_OK = int(os.getenv("COLD_ATTEMPTS_OK", "120"))

# Node.js API 설정
NODE_API_BASE_URL = os.getenv("NODE_API_BASE_URL", "http://localhost:3000/api")
NODE_API_TIMEOUT = int(os.getenv("NODE_API_TIMEOUT", "30"))

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
# 2) Node.js API 호출
# -----------------------------
def load_logs_from_api(patient_id: str) -> pd.DataFrame:
    """Node.js API에서 quiz_logs 데이터를 가져옴"""
    try:
        url = f"{NODE_API_BASE_URL}/quiz-logs"
        response = requests.get(
            url,
            params={"patient_id": patient_id},
            timeout=NODE_API_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 404:
            return pd.DataFrame(columns=[
                "patient_id", "quiz_id", "selected_index",
                "is_correct", "response_time_sec", "created_at"
            ])
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Node.js API 오류: {response.text}"
            )

        data = response.json()
        if isinstance(data, dict) and "data" in data:
            quiz_logs = data["data"]
        elif isinstance(data, list):
            quiz_logs = data
        else:
            raise HTTPException(
                status_code=500,
                detail="예상하지 못한 API 응답 형식"
            )

        if not quiz_logs:
            return pd.DataFrame(columns=[
                "patient_id", "quiz_id", "selected_index",
                "is_correct", "response_time_sec", "created_at"
            ])

        df = pd.DataFrame(quiz_logs)

        required_columns = ["patient_id", "quiz_id", "selected_index",
                            "is_correct", "response_time_sec", "created_at"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=500,
                detail=f"API 응답에서 누락된 컬럼: {missing_columns}"
            )

        # 날짜 파싱은 utils에서 tz-safe하게 표준화하므로 여기선 최소 처리
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
            df = df.dropna(subset=["created_at"])

        return df

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Node.js API 응답 시간 초과")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Node.js API 연결 실패")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API 호출 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 처리 오류: {type(e).__name__}: {e}")

# -----------------------------
# 3) 전처리 유틸
# -----------------------------
def _make_daily(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    # ✅ UTC 기준의 '일' 그리드로 집계 (tz-aware 상태 유지)
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
    # ✅ wnd_start/wnd_end는 tz-aware(UTC). 이미 tz-aware이므로 tz= 파라미터를 주면 안 됨!
    # pd.date_range는 start/end가 tz-aware면 자동으로 해당 타임존을 사용
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
    feat["daily_acc_rate"] = (
        feat["daily_acc_rate"]
        .fillna(0.0)
        .infer_objects(copy=False)
        .astype(float)
        .clip(0, 1)
    )
    feat["daily_avg_time"] = (
        feat["daily_avg_time"]
        .fillna(rt_med)
        .infer_objects(copy=False)
        .astype(float)
        .clip(0, 600)
    )
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
# 4) 예측 엔드포인트 (타임존/윈도우 패치 반영)
# -----------------------------
@router.get("/predict-accuracy-live")
def predict_accuracy_live(
    patient_id: str = Query(..., description="환자 ID"),
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$", description="예측 대상 월 (YYYY-MM)")
):
    try:
        # 1) 데이터 로드
        df = load_logs_from_api(patient_id)
        print(f"[DEBUG] 로드된 데이터 수: {len(df)}")
        if not df.empty:
            print(f"[DEBUG] (raw) 날짜 범위: {df['created_at'].min()} ~ {df['created_at'].max()}")

        # 2) created_at을 'UTC tz-aware'로 표준화
        if not df.empty:
            df["created_at"] = ensure_utc_series(df["created_at"])
            df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
            print(f"[DEBUG] (UTC 표준화) 날짜 범위: {df['created_at'].min()} ~ {df['created_at'].max()}")

        # 3) KST 월 경계 → UTC 윈도우(반열린) 계산
        try:
            start_utc, end_utc = kst_month_window_utc(month)  # [start_utc, end_utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="month must be YYYY-MM")
        print(f"[DEBUG] KST 기준 월 경계(UTC): [{start_utc} ~ {end_utc})")

        # 4) 모델 입력용 과거 W일 윈도우
        #    타깃 월 시작 직전 시점까지를 윈도우로 사용.
        #    예: 2025-08의 start_utc=2025-07-31 15:00:00Z라면,
        #        wnd_end = start_utc - 1초, wnd_start = wnd_end - (W-1)일
        wnd_end = start_utc - timedelta(seconds=1)
        wnd_start = wnd_end - timedelta(days=W - 1)
        print(f"[DEBUG] 모델 윈도우(UTC): {wnd_start} ~ {wnd_end}")

        # 5) 데이터가 없는 경우 (콜드 스타트)
        if df.empty:
            print("[DEBUG] 데이터가 비어있음 - Cold start")
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

        # 6) 형 변환 등 전처리
        df["is_correct"] = df["is_correct"].astype(int).clip(0, 1)

        # 7) 윈도우 내 시도 수 (UTC 윈도우 기준)
        in_wnd = df[(df["created_at"] >= wnd_start) & (df["created_at"] <= wnd_end)]
        attempts = int(in_wnd.shape[0])
        print(f"[DEBUG] 윈도우 내 시도 횟수: {attempts}")

        # 8) 일 단위 집계 만들기 (wnd_end 이전까지만 사용)
        daily = _make_daily(df[df["created_at"] <= wnd_end].copy())

        # 9) 피처 생성
        X = _build_features(daily, wnd_start, wnd_end, W)
        if X.shape[1] != F:
            raise HTTPException(status_code=500, detail=f"입력 차원 불일치: X({X.shape[1]}) vs 모델({F})")

        # 10) 활성 일수/커버리지 계산
        active_days = 0
        if not daily.empty:
            # ✅ wnd_start/wnd_end는 이미 tz-aware(UTC)이므로 tz= 파라미터 없이 변환
            # 이미 tz-aware인 값에 tz= 파라미터를 주면 "Cannot pass a datetime or Timestamp with tzinfo with the tz parameter" 에러 발생
            wnd_start_ts = pd.Timestamp(wnd_start)
            wnd_end_ts = pd.Timestamp(wnd_end)
            mask = (daily["date"] >= wnd_start_ts) & (daily["date"] <= wnd_end_ts)
            active_days = int(daily.loc[mask].shape[0])

        print(f"[DEBUG] 활성 일수: {active_days}, 임계값: {COLD_DAYS_OK}")
        print(f"[DEBUG] 시도 횟수: {attempts}, 임계값: {COLD_ATTEMPTS_OK}")

        day_coverage = active_days / float(W) if W > 0 else 0.0
        expected_attempts = max(1, W * EXPECTED_DAILY_SOLVES)
        attempt_coverage = min(1.0, attempts / float(expected_attempts))
        coverage = max(day_coverage, attempt_coverage)

        # 11) 모델 예측
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            raw_pred = float(_model(x).item())
        raw_pred = max(0.0, min(1.0, raw_pred))

        # 12) 블렌딩 및 cold-start 판정
        blended = coverage * raw_pred + (1.0 - coverage) * BASELINE_ACC
        cold_start_flag = not (active_days >= COLD_DAYS_OK or attempts >= COLD_ATTEMPTS_OK)

        print(f"[DEBUG] Cold start 판정: {cold_start_flag}")
        print(f"[DEBUG] Coverage: {coverage}, Raw pred: {raw_pred}, Blended: {blended}")

        return {
            "patient_id": patient_id,
            "month": month,
            "predicted_final_accuracy": round(blended, 4),
            "cold_start": cold_start_flag,
            "ui": {"badges": ["초기 예측 — 더 많은 풀이가 쌓일수록 정확도가 올라가요."] if cold_start_flag else []}
        }

    except HTTPException:
        raise
    except Exception as e:
        # tz 관련 오류 메시지를 그대로 노출해 디버깅 용이
        raise HTTPException(status_code=500, detail=f"[API] {type(e).__name__}: {e}")