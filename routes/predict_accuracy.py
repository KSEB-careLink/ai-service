# routes/predict_accuracy.py
import os, json
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException, Query

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
        
        # 수정: 한 번만 요청하고 patient_id 파라미터 유지
        response = requests.get(
            url, 
            params={"patient_id": patient_id},  # 파라미터 유지
            timeout=NODE_API_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        
        # HTTP 에러 체크
        if response.status_code == 404:
            # 환자 데이터가 없는 경우 빈 DataFrame 반환
            return pd.DataFrame(columns=[
                "patient_id", "quiz_id", "selected_index", 
                "is_correct", "response_time_sec", "created_at"
            ])
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Node.js API 오류: {response.text}"
            )
        
        # JSON 응답 파싱
        data = response.json()
        
        # API 응답이 리스트가 아닌 경우 처리
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
        
        # DataFrame으로 변환
        df = pd.DataFrame(quiz_logs)
        
        # 필수 컬럼 체크
        required_columns = ["patient_id", "quiz_id", "selected_index", 
                          "is_correct", "response_time_sec", "created_at"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=500,
                detail=f"API 응답에서 누락된 컬럼: {missing_columns}"
            )
        
        # 날짜 파싱
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
            # 날짜 파싱 실패한 행 제거
            df = df.dropna(subset=["created_at"])
        
        return df
        
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504, 
            detail="Node.js API 응답 시간 초과"
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503, 
            detail="Node.js API 연결 실패"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500, 
            detail=f"API 호출 오류: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"데이터 처리 오류: {type(e).__name__}: {e}"
        )

# -----------------------------
# 3) 전처리 유틸 (기존과 동일)
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
# 4) 예측 엔드포인트 (수정됨)
# -----------------------------
@router.get("/predict-accuracy-live")
def predict_accuracy_live(
    patient_id: str = Query(..., description="환자 ID"),
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$", description="예측 대상 월 (YYYY-MM)")
):
    try:
        # Node.js API에서 데이터 로드
        df = load_logs_from_api(patient_id)
        
        print(f"[DEBUG] 로드된 데이터 수: {len(df)}")
        if not df.empty:
            print(f"[DEBUG] 날짜 범위: {df['created_at'].min()} ~ {df['created_at'].max()}")

        try:
            M_start = datetime.strptime(month + "-01", "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="month must be YYYY-MM")

        wnd_end = M_start - timedelta(days=1)
        wnd_start = wnd_end - timedelta(days=W - 1)
        
        print(f"[DEBUG] 윈도우 범위: {wnd_start} ~ {wnd_end}")

        # 데이터가 없는 경우 (콜드 스타트)
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

        # 전처리
        df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
        df["is_correct"] = df["is_correct"].astype(int).clip(0, 1)

        # 윈도우 집계
        in_wnd = df[(df["created_at"] >= wnd_start) & (df["created_at"] <= wnd_end)]
        attempts = int(in_wnd.shape[0])
        
        print(f"[DEBUG] 윈도우 내 시도 횟수: {attempts}")

        daily = _make_daily(df[df["created_at"] <= wnd_end].copy())
        X = _build_features(daily, wnd_start, wnd_end, W)
        if X.shape[1] != F:
            raise HTTPException(status_code=500, detail=f"입력 차원 불일치: X({X.shape[1]}) vs 모델({F})")

        active_days = 0
        if not daily.empty:
            mask = (daily["date"] >= pd.Timestamp(wnd_start)) & (daily["date"] <= pd.Timestamp(wnd_end))
            active_days = int(daily.loc[mask].shape[0])

        print(f"[DEBUG] 활성 일수: {active_days}, 임계값: {COLD_DAYS_OK}")
        print(f"[DEBUG] 시도 횟수: {attempts}, 임계값: {COLD_ATTEMPTS_OK}")

        day_coverage = active_days / float(W) if W > 0 else 0.0
        expected_attempts = max(1, W * EXPECTED_DAILY_SOLVES)
        attempt_coverage = min(1.0, attempts / float(expected_attempts))
        coverage = max(day_coverage, attempt_coverage)

        # 모델 예측
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            raw_pred = float(_model(x).item())
        raw_pred = max(0.0, min(1.0, raw_pred))

        # 블렌딩
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
        # 이미 의미있는 detail을 설정한 경우 그대로 전달
        raise
    except Exception as e:
        # 어디서든 터지면 JSON detail로 보여주기
        raise HTTPException(status_code=500, detail=f"[API] {type(e).__name__}: {e}")