import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="quiz_logs.csv (원본 전체 로그)")
    ap.add_argument("--pred_csv", required=True, help="예측 결과 CSV (patient별 월 평균 예측치)")
    ap.add_argument("--patient_id", required=True)
    args = ap.parse_args()

    # 실제 로그 불러오기
    df = pd.read_csv(args.csv)
    df = df[df["patient_id"] == args.patient_id].copy()
    if df.empty:
        raise RuntimeError(f"no rows for patient_id={args.patient_id}")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").clip(0,1)

    # 월 단위 실제 평균 정답률
    df["month"] = df["created_at"].dt.to_period("M")
    gt_month = df.groupby("month")["is_correct"].mean().astype(float)

    # 예측 결과 불러오기
    pred_df = pd.read_csv(args.pred_csv)
    pred_df = pred_df[pred_df["patient_id"] == args.patient_id].copy()
    if "pred_month_avg" not in pred_df.columns:
        raise RuntimeError("예측 CSV에 pred_month_avg 컬럼이 없음")

    # 평가 집계
    rows = []
    for _, row in pred_df.iterrows():
        ym = row["month"]
        pred = row["pred_month_avg"]
        actual = float(gt_month.get(ym, np.nan)) if ym in gt_month.index else None
        if actual is None or np.isnan(actual) or pd.isna(pred):
            continue
        mae = abs(pred - actual)
        rmse = (pred - actual) ** 2
        rows.append({"month": ym, "pred": pred, "actual": actual, "mae": mae, "rmse": rmse})

    if not rows:
        raise RuntimeError("no overlapping months between prediction and actual")

    eval_df = pd.DataFrame(rows)
    # 전체 평균
    acc = (1 - eval_df["mae"].mean()) * 100   # Accuracy = 1 - MAE (간단 정의)
    mae = eval_df["mae"].mean()
    rmse = np.sqrt(eval_df["rmse"].mean())

    print("\n[월 단위 평가 결과]")
    print(eval_df)
    print("\n요약:")
    print(f"- Accuracy (월평균): {acc:.2f}%")
    print(f"- MAE (월평균 절대오차): {mae:.4f}")
    print(f"- RMSE (월평균 제곱근오차): {rmse:.4f}")

if __name__ == "__main__":
    main()
