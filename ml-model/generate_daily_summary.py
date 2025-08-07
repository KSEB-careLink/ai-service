import pandas as pd
from datetime import timedelta

input_path = "daily_summary.csv"
output_path = "monthly_summary.csv"

df = pd.read_csv(input_path)
df["solved_date"] = pd.to_datetime(df["solved_date"])

train_rows = []

for user_id, group in df.groupby("patient_id"):
    group = group.sort_values("solved_date").reset_index(drop=True)

    for i in range(90, len(group)):
        window = group.iloc[i-90:i]
        predict_date = group.iloc[i]["solved_date"]

        next_month_start = (predict_date + pd.offsets.MonthBegin(1)).replace(day=1)
        next_month_end = next_month_start + pd.offsets.MonthEnd(0)

        next_month_data = group[
            (group["solved_date"] >= next_month_start) &
            (group["solved_date"] <= next_month_end)
        ]

        if len(next_month_data) == 0:
            continue 

        target_acc = next_month_data["daily_acc_rate"].mean()

        train_rows.append({
            "patient_id": user_id,
            "predict_date": predict_date.date(),
            "avg_acc_rate_90d": window["daily_acc_rate"].mean(),
            "avg_time_90d": window["daily_avg_time"].mean(),
            "target_acc_rate": target_acc
        })

train_df = pd.DataFrame(train_rows)
train_df.to_csv(output_path, index=False)
print(f"✅ 저장 완료: {output_path} (총 {len(train_df)}개 row)")
