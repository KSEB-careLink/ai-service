import json
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

base_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(base_dir, '..', 'data', 'results.json')
topic_mapping_path = os.path.join(base_dir, '..', 'models', 'question_topics.json')
model_output_path = os.path.join(base_dir, '..', 'models', 'quiz_rf_model.pkl')

# 🔹 데이터 불러오기
with open(results_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

with open(topic_mapping_path, 'r', encoding='utf-8') as f:
    topic_mapping = json.load(f)

# 🔹 시간 기반 전처리를 위한 정렬 및 준비
results.sort(key=lambda x: (x.get("user_id", "default"), x.get("timestamp", "")))

user_day_map = {}       # { user_id: {date_str: day_index} }
user_last_date = {}     # { user_id: last_date }
user_daily_count = {}   # { (user_id, date_str): count }
user_prev_correct = {}  # { user_id: 이전 정답 여부 }

X = []
y = []

all_topics = list(set(topic_mapping.values()))
topic_encoder = LabelEncoder()
topic_encoder.fit(all_topics)

for r in results:
    user_id = r.get("user_id", "default")
    qid = str(r["question_id"])
    topic_str = topic_mapping.get(qid, "기타")
    topic_num = topic_encoder.transform([topic_str])[0]
    time_taken = r.get("time_taken", 0)
    correct = 1 if r.get("correct", False) else 0

    # 🔹 날짜 정보 파싱
    timestamp_str = r.get("timestamp")
    try:
        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except:
        continue  # 날짜 형식 이상한 건 제외

    date_str = ts.strftime("%Y-%m-%d")

    # 🔹 day_index 계산
    if user_id not in user_day_map:
        user_day_map[user_id] = {date_str: 1}
        user_last_date[user_id] = date_str
        day_index = 1
    elif date_str not in user_day_map[user_id]:
        last_idx = max(user_day_map[user_id].values())
        user_day_map[user_id][date_str] = last_idx + 1
        day_index = last_idx + 1
        user_last_date[user_id] = date_str
    else:
        day_index = user_day_map[user_id][date_str]

    # 🔹 seq_id 계산
    user_daily_key = (user_id, date_str)
    if user_daily_key not in user_daily_count:
        user_daily_count[user_daily_key] = 1
    else:
        user_daily_count[user_daily_key] += 1
    seq_id = user_daily_count[user_daily_key]

    # 🔹 previous_correct 계산
    prev = user_prev_correct.get(user_id, 0)
    user_prev_correct[user_id] = correct  # 이번 정답 여부 저장

    # 🔹 학습 데이터 포맷 추가
    X.append([int(qid), topic_num, time_taken, day_index, seq_id, prev])
    y.append(correct)

print(f"✅ 총 데이터 개수: {len(X)}")

# 🔹 학습 및 평가
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 테스트 세트 정확도: {acc:.4f}")

# 🔹 모델 저장
joblib.dump({
    "model": model,
    "topic_encoder": topic_encoder
}, model_output_path)

print(f"✅ 모델이 {model_output_path} 에 저장되었습니다.")
