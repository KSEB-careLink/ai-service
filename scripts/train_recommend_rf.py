import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

base_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(base_dir, '..', 'data', 'results.json')
topic_mapping_path = os.path.join(base_dir, '..', 'models', 'question_topics.json')
model_output_path = os.path.join(base_dir, '..', 'models', 'quiz_rf_model.pkl')

with open(results_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

with open(topic_mapping_path, 'r', encoding='utf-8') as f:
    topic_mapping = json.load(f)  # { "0": "가족", ... }

X = []
y = []

all_topics = list(set(topic_mapping.values()))
topic_encoder = LabelEncoder()
topic_encoder.fit(all_topics)

for r in results:
    qid = str(r["question_id"])
    topic_str = topic_mapping.get(qid, "기타")
    topic_num = topic_encoder.transform([topic_str])[0]
    time_taken = r.get("time_taken", 0)
    X.append([int(qid), topic_num, time_taken])
    y.append(1 if r["correct"] else 0)

print(f"✅ 총 데이터 개수: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,       # 트리 개수
    max_depth=None,        # 깊이 제한 없음
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 테스트 세트 정확도: {acc:.4f}")

joblib.dump({
    "model": model,
    "topic_encoder": topic_encoder
}, model_output_path)

print(f"✅ 모델이 {model_output_path} 에 저장되었습니다.")
