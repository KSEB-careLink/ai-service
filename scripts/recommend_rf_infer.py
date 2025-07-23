import json
import os
import random
from collections import defaultdict
import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'quiz_rf_model.pkl')
topic_mapping_path = os.path.join(base_dir, '..', 'models', 'question_topics.json')
results_path = os.path.join(base_dir, '..', 'data', 'results.json')
dataset_path = os.path.join(base_dir, '..', 'data', 'dataset.json')

with open(topic_mapping_path, 'r', encoding='utf-8') as f:
    topic_mapping = json.load(f)

with open(results_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

topic_stats = defaultdict(lambda: {"count": 0, "correct_count": 0, "time_sum": 0})
for r in results:
    qid = str(r["question_id"])
    topic = topic_mapping.get(qid, "기타")
    topic_stats[topic]["count"] += 1
    if r.get("correct", False):
        topic_stats[topic]["correct_count"] += 1
    topic_stats[topic]["time_sum"] += r.get("time_taken", 0)

topic_stats_final = {}
for topic, stat in topic_stats.items():
    if stat["count"] == 0:
        continue
    acc = stat["correct_count"] / stat["count"]
    avg_time = stat["time_sum"] / stat["count"] if stat["count"] > 0 else 0
    topic_stats_final[topic] = {
        "count": stat["count"],
        "correct_count": stat["correct_count"],
        "time_sum": stat["time_sum"],
        "accuracy": round(acc, 3),
        "avg_time": round(avg_time, 3)
    }

topic_scores = []
for topic, stat in topic_stats_final.items():
    wrong_rate = 1 - stat["accuracy"]
    topic_scores.append((topic, wrong_rate, stat["avg_time"]))

topic_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

if topic_scores:
    weakest_topic = topic_scores[0][0]
else:
    weakest_topic = None

print(f"🔥 취약 주제: {weakest_topic}")

model = joblib.load(model_path)
print("✅ 랜덤 포레스트 모델 로드 완료!")

weakest_candidates = [
    d for d in dataset if topic_mapping.get(str(d["question_id"]), "기타") == weakest_topic
]
weakest_question = random.choice(weakest_candidates) if weakest_candidates else None

other_candidates = [
    d for d in dataset if topic_mapping.get(str(d["question_id"]), "기타") != weakest_topic
]
random_questions = random.sample(other_candidates, k=2) if len(other_candidates) >= 2 else other_candidates

recommended = []
if weakest_question:
    recommended.append(weakest_question)
recommended.extend(random_questions)

print("\n✨ 추천된 문제:")
for q in recommended:
    print(f"📌 ID {q['question_id']} ({topic_mapping.get(str(q['question_id']), '기타')}): {q['question']}")
    print(f"📝 선택지: {q['options']}")
    print(f"✅ 정답: {q['answer']}")
    print("-" * 50)

recommended_output = [
    {
        "question_id": q["question_id"],
        "question": q["question"],
        "options": q["options"],
        "answer": q["answer"]
    }
    for q in recommended
]
recommended_path = os.path.join(base_dir, '..', 'models', 'recommended_rf_questions.json')
with open(recommended_path, 'w', encoding='utf-8') as f:
    json.dump(recommended_output, f, ensure_ascii=False, indent=2)
print(f"✅ 추천 문제 정보가 {recommended_path} 로 저장되었습니다.")

topic_analysis_output = {
    "weakest_topic": weakest_topic,
    "topic_stats": topic_stats_final
}
topic_analysis_path = os.path.join(base_dir, '..', 'models', 'topic_analysis.json')
with open(topic_analysis_path, 'w', encoding='utf-8') as f:
    json.dump(topic_analysis_output, f, ensure_ascii=False, indent=2)
print(f"✅ 주제 분석 정보가 {topic_analysis_path} 로 저장되었습니다.")
