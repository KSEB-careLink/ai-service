import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RecommendRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RecommendRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h_n, c_n) = self.rnn(x)
        out = self.fc(h_n[-1])
        return out

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(base_dir, '..', 'models', 'quiz_recommend_rnn.pth'))
dataset_path = os.path.normpath(os.path.join(base_dir, '..', 'data', 'dataset.json'))
results_path = os.path.normpath(os.path.join(base_dir, '..', 'data', 'results.json'))
stats_path = os.path.normpath(os.path.join(base_dir, '..', 'models', 'topic_stats.json'))
topics_path = os.path.normpath(os.path.join(base_dir, '..', 'models', 'question_topics.json'))

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
output_dim = len(dataset)

input_dim = 3
hidden_dim = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommendRNN(input_dim, hidden_dim, output_dim).to(device)

print("🔎 모델 파일 경로:", model_path)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 모델 로드 완료!")

with open(results_path, 'r', encoding='utf-8') as f:
    all_results = json.load(f)

new_data = []
for item in all_results:
    q_id = item["question_id"]
    correct = 1 if item["correct"] else 0
    time_taken = item.get("time_taken", 0)
    new_data.append([q_id, correct, time_taken])

X_tensor = torch.tensor([new_data], dtype=torch.float32).to(device)
print("🔥 전체 시퀀스 길이:", len(new_data))
print("✅ X_tensor shape:", X_tensor.shape)

with torch.no_grad():
    output = model(X_tensor)
    probs = F.softmax(output, dim=1)

with open(stats_path, 'r', encoding='utf-8') as f:
    stats_data = json.load(f)
difficult_topics = stats_data.get('difficult_topics', [])
print("🔥 어려운 주제:", difficult_topics)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def classify_topic(question_text: str) -> str:
    prompt = f"""
    다음 퀴즈 문제의 주제를 아래 중 하나로 분류해줘:
    [가족, 여행, 학창시절, 동네, 환자가 제일 행복했던 기억]
    퀴즈 문제: "{question_text}"
    답변은 반드시 위의 다섯 중 하나만 출력해.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 문장의 주제를 정확히 분류하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20,
        )
        topic = response.choices[0].message.content.strip()
        return topic
    except Exception as e:
        print("❌ LLM 호출 오류:", e)
        return "기타"

if os.path.exists(topics_path):
    with open(topics_path, 'r', encoding='utf-8') as f:
        id_to_topic = json.load(f)
    print(f"📂 기존 주제 매핑 로드 완료! ({len(id_to_topic)}개)")
else:
    id_to_topic = {}
    for item in dataset:
        q_text = item['question']
        topic = classify_topic(q_text)
        id_to_topic[item['question_id']] = topic
    with open(topics_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_topic, f, ensure_ascii=False, indent=2)
    print(f"📁 주제 매핑이 {topics_path} 에 저장되었습니다!")

adjusted_scores = []
for idx, prob in enumerate(probs.cpu().numpy()[0]):
    topic = id_to_topic.get(str(idx)) or id_to_topic.get(idx)
    score = prob
    if topic in difficult_topics:
        score *= 1.5
    adjusted_scores.append((idx, score))

adjusted_scores.sort(key=lambda x: x[1], reverse=True)

difficult_list = [idx for idx, _ in adjusted_scores if id_to_topic.get(str(idx)) in difficult_topics]
normal_list = [idx for idx, _ in adjusted_scores if id_to_topic.get(str(idx)) not in difficult_topics]

num_difficult = 1
num_normal = 2
final_recommendations = difficult_list[:num_difficult] + normal_list[:num_normal]

print("✨ 최종 추천 문제 ID:", final_recommendations)

for rec_id in final_recommendations:
    q = next((q for q in dataset if q['question_id'] == rec_id), None)
    if q:
        print(f"📌 추천 문제 (ID {rec_id}): {q['question']}")
        print(f"📝 선택지: {q['options']}")
