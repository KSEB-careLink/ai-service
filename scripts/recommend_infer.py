import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

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
model_path = os.path.join(base_dir, '..', 'models', 'quiz_recommend_rnn.pth')
dataset_path = os.path.join(base_dir, '..', 'data', 'dataset.json')
stats_path = os.path.join(base_dir, '..', 'models', 'topic_stats.json')

model_path = os.path.normpath(model_path)
dataset_path = os.path.normpath(dataset_path)
stats_path = os.path.normpath(stats_path)

input_dim = 3
hidden_dim = 64
output_dim = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommendRNN(input_dim, hidden_dim, output_dim).to(device)

print("🔎 모델 파일 경로:", model_path)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 모델 로드 완료!")

new_data = [
    [47, 0, 6],
    [48, 1, 5],
    [49, 1, 2],
]
X_tensor = torch.tensor([new_data], dtype=torch.float32).to(device)

with torch.no_grad():
    output = model(X_tensor)
    probs = F.softmax(output, dim=1)  

with open(stats_path, 'r', encoding='utf-8') as f:
    stats_data = json.load(f)
difficult_topics = stats_data.get('difficult_topics', [])
print("🔥 어려운 주제:", difficult_topics)

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
id_to_topic = {item['question_id']: item['topic'] for item in dataset}

adjusted_scores = []
for idx, prob in enumerate(probs.cpu().numpy()[0]):  
    topic = id_to_topic.get(idx)
    score = prob
    if topic in difficult_topics:
        score *= 1.5  
    adjusted_scores.append((idx, score))

adjusted_scores.sort(key=lambda x: x[1], reverse=True)

difficult_list = [idx for idx, _ in adjusted_scores if id_to_topic.get(idx) in difficult_topics]
normal_list = [idx for idx, _ in adjusted_scores if id_to_topic.get(idx) not in difficult_topics]

num_difficult = 1
num_normal = 2
final_recommendations = difficult_list[:num_difficult] + normal_list[:num_normal]

print("✨ 최종 추천 문제 ID:", final_recommendations)

for rec_id in final_recommendations:
    q = next((q for q in dataset if q['question_id'] == rec_id), None)
    if q:
        print(f"📌 추천 문제 (ID {rec_id}): {q['question']}")
        print(f"📝 선택지: {q['options']}")
