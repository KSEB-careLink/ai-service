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
    
input_dim = 3
hidden_dim = 64
output_dim = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommendRNN(input_dim, hidden_dim, output_dim).to(device)

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'quiz_recommend_rnn.pth')
model_path = os.path.abspath(model_path)  

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
    predicted_class = torch.argmax(probs, dim=1).item()

print("🔎 출력 확률:", probs.cpu().numpy())
print("✨ 추천 결과 클래스 index:", predicted_class)

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.json')
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

recommended_id = predicted_class
recommended_question = next((q for q in dataset if q['question_id'] == recommended_id), None)

if recommended_question:
    print("📌 추천 문제:", recommended_question['question'])
    print("📝 선택지:", recommended_question['options'])
else:
    print("❌ 해당 ID의 문제를 찾을 수 없습니다.")