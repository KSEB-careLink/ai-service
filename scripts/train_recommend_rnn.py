import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os

base_dir = os.path.dirname(os.path.abspath(__file__))   
data_path = os.path.join(base_dir, '..', 'data', 'results.json')  
data_path = os.path.normpath(data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

features = []
labels = []

for item in data:
    q_id = item["question_id"]
    correct = 1 if item["correct"] else 0
    time_taken = item.get("time_taken", 0)

    features.append([q_id, correct, time_taken])
    labels.append(q_id)

X_tensor = torch.tensor(features, dtype=torch.float32)
Y_tensor = torch.tensor(labels, dtype=torch.long)

print("✅ X_tensor:", X_tensor.shape)
print("✅ Y_tensor:", Y_tensor.shape)

class QuizResultDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = QuizResultDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

class RecommendRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RecommendRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1) 
        out, (h_n, c_n) = self.rnn(x)
        out = self.fc(h_n[-1])  
        return out

input_dim = 3
hidden_dim = 64
output_dim = max(labels) + 1 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RecommendRNN(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_X, batch_Y in dataloader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")

model_path = os.path.join(base_dir, '..', 'models', 'quiz_recommend_rnn.pth')
model_path = os.path.normpath(model_path)
torch.save(model.state_dict(), model_path)

print(f"🎉 학습 완료! {model_path} 로 저장되었습니다.")
