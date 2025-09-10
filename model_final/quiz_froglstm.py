import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW      = 45
HORIZON     = 10
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
TRAIN_RATIO = 0.7
EARLY_STOP  = 5
CSV_PATH    = "quiz_logs.csv"
MODEL_PATH  = "best_model_10d.pt"
PRED_CSV    = "predictions.csv"

def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=["created_at"])
    df = df.rename(columns={"created_at":"timestamp"})
    df["date"] = df["timestamp"].dt.date
    agg = (
        df.groupby(["patient_id","date"])
          .agg(
              correct_rate=("is_correct","mean"),
              avg_time=("response_time_sec","mean")
          )
          .reset_index()
    )
    agg["weekday"] = pd.to_datetime(agg["date"]).dt.weekday
    agg = pd.get_dummies(agg, columns=["weekday"])

    def fill_dates(g):
        dates = pd.date_range(g.date.min(), g.date.max())
        filled = (
            g.set_index("date")
             .reindex(dates)
             .ffill().bfill()
             .reset_index()
             .rename(columns={"index":"date"})
        )
        filled["patient_id"] = g.name
        return filled

    agg = agg.groupby("patient_id", group_keys=False).apply(fill_dates).reset_index(drop=True)
    return agg

class SequenceDataset(Dataset):
    def __init__(self, df, window, horizon, scaler=None):
        self.window  = window
        self.horizon = horizon
        self.features = [c for c in df.columns if c not in ["patient_id","date"]]
        self.scaler = scaler if scaler else StandardScaler()
        self.X, self.y, self.info = self._prepare(df)

    def _prepare(self, df):
        data_all = df[self.features].values
        self.scaler.fit(data_all)
        Xs, ys, infos = [], [], []
        for pid, group in df.groupby("patient_id"):
            group = group.sort_values("date")
            data = self.scaler.transform(group[self.features].values)
            L = len(data)
            for i in range(L - self.window - self.horizon + 1):
                seq_x = data[i:i+self.window]
                future = data[i+self.window:i+self.window+self.horizon]
                Xs.append(seq_x)
                ys.append(future[:,0].mean())
                infos.append({
                    "patient_id": pid,
                    "input_start": str(group["date"].iloc[i]),
                    "input_end":   str(group["date"].iloc[i+self.window-1]),
                    "pred_start":  str(group["date"].iloc[i+self.window]),
                    "pred_end":    str(group["date"].iloc[i+self.window+self.horizon-1])
                })
        return (
            torch.tensor(np.stack(Xs), dtype=torch.float32),
            torch.tensor(ys, dtype=torch.float32).unsqueeze(1),
            infos
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.info[idx]

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        out,_ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y, _ in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*X.size(0)
    return total_loss/len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_mse, total_mae = 0,0
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            total_mse += criterion(pred, y).item()*X.size(0)
            total_mae += torch.abs(pred-y).sum().item()
    return total_mse/len(loader.dataset), total_mae/len(loader.dataset)

def predict_all(model, dataset):
    model.eval()
    results=[]
    with torch.no_grad():
        for i in range(len(dataset)):
            X = dataset.X[i].unsqueeze(0).to(DEVICE)
            pred = torch.sigmoid(model(X)).item()
            info = dataset.info[i]
            results.append({
                "patient_id": info["patient_id"],
                "input_start": info["input_start"],
                "input_end": info["input_end"],
                "pred_start": info["pred_start"],
                "pred_end": info["pred_end"],
                "pred_10d": pred
            })
    return pd.DataFrame(results)

def main():
    df = load_and_preprocess(CSV_PATH)
    dataset = SequenceDataset(df, WINDOW, HORIZON)
    indices = np.arange(len(dataset))
    tr_idx, te_idx = train_test_split(indices, train_size=TRAIN_RATIO, random_state=42)
    val_idx, te_idx = train_test_split(te_idx, test_size=0.5, random_state=42)
    train_loader = DataLoader(Subset(dataset,tr_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(dataset,val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(Subset(dataset,te_idx), batch_size=BATCH_SIZE, shuffle=False)
    input_dim = len(dataset.features)
    model = LSTMPredictor(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    history = {"train_loss":[], "val_mse":[], "val_mae":[]}
    best_val = float('inf')
    patience = 0
    for ep in range(1,EPOCHS+1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_mse, val_mae = eval_epoch(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        print(f"Epoch {ep:02d} ▶ train loss: {tr_loss:.4f}, val MSE: {val_mse:.4f}, val MAE: {val_mae:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), MODEL_PATH)
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f"⏹ Early stopping at epoch {ep}")
                break
    plt.figure(figsize=(8,4))
    plt.plot(history["train_loss"], label="Train Loss (MSE)")
    plt.plot(history["val_mse"], label="Val MSE")
    plt.plot(history["val_mae"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MAE")
    plt.title("Training Curves")
    plt.legend()
    plt.savefig("training_curves.png")
    plt.close()
    pred_df = predict_all(model, dataset)
    pred_df.to_csv(PRED_CSV, index=False)
    print(f"예측 완료 → {PRED_CSV}")

if __name__=="__main__":
    main()
