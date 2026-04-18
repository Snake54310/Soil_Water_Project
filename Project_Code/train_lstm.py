# train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F   # ← Fixed crash
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler

INPUT_SIZE         = 12
SEQ_LEN            = 30
THRESHOLD_LR       = 0.001
GAIN_LR            = 0.005
SENSITIVITY_SCALE  = 5.0
RMSE_GOOD_THRESHOLD = 2.0
LOOKAHEAD_ROWS     = 144

def compute_gain_target(avg_error):
    target = 1.0 - math.tanh(avg_error / SENSITIVITY_SCALE)
    target = max(0.1, min(2.0, target))
    return torch.tensor([target, target, target], dtype=torch.float32)

def online_update_threshold(feature_seq, target_prob, model, optimizer, pos_weight=10.0):
    """Fixed: uses F.binary_cross_entropy with correct (1,1) shapes — no more crash."""
    model.train()
    optimizer.zero_grad()
    pred = model(feature_seq)                    # shape (1, 1) — do NOT squeeze
    target = torch.tensor([[target_prob]], dtype=torch.float32)
    weight = torch.tensor([[pos_weight if target_prob == 1.0 else 1.0]],
                          dtype=torch.float32)
    loss = F.binary_cross_entropy(pred, target, weight=weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()
    torch.save(model.state_dict(), "lstm_threshold.pth")
    return loss.item()

def online_update_gain_scheduler(feature_seq, avg_error, model, optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(feature_seq).squeeze()
    target = compute_gain_target(avg_error)
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()
    torch.save(model.state_dict(), "lstm_gain_scheduler.pth")
    return {
        'loss': loss.item(),
        'target_mult': target[0].item(),
        'predicted_kp': pred[0].item(),
        'predicted_ki': pred[1].item(),
        'predicted_kd': pred[2].item()
    }

# ... (the rest of the file — WateringDataset, GainSchedulerDataset, train functions — is unchanged)
class WateringDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['sensor_errors'] == 0].reset_index(drop=True)
        self.features = df[[
            'air_t_avg_since_last', 'ground_t_avg_since_last', 'humidity_avg_since_last',
            'air_t_current', 'ground_t_current', 'humidity_current',
            'soil_moisture_current', 'soil_moisture_corrected', 'soil_moisture_rolling_avg',
            'avg_error_since_last', 'waterings_today', 'minutes_until_next_allowed'
        ]].values.astype(np.float32)
        self.labels = df['pump_on'].values.astype(np.float32)

    def __len__(self):
        return max(0, len(self.features) - SEQ_LEN)

    def __getitem__(self, idx):
        x = self.features[idx:idx + SEQ_LEN]
        y = self.labels[idx + SEQ_LEN - 1]
        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)

class GainSchedulerDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['sensor_errors'] == 0].reset_index(drop=True)
        self.features = []
        self.targets = []
        self.weights = []
        for i in range(SEQ_LEN, len(df)):
            if df.iloc[i]['pump_on'] == 1:
                x = df.iloc[i-SEQ_LEN:i][[
                    'air_t_avg_since_last', 'ground_t_avg_since_last', 'humidity_avg_since_last',
                    'air_t_current', 'ground_t_current', 'humidity_current',
                    'soil_moisture_current', 'soil_moisture_corrected', 'soil_moisture_rolling_avg',
                    'avg_error_since_last', 'waterings_today', 'minutes_until_next_allowed'
                ]].values.astype(np.float32)
                avg_error = df.iloc[i]['avg_error_since_last']
                target = compute_gain_target(avg_error)
                window = df.iloc[i:i+LOOKAHEAD_ROWS]['soil_moisture_corrected'].values
                if len(window) > 0:
                    rmse = np.sqrt(np.mean((window - 50.0)**2))
                    weight = 1.0 / (1.0 + rmse)
                else:
                    weight = 1.0
                self.features.append(x)
                self.targets.append(target)
                self.weights.append(weight)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), self.targets[idx], self.weights[idx]

def train_threshold_model(csv_path, epochs=80):
    dataset = WateringDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTM_Threshold(input_size=INPUT_SIZE)
    try:
        model.load_state_dict(torch.load("lstm_threshold.pth", weights_only=True, map_location='cpu'))
        print("Loaded existing threshold model.")
    except FileNotFoundError:
        print("No threshold model found - starting fresh.")

    optimizer = optim.Adam(model.parameters(), lr=THRESHOLD_LR)
    pos_weight = max(1.0, (dataset.labels == 0).sum() / max((dataset.labels == 1).sum(), 1))

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss_fn = nn.BCELoss(pos_weight=torch.tensor([pos_weight]))
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Threshold Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "lstm_threshold.pth")
    print("Threshold model training complete.")

def train_gain_scheduler(csv_path, epochs=40):
    dataset = GainSchedulerDataset(csv_path)
    if len(dataset) < 10:
        print("Not enough watering events for gain scheduler training.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTM_GainScheduler(input_size=INPUT_SIZE)
    try:
        model.load_state_dict(torch.load("lstm_gain_scheduler.pth", weights_only=True, map_location='cpu'))
        print("Loaded existing gain scheduler model.")
    except FileNotFoundError:
        print("No gain scheduler model found - starting fresh.")

    optimizer = optim.Adam(model.parameters(), lr=GAIN_LR)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, target, weight in dataloader:
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = nn.MSELoss()(pred, target) * weight
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Gain Scheduler Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "lstm_gain_scheduler.pth")
    print("Gain scheduler training complete.")

if __name__ == "__main__":
    train_threshold_model('watering_log.csv', epochs=80)
    train_gain_scheduler('watering_log.csv', epochs=40)