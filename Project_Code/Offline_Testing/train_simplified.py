import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from datetime import timedelta

MOISTURE_TARGET = 63.0
SEQ_LEN = 30

# Global normalization
df_global = pd.read_csv('watering_log.csv')
soil_mean = df_global['soil_moisture_rolling_avg'].mean()
soil_std  = df_global['soil_moisture_rolling_avg'].std() + 1e-8
error_mean = (df_global['soil_moisture_rolling_avg'] - MOISTURE_TARGET).mean()
error_std  = (df_global['soil_moisture_rolling_avg'] - MOISTURE_TARGET).std() + 1e-8

def normalize_features(seq):
    soil = seq[:, 0]
    error = seq[:, 1]
    return torch.stack([(soil - soil_mean) / soil_std, (error - error_mean) / error_std], dim=1)

class SimpleThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)          # only uses current_error
        self.sigmoid = nn.Sigmoid()
        self.linear.bias.data.fill_(0.0)       # neutral start

    def forward(self, x):
        last_error = x[:, -1, 1:2]             # column 1 = current_error
        out = self.linear(last_error)
        return self.sigmoid(out)

def compute_forward_metrics(df, fire_time, moisture_target=MOISTURE_TARGET, window_hours=24):
    end_time = fire_time + timedelta(hours=window_hours)
    future_mask = (df['timestamp'] > fire_time) & (df['timestamp'] <= end_time)
    future_df = df[future_mask]
    if len(future_df) < 10:
        return 5.0, 0.0
    errors = future_df['soil_moisture_rolling_avg'] - moisture_target
    rmse = float(np.sqrt((errors ** 2).mean()))
    avg_error = float(errors.mean())
    return rmse, avg_error

def replay_diagnostic(epochs=15):
    print("=== SIMPLE LINEAR THRESHOLD DIAGNOSTIC (no LSTM recurrence) ===")

    df = pd.read_csv('watering_log.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    model_thresh = SimpleThreshold()
    optimizer = optim.Adam(model_thresh.parameters(), lr=0.01)

    for epoch in range(epochs):
        positive_count = 0
        total_loss = 0.0
        event_buffer = []
        feature_history = []

        for i, row in df.iterrows():
            now = row['timestamp']
            pulse_s = float(row.get('pulse_seconds', 0))
            sensor_errors = int(row['sensor_errors'])

            soil_avg = row['soil_moisture_rolling_avg']
            current_error = MOISTURE_TARGET - soil_avg
            feature_history.append([soil_avg, current_error])
            if len(feature_history) > SEQ_LEN:
                feature_history.pop(0)

            seq_tensor = None
            if len(feature_history) == SEQ_LEN:
                raw_seq = torch.tensor(feature_history, dtype=torch.float32)
                norm_seq = normalize_features(raw_seq)
                seq_tensor = norm_seq.unsqueeze(0)

            if pulse_s > 0.1 and sensor_errors == 0 and seq_tensor is not None:
                event = {'fire_time': now, 'feature_seq': seq_tensor.clone().detach()}
                event_buffer.append(event)
                positive_count += 1

            completed = [e for e in event_buffer if (now - e['fire_time']).total_seconds() >= 86400]
            for event in completed:
                rmse, avg_error = compute_forward_metrics(df, event['fire_time'])
                G = 10.0 / (1.0 + 4 * rmse)
                pos_weight = 3.0 * G

                loss = F.binary_cross_entropy(model_thresh(event['feature_seq']),
                                              torch.tensor([[1.0]]),
                                              weight=torch.tensor([[pos_weight]]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            event_buffer = [e for e in event_buffer if (now - e['fire_time']).total_seconds() < 86400]

        print(f"Epoch {epoch+1:2d} | Positives: {positive_count} | Avg loss: {total_loss/max(positive_count,1):.4f}")

    print("Diagnostic training complete.\n")

    # Quick test
    print("THRESHOLD TEST (Simple Linear Model)")
    for m in [60, 61, 62, 62.5, 63, 63.5, 64, 65, 66, 70]:
        soil_avg = m
        current_error = MOISTURE_TARGET - soil_avg
        seq = torch.tensor([[[soil_avg, current_error]] * SEQ_LEN], dtype=torch.float32)
        norm_seq = normalize_features(seq[0])
        prob = model_thresh(norm_seq.unsqueeze(0)).item()
        status = "BELOW" if m < MOISTURE_TARGET else "ABOVE"
        print(f"Soil={m:5.1f}% ({status}) → prob={prob:.4f}")

if __name__ == "__main__":
    replay_diagnostic(epochs=15)