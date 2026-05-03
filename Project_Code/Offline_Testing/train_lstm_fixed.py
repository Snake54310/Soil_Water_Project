import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from datetime import timedelta
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler

# ====================== FIXED LSTM CONFIG ======================
INPUT_SIZE = 2
SEQ_LEN = 30
MOISTURE_TARGET = 63.0
SENSITIVITY_SCALE = 5.0
# ============================================================

# Global normalization (computed once from the entire dataset)
df_global = pd.read_csv('watering_log.csv')
soil_mean = df_global['soil_moisture_rolling_avg'].mean()
soil_std = df_global['soil_moisture_rolling_avg'].std() + 1e-8
error_mean = (df_global['soil_moisture_rolling_avg'] - MOISTURE_TARGET).mean()
error_std = (df_global['soil_moisture_rolling_avg'] - MOISTURE_TARGET).std() + 1e-8
print(f"Normalization → soil_mean={soil_mean:.2f} std={soil_std:.2f} | error_mean={error_mean:.2f} std={error_std:.2f}")


def normalize_features(seq):
    """Normalize the 2-input sequence (soil_avg, current_error)"""
    soil = seq[:, 0]
    error = seq[:, 1]
    soil_norm = (soil - soil_mean) / soil_std
    error_norm = (error - error_mean) / error_std
    return torch.stack([soil_norm, error_norm], dim=1)


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


def online_update_threshold(feature_seq, target_prob, model, optimizer, pos_weight=10.0):
    model.train()
    optimizer.zero_grad()
    pred = model(feature_seq)
    target = torch.tensor([[target_prob]], dtype=torch.float32)
    weight = torch.tensor([[pos_weight]], dtype=torch.float32)
    loss = F.binary_cross_entropy(pred, target, weight=weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    model.eval()
    torch.save(model.state_dict(), "lstm_threshold.pth")
    return loss.item()


def online_update_gain_scheduler(feature_seq, value, model, optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(feature_seq).squeeze()
    target = torch.tensor([value, value, value], dtype=torch.float32)
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    model.eval()
    torch.save(model.state_dict(), "lstm_gain_scheduler.pth")
    return {'loss': loss.item(), 'target_mult': value}


def replay_lstm_fixed(epochs=15):
    print(f"=== FIXED LSTM TRAINING (LayerNorm, {epochs} epochs, NO IDLE) ===")

    df = pd.read_csv('watering_log.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Fresh models with LayerNorm and neutral bias
    model_threshold = LSTM_Threshold(input_size=INPUT_SIZE)
    model_gains = LSTM_GainScheduler(input_size=INPUT_SIZE)
    model_threshold.fc.bias.data.fill_(0.0)  # start at ~0.5 probability

    optimizer_threshold = optim.Adam([
        {'params': model_threshold.lstm.parameters(), 'lr': 0.0001},
        {'params': model_threshold.fc.parameters(), 'lr': 0.02},
    ])
    optimizer_gains = optim.Adam(model_gains.parameters(), lr=0.001)

    model_threshold.train()
    model_gains.train()

    for epoch in range(epochs):
        positive_count = 0
        total_pos_loss = 0.0
        event_buffer = []
        feature_history = []

        for i, row in df.iterrows():
            now = row['timestamp']
            pulse_s = float(row.get('pulse_seconds', 0))
            sensor_errors = int(row['sensor_errors'])

            soil_avg = row['soil_moisture_rolling_avg']
            current_error = MOISTURE_TARGET - soil_avg

            features = [soil_avg, current_error]
            feature_history.append(features)
            if len(feature_history) > SEQ_LEN:
                feature_history.pop(0)

            seq_tensor = None
            if len(feature_history) == SEQ_LEN:
                raw_seq = torch.tensor(feature_history, dtype=torch.float32)
                norm_seq = normalize_features(raw_seq)
                seq_tensor = norm_seq.unsqueeze(0)

            # Watering event
            if pulse_s > 0.1 and sensor_errors == 0 and seq_tensor is not None:
                with torch.no_grad():
                    raw = model_gains(seq_tensor).squeeze()
                kp_raw, ki_raw, kd_raw = raw.tolist()
                kp_mult = 1.0 + 0.5 * (kp_raw - 1.0)

                event = {
                    'fire_time': now,
                    'feature_seq': seq_tensor.clone().detach(),
                    'kp_mult': kp_mult,
                    'waterings_at_fire': int(row['waterings_today']),
                }
                event_buffer.append(event)
                positive_count += 1

            # 24h forward-looking positive updates
            completed = [e for e in event_buffer if (now - e['fire_time']).total_seconds() >= 86400]
            for event in completed:
                rmse, avg_error = compute_forward_metrics(df, event['fire_time'])
                G = 10.0 / (1.0 + 4 * rmse)
                pos_weight = 3.0 * G

                loss = online_update_threshold(
                    event['feature_seq'], 1.0, model_threshold, optimizer_threshold, pos_weight=pos_weight
                )
                total_pos_loss += loss

                # Gain update
                previous_mult = event.get('kp_mult', 1.0)
                target_mult = previous_mult - math.tanh(avg_error / SENSITIVITY_SCALE)
                target_mult = max(0.5, min(1.5, target_mult))
                target_raw = 1.0 + 2.0 * (target_mult - 1.0)
                target_raw = max(0.0, min(2.0, target_raw))
                online_update_gain_scheduler(event['feature_seq'], target_raw, model_gains, optimizer_gains)

            event_buffer = [e for e in event_buffer if (now - e['fire_time']).total_seconds() < 86400]

        avg_loss = total_pos_loss / max(positive_count, 1)
        print(f"Epoch {epoch + 1:2d} | Positives: {positive_count} | Avg loss: {avg_loss:.4f}")

    print("\nFixed LSTM training complete.")


if __name__ == "__main__":
    # Force fresh 2-input models
    from initialize_lstm_models import initialize_models

    initialize_models(input_size=2, save=True)

    replay_lstm_fixed(epochs=15)