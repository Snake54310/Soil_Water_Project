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

MOISTURE_TARGET    = 63.0
SEQ_LEN            = 30
SENSITIVITY_SCALE  = 1.5
INPUT_SIZE         = 13
THRESHOLD_LR       = 0.01
GAIN_LR            = 0.005
LSTM_LR            = 1e-4    # LSTM layers train slower to stay stable on sparse data

# Global normalization for all 13 features (must match test script)
df_global = pd.read_csv('watering_log.csv')
feature_cols = [
    'air_t_avg_since_last', 'ground_t_avg_since_last', 'humidity_avg_since_last',
    'air_t_current', 'ground_t_current', 'humidity_current',
    'soil_moisture_current', 'soil_moisture_corrected', 'soil_moisture_rolling_avg',
    'avg_error_since_last', 'waterings_today', 'minutes_until_next_allowed'
]
means = df_global[feature_cols].mean().values
stds  = df_global[feature_cols].std().values + 1e-8

def normalize_features(feat_array: np.ndarray) -> np.ndarray:
    """
    Normalize a (13,) feature array in-place using the global means/stds.

    The first 12 features are z-scored; feat[12] (current_error = target -
    soil_moisture_rolling_avg) is intentionally left on its natural scale so
    the model receives a signed, interpretable error signal.

    Import this from the test / inference scripts so training and inference
    always use identical normalization:

        from train_lstm import normalize_features
    """
    out = feat_array.copy()
    out[:-1] = (out[:-1] - means) / stds
    return out

def build_and_normalize_features(row) -> torch.Tensor:
    """Build a normalized 13-D feature tensor from a single CSV row."""
    soil_avg = row['soil_moisture_rolling_avg']
    current_error = MOISTURE_TARGET - soil_avg
    feat = np.array([
        row['air_t_avg_since_last'], row['ground_t_avg_since_last'], row['humidity_avg_since_last'],
        row['air_t_current'], row['ground_t_current'], row['humidity_current'],
        row['soil_moisture_current'], row['soil_moisture_corrected'], soil_avg,
        row['avg_error_since_last'], float(row['waterings_today']),
        row['minutes_until_next_allowed'], current_error,
    ], dtype=np.float32)
    return torch.tensor(normalize_features(feat), dtype=torch.float32)

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
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()
    torch.save(model.state_dict(), "lstm_threshold.pth")
    return loss.item()

def compute_gain_target(prev_mult, mean_error):
    """
    Compute the raw [0, 2] gain target for a single gain channel given its
    previous damped multiplier and the 24h mean moisture error.

    Shared by replay_linear_13 and main_Operations.py so both use identical
    gain target logic. Import via:

        from train_lstm import compute_gain_target
    """
    target_mult = prev_mult - math.tanh(mean_error / SENSITIVITY_SCALE)
    target_mult = max(0.5, min(1.5, target_mult))
    target_raw  = 1.0 + 2.0 * (target_mult - 1.0)
    return max(0.0, min(2.0, target_raw))

def online_update_gain_scheduler(feature_seq, value, model, optimizer):
    """value is a list [kp_raw, ki_raw, kd_raw] of independent raw targets in [0, 2]."""
    model.train()
    optimizer.zero_grad()
    pred = model(feature_seq).squeeze()
    target = torch.tensor(value, dtype=torch.float32)
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()
    torch.save(model.state_dict(), "lstm_gain_scheduler.pth")
    return {'loss': loss.item()}

def replay_linear_13(epochs=3):
    print("=== LINEAR 13-INPUT TRAINING (corrected - respects initialization) ===")

    df = pd.read_csv('watering_log.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Load initialized models (this is the key fix for consistency across initialize_lstm_models.py)
    model_threshold = LSTM_Threshold(input_size=INPUT_SIZE)
    model_gains = LSTM_GainScheduler(input_size=INPUT_SIZE)
    try:
        model_threshold.load_state_dict(torch.load("lstm_threshold.pth", weights_only=True, map_location='cpu'))
        model_gains.load_state_dict(torch.load("lstm_gain_scheduler.pth", weights_only=True, map_location='cpu'))
        print("✅ Loaded initialized models successfully.")
    except FileNotFoundError:
        print("No existing models found – using fresh initialization.")

    optimizer_threshold = optim.Adam([
        {'params': model_threshold.lstm.parameters(),   'lr': LSTM_LR},
        {'params': model_threshold.linear.parameters(), 'lr': THRESHOLD_LR},
    ])
    optimizer_gains = optim.Adam([
        {'params': model_gains.lstm.parameters(),   'lr': LSTM_LR},
        {'params': model_gains.linear.parameters(), 'lr': GAIN_LR},
    ])

    for epoch in range(epochs):
        positive_count = 0
        non_watering_count = 0
        event_buffer = []
        feature_history = []

        for i, row in df.iterrows():
            now = row['timestamp']
            pulse_s = float(row.get('pulse_seconds', 0))
            sensor_errors = int(row['sensor_errors'])

            feat = build_and_normalize_features(row)
            feature_history.append(feat)
            if len(feature_history) > SEQ_LEN:
                feature_history.pop(0)

            seq_tensor = None
            if len(feature_history) == SEQ_LEN:
                seq_tensor = torch.stack(feature_history, dim=0).unsqueeze(0)

            if pulse_s > 0.1 and sensor_errors == 0 and seq_tensor is not None:
                # Positive watering event
                with torch.no_grad():
                    raw = model_gains(seq_tensor).squeeze()
                kp_mult = 1.0 + 0.5 * (raw[0].item() - 1.0)
                ki_mult = 1.0 + 0.5 * (raw[1].item() - 1.0)
                kd_mult = 1.0 + 0.5 * (raw[2].item() - 1.0)

                event = {
                    'fire_time': now,
                    'feature_seq': seq_tensor.clone().detach(),
                    'kp_mult': kp_mult,
                    'ki_mult': ki_mult,
                    'kd_mult': kd_mult,
                    'waterings_at_fire': int(row['waterings_today']),
                    'is_negative': False
                }
                event_buffer.append(event)
                positive_count += 1

            else:
                # Negative sampling: 1 in 1200 (includes the 100th)
                non_watering_count += 1
                if non_watering_count % 1200 == 100 and seq_tensor is not None:
                    event = {
                        'fire_time': now,
                        'feature_seq': seq_tensor.clone().detach(),
                        'is_negative': True
                    }
                    event_buffer.append(event)

            completed = [e for e in event_buffer if (now - e['fire_time']).total_seconds() >= 86400]
            for event in completed:
                rmse, avg_error = compute_forward_metrics(df, event['fire_time'])
                G = 10.0 / (1.0 + 4 * rmse)

                if event.get('is_negative', False):
                    online_update_threshold(event['feature_seq'], 0.0, model_threshold, optimizer_threshold, pos_weight=G * 0.1)
                else:
                    waterings = event.get('waterings_at_fire', 1)
                    if waterings == 1:
                        factor = 1.0
                    elif waterings == 2:
                        factor = 0.5
                    elif waterings == 3:
                        factor = 0.2
                    else:  # 4th and 5th
                        factor = 0.1
                    online_update_threshold(event['feature_seq'], 1.0, model_threshold, optimizer_threshold, pos_weight=G * factor)

                    target_raw = [
                        compute_gain_target(event.get('kp_mult', 1.0), avg_error),
                        compute_gain_target(event.get('ki_mult', 1.0), avg_error),
                        compute_gain_target(event.get('kd_mult', 1.0), avg_error),
                    ]
                    online_update_gain_scheduler(event['feature_seq'], target_raw, model_gains, optimizer_gains)

            event_buffer = [e for e in event_buffer if (now - e['fire_time']).total_seconds() < 86400]

        print(f"Epoch {epoch+1:2d} | Positives: {positive_count} | Negatives sampled: {non_watering_count}")

    print("Training complete.")

if __name__ == "__main__":
    from initialize_lstm_models import initialize_models
    initialize_models(input_size=13, save=True)
    replay_linear_13(epochs=1)