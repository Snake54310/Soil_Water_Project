#!/usr/bin/env python3
"""
Simplified 2-input test (soil_rolling_avg + current_error only).
"""
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler

MOISTURE_TARGET = 63.0
SEQ_LEN = 30

THRESHOLD_TEST_MOISTURES = [60.0, 61.0, 62.0, 62.5, 63.0, 63.5, 64.0, 65.0, 66.0, 70.0]
GAIN_TEST_MOISTURES = THRESHOLD_TEST_MOISTURES


def load_csv_sequences(csv_path, target_moistures):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    all_features = []
    for _, row in df.iterrows():
        soil_avg = row['soil_moisture_rolling_avg']
        current_error = MOISTURE_TARGET - soil_avg
        feat = [soil_avg, current_error]          # ONLY 2 inputs
        all_features.append(feat)

    all_features = np.array(all_features, dtype=np.float32)
    soil_col = all_features[:, 0]

    sequences = {}
    for target_m in target_moistures:
        idx = int(np.argmin(np.abs(soil_col - target_m)))
        if idx < SEQ_LEN:
            idx = SEQ_LEN
        window = all_features[idx - SEQ_LEN:idx]
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        actual = soil_col[idx - 1]
        sequences[target_m] = (tensor, actual, idx)

    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="watering_log.csv")
    args = parser.parse_args()

    print("=" * 75)
    print("SIMPLIFIED 2-INPUT MODEL TEST")
    print("=" * 75)

    sequences = load_csv_sequences(args.csv, set(THRESHOLD_TEST_MOISTURES + GAIN_TEST_MOISTURES))

    # Gain test
    print("\n[1] GAIN MULTIPLIER TEST")
    model_gains = LSTM_GainScheduler(input_size=2)
    model_gains.load_state_dict(torch.load("lstm_gain_scheduler.pth", weights_only=True, map_location="cpu"))
    model_gains.eval()

    for m in GAIN_TEST_MOISTURES:
        x, actual, idx = sequences[m]
        with torch.no_grad():
            raw = model_gains(x).squeeze()
        kp_raw, ki_raw, kd_raw = raw.tolist()
        kp_mult = 1.0 + 0.5 * (kp_raw - 1.0)
        print(f"Soil={m:5.1f}%  mult=[{kp_mult:5.3f}, {1.0+0.5*(ki_raw-1.0):5.3f}, {1.0+0.5*(kd_raw-1.0):5.3f}]")

    # Threshold test
    print("\n[2] THRESHOLD PROBABILITY TEST")
    model_thresh = LSTM_Threshold(input_size=2)
    model_thresh.load_state_dict(torch.load("lstm_threshold.pth", weights_only=True, map_location="cpu"))
    model_thresh.eval()

    for m in THRESHOLD_TEST_MOISTURES:
        x, actual, idx = sequences[m]
        with torch.no_grad():
            prob = model_thresh(x).item()
        status = "ABOVE" if actual >= MOISTURE_TARGET else "BELOW"
        print(f"Soil={m:5.1f}% → actual={actual:.2f}% ({status}) → prob={prob:.4f}")

    print("\nTest complete.")


if __name__ == "__main__":
    main()