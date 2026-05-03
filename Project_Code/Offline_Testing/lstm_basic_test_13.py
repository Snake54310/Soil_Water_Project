#!/usr/bin/env python3
"""
Inference test for LSTM models — 13-input version.
Sequences are built from real CSV history, NOT [feat]*30.

[feat]*30 (constant input) causes the LSTM hidden state to converge
to a moisture-blind fixed point, hiding any slope the model has learned.
Real sequences preserve temporal dynamics and match what the model
was actually trained on.

Usage:
    python lstm_basic_test_13.py
    python lstm_basic_test_13.py --csv path/to/watering_log.csv
"""

import sys
import argparse
import torch
import pandas as pd
import numpy as np
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler
from train_lstm import normalize_features   # single source of truth for normalization

# ============================================================
# CONFIG
# ============================================================
MOISTURE_TARGET = 63.0
SEQ_LEN         = 30
CSV_PATH        = "watering_log.csv"

THRESHOLD_TEST_MOISTURES  = [60.0, 60.5, 61.0, 61.5, 62.0, 62.2, 62.5, 62.75, 63.0, 63.5, 64.5, 65.0, 65.5, 66.0, 70.0]
GAIN_TEST_MOISTURES       = [60.0, 60.5, 61.0, 61.5, 62.0, 62.2, 62.5, 62.75, 63.0, 63.2, 63.4, 64.0, 66.0]


# ============================================================
# SEQUENCE BUILDER
# ============================================================
def load_csv_sequences(csv_path: str, target_moistures: list[float], seq_len: int = SEQ_LEN):
    """
    For each target moisture value, find the closest real row in the CSV
    and return the preceding seq_len rows as the input sequence.

    This matches exactly what the live system and replay training use:
    a rolling window of real sensor readings ending at the moment of interest.

    Returns: dict {moisture_value: tensor of shape (1, seq_len, 13)}
    """
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Build full feature array for every row — normalized identically to training.
    # normalize_features() z-scores the first 12 features using the same global
    # means/stds computed in train_lstm.py; feat[12] (current_error) is left raw.
    all_features = []
    for _, row in df.iterrows():
        soil_avg = row['soil_moisture_rolling_avg']
        current_error = MOISTURE_TARGET - soil_avg
        feat = np.array([
            row['air_t_avg_since_last'],
            row['ground_t_avg_since_last'],
            row['humidity_avg_since_last'],
            row['air_t_current'],
            row['ground_t_current'],
            row['humidity_current'],
            row['soil_moisture_current'],
            row['soil_moisture_corrected'],
            soil_avg,
            row['avg_error_since_last'],
            float(row['waterings_today']),
            row['minutes_until_next_allowed'],
            current_error,
        ], dtype=np.float32)
        all_features.append(normalize_features(feat))

    all_features = np.array(all_features, dtype=np.float32)   # (N, 13) — normalized
    # soil_corrected_col must come from the raw CSV, not all_features[:,8], because
    # all_features is normalized — index 8 holds a z-score, not a moisture percentage.
    # Using a normalized value for argmin against targets like 63.0 would always resolve
    # to the same row (whichever z-score is least far from 63).
    soil_corrected_col = df['soil_moisture_rolling_avg'].values.astype(np.float32)

    sequences = {}
    for target_m in target_moistures:
        # Find the row whose soil_corrected is closest to target_m
        idx = int(np.argmin(np.abs(soil_corrected_col - target_m)))

        # Need at least seq_len rows of history before it
        if idx < seq_len:
            idx = seq_len   # clamp to first valid position

        window = all_features[idx - seq_len : idx]             # (seq_len, 13) — normalized
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 13)

        actual_end_moisture = soil_corrected_col[idx - 1]
        sequences[target_m] = (tensor, actual_end_moisture, idx)

    return sequences


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH, help="Path to watering_log.csv")
    args = parser.parse_args()

    print("=" * 75)
    print("MODEL OUTCOME TEST — 13-INPUT MODEL (real CSV sequences)")
    print("=" * 75)
    print(f"Target moisture = {MOISTURE_TARGET}%")
    print(f"CSV             = {args.csv}")
    print(f"Sequence length = {SEQ_LEN} steps")
    print()
    print("NOTE: Each test point uses a real 30-step rolling window from the CSV,")
    print("      ending at the row closest to the requested moisture level.")
    print("      This matches training exactly — no [feat]*30 constant sequences.")
    print()

    # ----------------------------------------------------------------
    # Pre-load sequences for both tests
    # ----------------------------------------------------------------
    all_moistures = sorted(set(THRESHOLD_TEST_MOISTURES + GAIN_TEST_MOISTURES))
    try:
        sequences = load_csv_sequences(args.csv, all_moistures)
    except FileNotFoundError:
        print(f"ERROR: Could not find CSV at '{args.csv}'")
        print("       Pass the correct path with --csv path/to/watering_log.csv")
        sys.exit(1)

    # ----------------------------------------------------------------
    # [1] GAIN SCHEDULER TEST
    # ----------------------------------------------------------------
    print("[1] GAIN MULTIPLIER TEST (around 63% target)")
    print("-" * 60)

    model_gains = LSTM_GainScheduler(input_size=13, lstm_hidden=8)
    model_gains.load_state_dict(
        torch.load("lstm_gain_scheduler.pth", weights_only=True, map_location="cpu")
    )
    model_gains.eval()

    for m in GAIN_TEST_MOISTURES:
        x, actual_m, row_idx = sequences[m]

        with torch.no_grad():
            raw = model_gains(x).squeeze()

        kp_raw, ki_raw, kd_raw = raw.tolist()
        kp_mult = 1.0 + 0.5 * (kp_raw - 1.0)
        ki_mult = 1.0 + 0.5 * (ki_raw - 1.0)
        kd_mult = 1.0 + 0.5 * (kd_raw - 1.0)

        print(f"Soil = {m:5.1f}%  (actual end = {actual_m:.2f}%, row {row_idx})")
        print(f"  raw  [kp,ki,kd] = [{kp_raw:6.3f}, {ki_raw:6.3f}, {kd_raw:6.3f}]")
        print(f"  mult [kp,ki,kd] = [{kp_mult:6.3f}, {ki_mult:6.3f}, {kd_mult:6.3f}]")
        print()

    # ----------------------------------------------------------------
    # [2] THRESHOLD TEST
    # ----------------------------------------------------------------
    print("[2] THRESHOLD PROBABILITY TEST (around 63% target)")
    print("-" * 60)

    model_thresh = LSTM_Threshold(input_size=13, lstm_hidden=8)
    model_thresh.load_state_dict(
        torch.load("lstm_threshold.pth", weights_only=True, map_location="cpu")
    )
    model_thresh.eval()

    for m in THRESHOLD_TEST_MOISTURES:
        x, actual_m, row_idx = sequences[m]

        with torch.no_grad():
            prob = model_thresh(x).item()

        status = "ABOVE" if actual_m >= MOISTURE_TARGET else "BELOW"
        print(f"Soil = {m:5.1f}%  →  actual = {actual_m:.2f}% ({status})  "
              f"row {row_idx:>5}  →  prob = {prob:.4f}")

    print()
    print("=" * 75)
    print("Test complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()