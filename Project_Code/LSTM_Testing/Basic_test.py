#!/usr/bin/env python3
"""
Simple inference-only test for LSTM models.
Tests GainScheduler at 50.8% and 51.7% soil moisture,
and Threshold LSTM at 52.5% (above) and 51.5% (below) target.
"""

import torch
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler

# ============================================================
# CONSTANT INPUTS (same for all tests)
# ============================================================
AIR_AVG          = 19.0
GROUND_AVG       = 18.0
HUM_AVG          = 16.0
AIR_T            = 18.5
GROUND_T         = 17.0
HUMIDITY         = 20.0
WATERINGS_TODAY  = 1.0
MINUTES_UNTIL    = 0.0
AVG_ERROR        = 0.8

def build_feature(soil_moisture: float) -> list:
    corrected    = soil_moisture / 1.012
    moisture_avg = soil_moisture * 0.997
    return [
        AIR_AVG, GROUND_AVG, HUM_AVG,
        AIR_T, GROUND_T, HUMIDITY,
        soil_moisture, corrected, moisture_avg,
        AVG_ERROR, WATERINGS_TODAY, MINUTES_UNTIL
    ]

def main():
    print("=" * 65)
    print("MODEL OUTCOME TEST — Inference Only (Corrected)")
    print("=" * 65)

    # ============================================================
    # GAIN SCHEDULER TEST
    # ============================================================
    print("\n[1] GAIN MULTIPLIER TEST (50.8% vs 51.7%)")
    print("-" * 50)

    model_gains = LSTM_GainScheduler(input_size=12)
    model_gains.load_state_dict(torch.load("lstm_gain_scheduler.pth",
                                           weights_only=True, map_location="cpu"))
    model_gains.eval()

    for m in [55, 59]:
        feat = build_feature(m)
        # Create full 30-step sequence (matches live system + training)
        x = torch.tensor([feat] * 30, dtype=torch.float32).unsqueeze(0)  # (1, 30, 12)

        with torch.no_grad():
            raw = model_gains(x).squeeze()          # raw ∈ [0, 2]

        kp_raw, ki_raw, kd_raw = raw.tolist()
        kp_mult = 1.0 + 0.5 * (kp_raw - 1.0)
        ki_mult = 1.0 + 0.5 * (ki_raw - 1.0)
        kd_mult = 1.0 + 0.5 * (kd_raw - 1.0)

        print(f"Soil = {m:5.1f}%  →  raw [kp,ki,kd] = [{kp_raw:6.3f}, {ki_raw:6.3f}, {kd_raw:6.3f}]")
        print(f"                     damped mult     = [{kp_mult:6.3f}, {ki_mult:6.3f}, {kd_mult:6.3f}]\n")

    # ============================================================
    # THRESHOLD LSTM TEST
    # ============================================================
    print("[2] THRESHOLD PROBABILITY TEST (52.5% above vs 51.5% below 52%)")
    print("-" * 50)

    model_thresh = LSTM_Threshold(input_size=12)
    model_thresh.load_state_dict(torch.load("lstm_threshold.pth",
                                            weights_only=True, map_location="cpu"))
    model_thresh.eval()

    for m in [59, 61]:
        feat = build_feature(m)
        x = torch.tensor([feat] * 30, dtype=torch.float32).unsqueeze(0)  # (1, 30, 12)

        with torch.no_grad():
            prob = model_thresh(x).item()

        status = "ABOVE target (52%)" if m > 52.0 else "BELOW target (52%)"
        print(f"Soil = {m:5.1f}%  ({status})  →  raw threshold prob = {prob:.4f}")

    print("\n" + "=" * 65)
    print("Test complete.")
    print("=" * 65)

if __name__ == "__main__":
    main()