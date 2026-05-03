#!/usr/bin/env python3
"""
Simple inference-only test for LSTM models.
Tests GainScheduler at various soil moisture levels around the new 63% target,
and Threshold LSTM at 62.5% (below) vs 63.5% (above) target.
Now uses the 13-input model (12 original features + current_error).
"""

import torch
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler

# ============================================================
# CONSTANT INPUTS (same for all tests)
# ============================================================
MOISTURE_TARGET  = 63.0          # ← matches your live system

AIR_AVG          = 19.0
GROUND_AVG       = 18.0
HUM_AVG          = 16.0
AIR_T            = 18.5
GROUND_T         = 17.0
HUMIDITY         = 20.0
WATERINGS_TODAY  = 4.0
MINUTES_UNTIL    = 0.0
AVG_ERROR        = 0.0           # avg_error_since_last (from live code)

def build_feature(soil_moisture: float) -> list:
    """
    Builds the exact 13-feature vector used by the new models.
    The 13th feature is the instantaneous current_error.
    """
    corrected     = soil_moisture / 1.012          # same temp correction heuristic as live
    moisture_avg  = soil_moisture * 0.997          # same rolling avg heuristic as live

    current_error = MOISTURE_TARGET - corrected    # ← NEW 13th feature (error heuristic)

    return [
        AIR_AVG, GROUND_AVG, HUM_AVG,
        AIR_T, GROUND_T, HUMIDITY,
        soil_moisture, corrected, moisture_avg,
        AVG_ERROR, WATERINGS_TODAY, MINUTES_UNTIL,
        current_error                                  # 13th input
    ]

def main():
    print("=" * 75)
    print("MODEL OUTCOME TEST — 13-INPUT MODEL (with current_error)")
    print("=" * 75)
    print(f"Target moisture = {MOISTURE_TARGET}%")
    print()

    # ============================================================
    # GAIN SCHEDULER TEST
    # ============================================================
    print("\n[1] GAIN MULTIPLIER TEST (around new 63% target)")
    print("-" * 60)

    model_gains = LSTM_GainScheduler(input_size=13)
    model_gains.load_state_dict(torch.load("lstm_gain_scheduler.pth",
                                           weights_only=True, map_location="cpu"))
    model_gains.eval()

    for m in [60.0, 62.0, 63.0, 64.0, 66.0]:
        feat = build_feature(m)
        # Create full 30-step sequence (matches live system + training)
        x = torch.tensor([feat] * 30, dtype=torch.float32).unsqueeze(0)  # (1, 30, 13)

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
    print("[2] THRESHOLD PROBABILITY TEST (around 63% target)")
    print("-" * 60)

    model_thresh = LSTM_Threshold(input_size=13)
    model_thresh.load_state_dict(torch.load("lstm_threshold.pth",
                                            weights_only=True, map_location="cpu"))
    model_thresh.eval()

    for m in [60, 61.5, 62.5, 63.0, 63.5, 64.5, 66, 70]:
        feat = build_feature(m)
        x = torch.tensor([feat] * 30, dtype=torch.float32).unsqueeze(0)  # (1, 30, 13)

        with torch.no_grad():
            prob = model_thresh(x).item()

        status = "ABOVE target" if m >= MOISTURE_TARGET else "BELOW target"
        print(f"Soil = {m:5.1f}%  ({status})  →  raw threshold prob = {prob:.4f}")

    print("\n" + "=" * 75)
    print("Test complete. (13-input model with current_error heuristic)")
    print("=" * 75)

if __name__ == "__main__":
    main()