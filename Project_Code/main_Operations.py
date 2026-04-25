import time
import csv
from datetime import datetime
import sys
from collections import deque
import torch
import math
from torch.optim import Adam

from Input_Output_Operations import Input_Output_Operations
from PID_Controller import PID
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler
from train_lstm import (
    online_update_threshold,
    online_update_gain_scheduler,
    INPUT_SIZE, SEQ_LEN,
    THRESHOLD_LR, GAIN_LR,
    RMSE_GOOD_THRESHOLD,
)

event_buffer    = []
feature_history = deque(maxlen=SEQ_LEN)
ground_t_avg    = 15.0

def setup():
    Input_Output = Input_Output_Operations(minimumUptime=0, maximumUptime=0)
    return Input_Output

def main(Input_Output):
    global event_buffer, feature_history, ground_t_avg

    MOISTURE_TARGET = 60.0
    LOWEST_THRESHOLD = 58.8

    MAX_WATERINGS_PER_DAY = 5
    MIN_TIME_BETWEEN_MIN  = 120

    MIN_PULSE_S = 0.0
    MAX_PULSE_S = 18.0

    base_kp = 0.009
    base_ki = 0.0015
    base_kd = 0.0

    pid = PID(kp=base_kp, ki=base_ki, kd=base_kd, setpoint=MOISTURE_TARGET)

    model_threshold = LSTM_Threshold(input_size=INPUT_SIZE)
    model_gains     = LSTM_GainScheduler(input_size=INPUT_SIZE)

    try:
        model_threshold.load_state_dict(torch.load("lstm_threshold.pth", weights_only=True, map_location='cpu'))
        model_gains.load_state_dict(torch.load("lstm_gain_scheduler.pth", weights_only=True, map_location='cpu'))
        print("Loaded saved LSTM models.")
    except FileNotFoundError:
        print("WARNING: LSTM models not found. Run initialize_lstm_models.py first.")

    model_threshold.eval()
    model_gains.eval()

    optimizer_threshold = Adam(model_threshold.parameters(), lr=THRESHOLD_LR)
    optimizer_gains     = Adam(model_gains.parameters(),     lr=GAIN_LR)

    waterings_today    = 0
    last_watering_time = time.time() - 99999
    last_midnight      = datetime.now().date()

    sum_air_t = sum_ground_t = sum_humidity = sum_error = 0.0
    sample_count = 0

    MAX_MOISTURE_QUEUE = 20
    moisture_window    = deque(maxlen=MAX_MOISTURE_QUEUE)

    MAX_GROUND_T_QUEUE = 20
    ground_t_window    = deque(maxlen=MAX_GROUND_T_QUEUE)

    MAX_ROLLING_RMSE_QUEUE = 144
    rolling_moisture       = deque(maxlen=MAX_ROLLING_RMSE_QUEUE)

    TEMP_CORRECTION_CENTER    = 15.0
    MOISTURE_PCT_PER_DEGREE_C = 0.011

    IDLE_TRAIN_CYCLES  = 1200
    idle_cycle_counter = IDLE_TRAIN_CYCLES - 1

    pulse_this_cycle = 0.0
    pump_on_this_cycle = 0

    log_file = open('watering_log.csv', 'a', newline='')
    writer   = csv.writer(log_file)
    if log_file.tell() == 0:
        writer.writerow([
            'timestamp', 'air_t_avg_since_last', 'ground_t_avg_since_last',
            'humidity_avg_since_last', 'air_t_current', 'ground_t_current',
            'humidity_current', 'soil_moisture_current', 'soil_moisture_corrected',
            'soil_moisture_rolling_avg', 'ground_t_rolling_avg',
            'avg_error_since_last', 'waterings_today', 'waterings_remaining',
            'minutes_until_next_allowed', 'pulse_seconds', 'pump_on', 'sensor_errors',
        ])

    changes_log_file = open('lstm_changes_log.csv', 'a', newline='')
    changes_writer   = csv.writer(changes_log_file)
    if changes_log_file.tell() == 0:
        changes_writer.writerow([
            'timestamp', 'soil_moisture_current', 'soil_moisture_corrected',
            'soil_moisture_rolling_avg', 'avg_error_since_last', 'waterings_today',
            'threshold_prob', 'effective_threshold',
            'kp_mult', 'ki_mult', 'kd_mult',
            'kp_current', 'ki_current', 'kd_current', 'pulse_seconds',
        ])

    gain_history_log_file = open('gain_history_log.csv', 'a', newline='')
    gain_history_writer   = csv.writer(gain_history_log_file)
    if gain_history_log_file.tell() == 0:
        gain_history_writer.writerow([
            'timestamp', 'fire_timestamp',
            'kp_mult', 'ki_mult', 'kd_mult',
            'kp_actual', 'ki_actual', 'kd_actual',
            'pulse_seconds', 'avg_error_at_fire',
            'waterings_today', 'threshold_prob',
        ])

    training_log_file = open('lstm_training_log.csv', 'a', newline='')
    training_writer   = csv.writer(training_log_file)
    if training_log_file.tell() == 0:
        training_writer.writerow([
            'timestamp', 'fire_timestamp',
            'rmse_24hr', 'mean_error_24hr',
            'threshold_target_prob', 'threshold_loss',
            'gain_target_mult', 'gain_loss',
        ])

    print("AI-Regulated Watering System starting...")

    while True:
        now_dt = datetime.now()
        now    = time.time()

        rmse = 0.0
        pos_weight = 1.0

        if now_dt.date() != last_midnight:
            waterings_today = 0
            last_midnight   = now_dt.date()

        Input         = Input_Output.readAll()
        ground_t      = Input[0]
        air_t         = Input[1]
        humidity      = Input[2]
        soil_moisture = Input[3]

        sensor_errors = 1 if (ground_t == 0 or air_t == 0 or humidity == 0 or soil_moisture == 0) else 0

        ground_t_window.append(ground_t)
        if len(ground_t_window) >= 9:
            sorted_temps = sorted(ground_t_window)
            ground_t_avg = sum(sorted_temps[4:-4]) / len(sorted_temps[4:-4]) # drop 2 upper and 2 lower outliers
        else:
            ground_t_avg = sum(ground_t_window) / len(ground_t_window)

        delta_t               = ground_t_avg - TEMP_CORRECTION_CENTER
        temp_correction_factor = 1.0 + MOISTURE_PCT_PER_DEGREE_C * delta_t
        soil_moisture_corrected = soil_moisture / temp_correction_factor

        moisture_window.append(soil_moisture_corrected)
        if len(moisture_window) >= 9:
            sorted_moisture = sorted(moisture_window)
            moisture_avg    = sum(sorted_moisture[4:-4]) / len(sorted_moisture[4:-4]) # drop 2 upper and 2 lower outliers
        else:
            moisture_avg = sum(moisture_window) / len(moisture_window)

        rolling_moisture.append(soil_moisture_corrected)

        sum_air_t    += air_t
        sum_ground_t += ground_t
        sum_humidity += humidity
        sum_error    += (soil_moisture_corrected - MOISTURE_TARGET)
        sample_count += 1

        air_avg   = sum_air_t    / sample_count
        ground_avg = sum_ground_t / sample_count
        hum_avg   = sum_humidity  / sample_count
        avg_error = sum_error     / sample_count

        time_since_last = now - last_watering_time
        minutes_until   = max(0.0, MIN_TIME_BETWEEN_MIN - (time_since_last / 60))
        can_water       = (waterings_today < MAX_WATERINGS_PER_DAY
                           and time_since_last / 60 >= MIN_TIME_BETWEEN_MIN)

        feature_list = [
            air_avg, ground_avg, hum_avg,
            air_t, ground_t, humidity,
            soil_moisture, soil_moisture_corrected, moisture_avg,
            avg_error, float(waterings_today), minutes_until,
        ]
        feature_history.append(feature_list)

        feature_vec = torch.tensor([feature_list], dtype=torch.float32).unsqueeze(0)

        seq_tensor = None
        if len(feature_history) == SEQ_LEN:
            seq_tensor = torch.tensor(list(feature_history), dtype=torch.float32).unsqueeze(0)

        inference_tensor = seq_tensor if seq_tensor is not None else feature_vec

        with torch.no_grad():
            threshold_prob = model_threshold(inference_tensor).item()

        confidence          = max(0.0, (threshold_prob - 0.5) * 2.0)
        effective_threshold = LOWEST_THRESHOLD + (MOISTURE_TARGET - LOWEST_THRESHOLD) * confidence

        allow_water = (
            can_water
            and moisture_avg < effective_threshold
            and moisture_avg < MOISTURE_TARGET
            and threshold_prob > 0.5
        )

        # === GAIN SCHEDULER + PID RESET (now matches context prompt exactly) ===
        # Always reset to base gains every cycle
        pid.kp = base_kp
        pid.ki = base_ki
        pid.kd = base_kd
        kp_mult = ki_mult = kd_mult = 1.0

        if allow_water:
            with torch.no_grad():
                raw = model_gains(inference_tensor).squeeze()   # raw ∈ [0, 2]
            kp_raw = raw[0].item()
            ki_raw = raw[1].item()
            kd_raw = raw[2].item()

            kp_mult = 1.0 + 0.5 * (kp_raw - 1.0)
            ki_mult = 1.0 + 0.5 * (ki_raw - 1.0)
            kd_mult = 1.0 + 0.5 * (kd_raw - 1.0)

            pid.kp = base_kp * kp_mult
            pid.ki = base_ki * ki_mult
            pid.kd = base_kd * kd_mult
        # else: gains remain at base (neutral) — exactly as documented

        if allow_water and len(moisture_window) == MAX_MOISTURE_QUEUE and seq_tensor is not None:
            duty    = pid.compute(moisture_avg)
            pulse_s = max(MIN_PULSE_S, min(MAX_PULSE_S,
                          MIN_PULSE_S + duty * (MAX_PULSE_S - MIN_PULSE_S)))

            if pulse_s > 0.1:
                print(f"{now_dt.strftime('%H:%M:%S')} | Soil: {soil_moisture:5.1f}% "
                      f"(corrected: {soil_moisture_corrected:5.1f}%, avg: {moisture_avg:5.1f}%) | "
                      f"Thresh: {threshold_prob:.3f} | Gate: {effective_threshold:.2f}% | "
                      f"Gains: Kp={pid.kp:.4f} Ki={pid.ki:.4f} Kd={pid.kd:.4f} | "
                      f"Watering #{waterings_today+1}/5 | ON for {pulse_s:.2f}s")

                Input_Output.activatePump(pulse_s)

                changes_writer.writerow([
                    now_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    round(soil_moisture, 2), round(soil_moisture_corrected, 2),
                    round(moisture_avg, 2),  round(avg_error, 2),
                    waterings_today,
                    round(threshold_prob, 3), round(effective_threshold, 2),
                    round(kp_mult, 3), round(ki_mult, 3), round(kd_mult, 3),
                    round(pid.kp, 4),  round(pid.ki, 4),  round(pid.kd, 4),
                    round(pulse_s, 2),
                ])
                changes_log_file.flush()

                fire_ts = now_dt.strftime('%Y-%m-%d %H:%M:%S')
                gain_history_writer.writerow([
                    fire_ts, fire_ts,
                    round(kp_mult, 3), round(ki_mult, 3), round(kd_mult, 3),
                    round(pid.kp, 4), round(pid.ki, 4), round(pid.kd, 4),
                    round(pulse_s, 2), round(avg_error, 4),
                    waterings_today, round(threshold_prob, 3),
                ])
                gain_history_log_file.flush()

                event_buffer.append({
                    'fire_time':         now,
                    'fire_timestamp':    fire_ts,
                    'feature_seq':       seq_tensor.clone().detach(),
                    'avg_error_at_fire': avg_error,
                    'kp_mult':           kp_mult,
                    'ki_mult':           ki_mult,
                    'kd_mult':           kd_mult,
                    'pulse_s':           pulse_s,
                    'moisture_readings': [],
                })

                last_watering_time = now
                waterings_today   += 1

                pulse_this_cycle = pulse_s
                pump_on_this_cycle = 1

                sum_air_t = sum_ground_t = sum_humidity = sum_error = 0.0
                sample_count     = 0
                idle_cycle_counter = 0
            else:
                last_watering_time = now

        for event in event_buffer:
            event['moisture_readings'].append(soil_moisture_corrected)

        completed = [e for e in event_buffer if (now - e['fire_time']) >= 86400]
        for event in completed:
            if len(event['moisture_readings']) == 0:
                continue

            readings   = event['moisture_readings']
            rmse       = math.sqrt(sum((r - MOISTURE_TARGET) ** 2 for r in readings) / len(readings))
            mean_error = sum(r - MOISTURE_TARGET for r in readings) / len(readings)

            G = 1.0 / (1.0 + rmse)

            if event.get('pulse_s', 0) > 0.1:
                target_prob = 1.0
                pos_weight  = 2 * G
            else:
                target_prob = 0.0
                pos_weight  = 0.1 * G

            t_loss = online_update_threshold(
                event['feature_seq'], target_prob,
                model_threshold, optimizer_threshold,
                pos_weight=pos_weight,
            )

            # Residual learning — correct damping inversion
            previous_mult = event.get('kp_mult', 1.0)          # damped final_mult
            target_mult   = previous_mult - math.tanh(mean_error / 5.0)
            target_mult   = max(0.5, min(1.5, target_mult))    # desired damped
            target_raw    = 1.0 + 2.0 * (target_mult - 1.0)    # ← INVERT to raw [0,2]

            g_info = online_update_gain_scheduler(
                event['feature_seq'], target_raw,   # now passes raw target
                model_gains, optimizer_gains,
            )

            training_writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                event['fire_timestamp'],
                round(rmse, 4),
                round(mean_error, 4),
                round(target_prob, 1),
                round(t_loss, 4),
                round(g_info['target_mult'], 3),
                round(g_info['loss'], 4),
            ])
            training_log_file.flush()

        event_buffer = [e for e in event_buffer if (now - e['fire_time']) < 86400]

        # Idle negative training (unified G)
        if not allow_water and seq_tensor is not None and sensor_errors == 0:
            idle_cycle_counter += 1
            if idle_cycle_counter >= IDLE_TRAIN_CYCLES and len(rolling_moisture) >= 30:
                idle_cycle_counter = 0
                readings = list(rolling_moisture)
                rmse = math.sqrt(sum((r - MOISTURE_TARGET) ** 2 for r in readings) / len(readings))
                G = 1.0 / (1.0 + 4 * rmse)

                # Capture actual loss from idle update
                t_loss = online_update_threshold(
                    seq_tensor, 0.0, model_threshold, optimizer_threshold, pos_weight=G
                )

                # === LOG IDLE NEGATIVE TRAINING (clean, meaningful) ===
                training_writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "IDLE",  # fire_timestamp
                    round(rmse, 4),
                    0.0,  # mean_error not applicable for idle
                    0.0,  # target_prob = 0.0
                    round(t_loss, 4),  # real threshold loss
                    0.0,  # gain not updated during idle
                    0.0,  # gain loss N/A
                ])
                training_log_file.flush()

        writer.writerow([
            now_dt.strftime('%Y-%m-%d %H:%M:%S'),
            round(air_avg, 2), round(ground_avg, 2), round(hum_avg, 2),
            round(air_t, 2), round(ground_t, 2), round(humidity, 2),
            round(soil_moisture, 2), round(soil_moisture_corrected, 2),
            round(moisture_avg, 2), round(ground_t_avg, 2),
            round(avg_error, 2),
            waterings_today,
            MAX_WATERINGS_PER_DAY - waterings_today,
            round(minutes_until, 2),
            round(pulse_this_cycle, 2), pump_on_this_cycle, sensor_errors,
        ])
        log_file.flush()

        print(f"{now_dt.strftime('%H:%M:%S')} | Soil: {soil_moisture:5.1f}% "
              f"(corr: {soil_moisture_corrected:5.1f}%, avg: {moisture_avg:5.1f}%) | "
              f"Thresh: {threshold_prob:.3f} | Gate: {effective_threshold:.2f}% | "
              f"Gains: Kp={pid.kp:.4f} Ki={pid.ki:.4f} Kd={pid.kd:.4f} | "
              f"Avg err: {avg_error:+5.1f}% | Waterings today: {waterings_today}/5")

        pulse_this_cycle = 0.0
        pump_on_this_cycle = 0

        time.sleep(10)


if __name__ == "__main__":
    Input_Output = setup()

    if Input_Output.detect_stemma():
        print("STEMMA sensors detected — starting controller.")
        # Input_Output.activatePump(1.0)
        time.sleep(2)
        main(Input_Output)
    else:
        print("ERROR: STEMMA sensors not detected on I2C. Exiting cleanly (no restart).")
        sys.exit(0)