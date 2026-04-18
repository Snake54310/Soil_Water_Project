import time
import csv
from datetime import datetime
import sys
from collections import deque

from Input_Output_Operations import Input_Output_Operations
from PID_Controller import PID

def setup():
    Input_Output = Input_Output_Operations(minimumUptime=0, maximumUptime=0)
    return Input_Output

def main(Input_Output):
    MOISTURE_TARGET = 50.0
    PUMP_THRESHOLD = 3.00

    # Scheduling limits (pure constraints)
    MAX_WATERINGS_PER_DAY = 5
    MIN_TIME_BETWEEN_MIN = 120          # 2 hours safety

    # Pulse limits - your final request
    MIN_PULSE_S = 0.0
    MAX_PULSE_S = 18.0     # strict failsafe only

    # Final PID gains (gentle, matched to 0.17 cu ft pot)
    pid = PID(kp=0.009, ki=0.0015, kd=0.0, setpoint=MOISTURE_TARGET)

    # State tracking
    waterings_today = 0
    last_watering_time = time.time() - 99999
    last_midnight = datetime.now().date()

    # Accumulators for averages since LAST watering
    sum_air_t = sum_ground_t = sum_humidity = sum_error = 0.0
    sample_count = 0

    # Rolling average for moisture (last 10 readings)
    MAX_MOISTURE_QUEUE = 10
    moisture_window = deque(maxlen=MAX_MOISTURE_QUEUE)

    # Rolling average for ground temperature (last 10 readings)
    # Used exclusively for temperature-based moisture correction
    MAX_GROUND_T_QUEUE = 10
    ground_t_window = deque(maxlen=MAX_GROUND_T_QUEUE)

    # Temperature correction parameters
    TEMP_CORRECTION_CENTER = 15.0       # degrees C at which no correction is applied
    MOISTURE_PCT_PER_DEGREE_C = 0.016   # fractional moisture increase per degree above center
                                        # e.g. 0.015 means +1.5% capacitance per +1 degree C
                                        # correction formula: corrected = raw / (1 + rate * delta_t)

    # Log file setup
    log_file = open('watering_log.csv', 'a', newline='')
    writer = csv.writer(log_file)
    if log_file.tell() == 0:
        writer.writerow(['timestamp', 'air_t_avg_since_last', 'ground_t_avg_since_last',
                         'humidity_avg_since_last', 'air_t_current', 'ground_t_current',
                         'humidity_current', 'soil_moisture_current', 'soil_moisture_corrected',
                         'soil_moisture_rolling_avg', 'ground_t_rolling_avg',
                         'avg_error_since_last', 'waterings_today', 'waterings_remaining',
                         'minutes_until_next_allowed', 'pulse_seconds', 'pump_on', 'sensor_errors'])

    print("AI-Regulated Watering System starting...")

    while True:
        now_dt = datetime.now()
        now = time.time()

        if now_dt.date() != last_midnight:
            waterings_today = 0
            last_midnight = now_dt.date()

        Input = Input_Output.readAll()
        ground_t = Input[0]
        air_t = Input[1]
        humidity = Input[2]
        soil_moisture = Input[3]

        # ====================== GROUND TEMP ROLLING AVERAGE (trimmed) ======================
        ground_t_window.append(ground_t)

        if len(ground_t_window) >= 3:
            # Sort and strip the single highest and single lowest outlier,
            # then average the remaining values.
            sorted_temps = sorted(ground_t_window)
            trimmed_temps = sorted_temps[1:-1]
            ground_t_avg = sum(trimmed_temps) / len(trimmed_temps)
        else:
            ground_t_avg = sum(ground_t_window) / len(ground_t_window)
        # ====================================================================================

        # ====================== TEMPERATURE-CORRECTED MOISTURE ======================
        delta_t = ground_t_avg - TEMP_CORRECTION_CENTER
        temp_correction_factor = 1.0 + MOISTURE_PCT_PER_DEGREE_C * delta_t
        soil_moisture_corrected = soil_moisture / temp_correction_factor
        # ============================================================================

        # Update rolling moisture average using temperature-corrected value
        moisture_window.append(soil_moisture_corrected)

        if len(moisture_window) >= 3:
            sorted_moisture = sorted(moisture_window)
            trimmed_moisture = sorted_moisture[1:-1]
            moisture_avg = sum(trimmed_moisture) / len(trimmed_moisture)
        else:
            moisture_avg = sum(moisture_window) / len(moisture_window)

        # Accumulate (log raw ground_t, corrected moisture error)
        sum_air_t += air_t
        sum_ground_t += ground_t
        sum_humidity += humidity
        sum_error += (soil_moisture_corrected - MOISTURE_TARGET)
        sample_count += 1

        if sample_count > 0:
            air_avg = sum_air_t / sample_count
            ground_avg = sum_ground_t / sample_count
            hum_avg = sum_humidity / sample_count
            avg_error = sum_error / sample_count
        else:
            air_avg = ground_avg = hum_avg = avg_error = 0.0

        time_since_last = now - last_watering_time
        can_water = (waterings_today < MAX_WATERINGS_PER_DAY and
                     time_since_last / 60 >= MIN_TIME_BETWEEN_MIN)

        sensor_errors = 1 if (ground_t == 0 or air_t == 0 or humidity == 0 or soil_moisture == 0) else 0

        if can_water and moisture_avg < (MOISTURE_TARGET - PUMP_THRESHOLD) and len(moisture_window) == MAX_MOISTURE_QUEUE:
            duty = pid.compute(moisture_avg)            # PID on corrected rolling avg
            pulse_s = MIN_PULSE_S + duty * (MAX_PULSE_S - MIN_PULSE_S)
            pulse_s = max(MIN_PULSE_S, min(MAX_PULSE_S, pulse_s))

            if pulse_s > 0.1:      # avoid zero-duration calls
                print(f"{now_dt.strftime('%H:%M:%S')} | Soil: {soil_moisture:5.1f}% "
                      f"(corrected: {soil_moisture_corrected:5.1f}%, avg: {moisture_avg:5.1f}%) | "
                      f"GndT avg: {ground_t_avg:5.2f}C | "
                      f"Watering #{waterings_today+1}/5 | ON for {pulse_s:.2f}s")

                Input_Output.activatePump(pulse_s)

                last_watering_time = now
                waterings_today += 1

                writer.writerow([now_dt.strftime('%Y-%m-%d %H:%M:%S'),
                                  round(air_avg,2), round(ground_avg,2), round(hum_avg,2),
                                  round(air_t,2), round(ground_t,2), round(humidity,2),
                                  round(soil_moisture,2), round(soil_moisture_corrected,2),
                                  round(moisture_avg,2), round(ground_t_avg,2),
                                  round(avg_error,2),
                                  waterings_today,
                                  MAX_WATERINGS_PER_DAY - waterings_today,
                                  max(0, MIN_TIME_BETWEEN_MIN - (time_since_last / 60)),
                                  round(pulse_s,2), 1, sensor_errors])

                sum_air_t = sum_ground_t = sum_humidity = sum_error = 0.0
                sample_count = 0
            else:
                last_watering_time = now

        # Log every 10 s
        writer.writerow([now_dt.strftime('%Y-%m-%d %H:%M:%S'),
                          round(air_avg,2), round(ground_avg,2), round(hum_avg,2),
                          round(air_t,2), round(ground_t,2), round(humidity,2),
                          round(soil_moisture,2), round(soil_moisture_corrected,2),
                          round(moisture_avg,2), round(ground_t_avg,2),
                          round(avg_error,2),
                          waterings_today,
                          MAX_WATERINGS_PER_DAY - waterings_today,
                          max(0, MIN_TIME_BETWEEN_MIN - (time_since_last / 60)),
                          0.0, 0, sensor_errors])

        log_file.flush()

        print(f"{now_dt.strftime('%H:%M:%S')} | Soil: {soil_moisture:5.1f}% "
              f"(corrected: {soil_moisture_corrected:5.1f}%, avg: {moisture_avg:5.1f}%) | "
              f"GndT avg: {ground_t_avg:5.2f}C | "
              f"Avg err: {avg_error:+5.1f}% | Waterings today: {waterings_today}/5")

        time.sleep(10)

# ====================== STARTUP LOGIC ======================
if __name__ == "__main__":
    Input_Output = setup()

    if Input_Output.detect_stemma():
        print("STEMMA sensors detected — starting controller.")
        Input_Output.activatePump(1.0)
        time.sleep(2)
        main(Input_Output)
    else:
        print("ERROR: STEMMA sensors not detected on I2C. Exiting cleanly (no restart).")
        sys.exit(0)