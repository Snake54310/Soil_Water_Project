# main.py  (or whatever you named your main script)
import time
import csv
from datetime import datetime

from Soil_Water_Project.Project_Code.Input_Output_Operations import Input_Output_Operations
from PID_Controller import PID   # ←←← new import

def setup():
    Input_Output = Input_Output_Operations(minimumUptime=0, maximumUptime=0)  # your original values
    return Input_Output

def main(Input_Output):
    MOISTURE_TARGET = 50.0

    # Scheduling limits (pure constraints — no intelligence)
    MAX_WATERINGS_PER_DAY = 5
    MIN_TIME_BETWEEN_MIN = 120          # 2 hours safety

    # Pulse limits (PID decides length inside this range)
    MIN_PULSE_S = 3.0
    MAX_PULSE_S = 30.0

    pid = PID(kp=0.8, ki=0.05, kd=0.1, setpoint=MOISTURE_TARGET)

    # State tracking
    waterings_today = 0
    last_watering_time = time.time() - 99999   # force first possible event
    last_midnight = datetime.now().date()

    # Accumulators for averages since LAST watering
    sum_air_t = 0.0
    sum_ground_t = 0.0
    sum_humidity = 0.0
    sum_error = 0.0
    sample_count = 0

    # Open log file (appends if it exists)
    log_file = open('watering_log.csv', 'a', newline='')
    writer = csv.writer(log_file)
    # Header
    if log_file.tell() == 0:
        writer.writerow(['timestamp', 'air_t_avg_since_last', 'ground_t_avg_since_last',
                         'humidity_avg_since_last', 'air_t_current', 'ground_t_current',
                         'humidity_current', 'soil_moisture_current', 'avg_error_since_last',
                         'waterings_today', 'waterings_remaining', 'minutes_until_next_allowed',
                         'pulse_seconds', 'pump_on'])

    print("AI-Regulated Watering System (5/day scheduled PID + LSTM-ready logging) starting...")

    while True:
        now_dt = datetime.now()
        now = time.time()

        # Reset daily counter at midnight
        if now_dt.date() != last_midnight:
            waterings_today = 0
            last_midnight = now_dt.date()

        # Read sensors
        Input = Input_Output.readAll()          # [ground_temp, air_temp, humidity, soil_moisture]
        ground_t = Input[0]
        air_t = Input[1]
        humidity = Input[2]
        soil_moisture = Input[3]

        # Accumulate since last watering
        sum_air_t += air_t
        sum_ground_t += ground_t
        sum_humidity += humidity
        sum_error += (soil_moisture - MOISTURE_TARGET)
        sample_count += 1

        # Averages since last watering
        if sample_count > 0:
            air_avg = sum_air_t / sample_count
            ground_avg = sum_ground_t / sample_count
            hum_avg = sum_humidity / sample_count
            avg_error = sum_error / sample_count
        else:
            air_avg = ground_avg = hum_avg = avg_error = 0.0

        # === Decide whether to schedule a watering event ===
        time_since_last = now - last_watering_time
        can_water = (waterings_today < MAX_WATERINGS_PER_DAY and
                     time_since_last / 60 >= MIN_TIME_BETWEEN_MIN)

        if can_water:
            duty = pid.compute(soil_moisture)   # pure PID — no extra rules
            pulse_s = MIN_PULSE_S + duty * (MAX_PULSE_S - MIN_PULSE_S)
            pulse_s = max(MIN_PULSE_S, min(MAX_PULSE_S, pulse_s))

            if pulse_s > MIN_PULSE_S + 0.5:
                print(f"{now_dt.strftime('%H:%M:%S')} | Soil: {soil_moisture:5.1f}% | "
                      f"Watering #{waterings_today+1}/5 | ON for {pulse_s:.1f}s")

                # Use your verified pump method (blocks only for the pulse duration)
                Input_Output.activatePump(pulse_s)

                last_watering_time = now
                waterings_today += 1

                # Log the event
                writer.writerow([now_dt.strftime('%Y-%m-%d %H:%M:%S'),
                                  round(air_avg,2), round(ground_avg,2), round(hum_avg,2),
                                  round(air_t,2), round(ground_t,2), round(humidity,2),
                                  round(soil_moisture,2), round(avg_error,2),
                                  waterings_today,
                                  MAX_WATERINGS_PER_DAY - waterings_today,
                                  max(0, MIN_TIME_BETWEEN_MIN - (time_since_last / 60)),
                                  round(pulse_s,1), 1])

                # Reset accumulators
                sum_air_t = sum_ground_t = sum_humidity = sum_error = 0.0
                sample_count = 0
            else:
                last_watering_time = now

        # Log every 10 s (full dataset for LSTM + error-over-time tracking)
        writer.writerow([now_dt.strftime('%Y-%m-%d %H:%M:%S'),
                          round(air_avg,2), round(ground_avg,2), round(hum_avg,2),
                          round(air_t,2), round(ground_t,2), round(humidity,2),
                          round(soil_moisture,2), round(avg_error,2),
                          waterings_today,
                          MAX_WATERINGS_PER_DAY - waterings_today,
                          max(0, MIN_TIME_BETWEEN_MIN - (time_since_last / 60)),
                          0.0, 0])

        log_file.flush()

        # Live status
        print(f"{now_dt.strftime('%H:%M:%S')} | Soil: {soil_moisture:5.1f}% | "
              f"Avg err since last: {avg_error:+5.1f}% | Waterings today: {waterings_today}/5")

        time.sleep(10)

# ====================== START ======================
if __name__ == "__main__":
    Input_Output = setup()
    time.sleep(2)
    main(Input_Output)