import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)  # for reproducible test data

num_rows = 2000 # ~1.5 days at 10-second intervals

# Realistic base values
base_soil = 52.0
base_air_t = 22.5
base_ground_t = 15.0
base_humidity = 55.0

timestamps = [datetime(2026, 4, 18, 0, 0) + timedelta(seconds=10 * i) for i in range(num_rows)]

data = {
    'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
    'air_t_avg_since_last': np.random.normal(base_air_t, 1.5, num_rows),
    'ground_t_avg_since_last': np.random.normal(base_ground_t, 0.8, num_rows),
    'humidity_avg_since_last': np.random.normal(base_humidity, 5.0, num_rows),
    'air_t_current': np.random.normal(base_air_t, 2.0, num_rows),
    'ground_t_current': np.random.normal(base_ground_t, 1.0, num_rows),
    'humidity_current': np.random.normal(base_humidity, 6.0, num_rows),
    'soil_moisture_current': np.clip(np.random.normal(base_soil, 3.0, num_rows), 35, 65),
    'soil_moisture_corrected': np.clip(np.random.normal(base_soil, 2.5, num_rows), 35, 65),
    'soil_moisture_rolling_avg': np.clip(np.random.normal(base_soil, 1.8, num_rows), 40, 60),
    'ground_t_rolling_avg': np.random.normal(base_ground_t, 0.7, num_rows),
    'avg_error_since_last': np.random.normal(0.0, 1.2, num_rows),
    'waterings_today': np.random.randint(0, 4, num_rows),
    'waterings_remaining': np.random.randint(1, 6, num_rows),
    'minutes_until_next_allowed': np.random.uniform(0, 180, num_rows),
    'pulse_seconds': np.zeros(num_rows),
    'pump_on': np.zeros(num_rows, dtype=int),
    'sensor_errors': np.zeros(num_rows, dtype=int),
}

# Add 6 realistic watering events
watering_indices = np.random.choice(range(50, num_rows-50), size=20, replace=False)
watering_indices.sort()

for idx in watering_indices:
    data['pump_on'][idx] = 1
    data['pulse_seconds'][idx] = np.random.uniform(4.0, 12.0)
    # Simulate moisture recovery after watering
    if idx + 30 < num_rows:
        data['soil_moisture_corrected'][idx:idx+30] = np.linspace(47, 52, 30) + np.random.normal(0, 0.4, 30)
    # Slight negative error before watering
    if idx > 10:
        data['avg_error_since_last'][idx-10:idx] = np.random.normal(-1.5, 0.5, 10)

df = pd.DataFrame(data)
df.to_csv('test_watering_log.csv', index=False)

print("✅ Created test_watering_log.csv")
print(f"   Total rows: {len(df)}")
print(f"   Watering events: {df['pump_on'].sum()}")
print("\nYou can now test with:")
print("   python3 train_lstm.py")
print("   (or rename test_watering_log.csv to watering_log.csv first)")