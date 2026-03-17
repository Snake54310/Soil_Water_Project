from w1thermsensor import W1ThermSensor

sensor = W1ThermSensor()           # auto-detects the first sensor
temperature_in_c = sensor.get_temperature()
# or: temperature_in_f = sensor.get_temperature(W1ThermSensor.DEGREES_F)
print(f"Temperature: {temperature_in_c:.2f} °C")