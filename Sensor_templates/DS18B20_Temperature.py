from w1thermsensor import W1ThermSensor
# Note: this will occasionally fail.. maybe sensor does not default to 'wake' state on power?
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.IN, pull_up_down=GPIO.PUD_UP) # GPIO 4 (pin 7) is default for w1thermsensor package

sensor = W1ThermSensor()           # auto-detects the first sensor
temperature_in_c = sensor.get_temperature()
# or: temperature_in_f = sensor.get_temperature(W1ThermSensor.DEGREES_F)
print(f"Temperature: {temperature_in_c:.2f} °C")
