from w1thermsensor import W1ThermSensor
import RPi.GPIO as GPIO
import time
import board
import adafruit_hts221
from adafruit_seesaw.seesaw import Seesaw

# Reads ground temperature from class Input_Output_Operations:

class Input_Output_Operations:
    def __init__(self, minimumUptime, maximumUptime):
        self.groundTemp = 0
        self.airtemp = 0
        self.humidity = 0
        self.soilMoisture = 0   # Now stores averaged calibrated percentage (0-100)
        self.minUptime = minimumUptime
        self.maxUptime = maximumUptime

        # ====================== STEMMA SOIL SENSOR CALIBRATION ======================
        # These must be measured in your actual soil:
        #   1. Completely dry soil → record raw value for each sensor
        #   2. Fully saturated soil → record raw value for each sensor
        # Then set the constants below.  (Typical starting guesses: dry ~300, wet ~1800)
        self.DRY_1 = 300        # raw value for sensor at 0x36 when completely dry
        self.WET_1 = 1800       # raw value for sensor at 0x36 when fully saturated
        self.DRY_2 = 300        # raw value for sensor at 0x37 when completely dry
        self.WET_2 = 1800       # raw value for sensor at 0x37 when fully saturated
        # =============================================================================

    def readGroundTemp(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(4, GPIO.IN, pull_up_down=GPIO.PUD_UP) # GPIO 4 (pin 7) is default for w1thermsensor package
        sensor = W1ThermSensor() # auto-detects the first sensor
        temperature_in_c = sensor.get_temperature() # or: temperature_in_f = sensor.get_temperature(W1ThermSensor.DEGREES_F)
        # print(f"Temperature: {temperature_in_c:.2f} °C")
        self.groundTemp = temperature_in_c
        return temperature_in_c

    def readHumidityAndTemp(self):
        i2c = board.I2C() # uses board.SCL and board.SDA
        # i2c = board.STEMMA_I2C() # For using the built-in STEMMA QT connector on a microcontroller
        hts = adafruit_hts221.HTS221(i2c)
        #data_rate = adafruit_hts221.Rate.label[hts.data_rate]
        #print(f"Using data rate of: {data_rate:.1f} Hz")
        #print("")
        # while True:
        #print(f"Relative Humidity: {hts.relative_humidity:.2f} % rH")
        #print(f"Temperature: {hts.temperature:.2f} C")
        #print("")
        #time.sleep(1)
        temp_humid = [hts.temperature, hts.humidity]
        self.airtemp = temp_humid[0]
        self.humidity = temp_humid[1]
        return temp_humid

    def activatePump(self, pumpTimer=5):
        # Set up GPIO
        GPIO.setmode(GPIO.BCM) # or GPIO.BOARD for physical pin numbers
        CONTROL_PIN = 22 # Change to your chosen GPIO pin. currently GPIO 22 (pin 15)
        GPIO.setup(CONTROL_PIN, GPIO.OUT)
        try:
            #print("Turning MOSFET ON (load powered...)")
            GPIO.output(CONTROL_PIN, GPIO.HIGH) # High = ON (module logic: high turns MOSFET on)
            time.sleep(pumpTimer)
            #print("Turning MOSFET OFF...")
            GPIO.output(CONTROL_PIN, GPIO.LOW)
        finally:
            GPIO.cleanup() # Clean up on exit
        return

    def getGroundMoisture(self):
        """Returns averaged calibrated soil moisture percentage (0-100) from BOTH STEMMA sensors"""
        i2c_bus = board.I2C()

        # Sensor 1 at 0x36
        ss1 = Seesaw(i2c_bus, addr=0x36)
        raw1 = ss1.moisture_read()

        # Sensor 2 at 0x37
        ss2 = Seesaw(i2c_bus, addr=0x37)
        raw2 = ss2.moisture_read()

        # Convert each raw reading to percentage independently
        percent1 = ((raw1 - self.DRY_1) / (self.WET_1 - self.DRY_1)) * 100
        percent2 = ((raw2 - self.DRY_2) / (self.WET_2 - self.DRY_2)) * 100

        # Clamp to 0-100
        percent1 = max(0, min(100, percent1))
        percent2 = max(0, min(100, percent2))

        # Average the two calibrated percentages
        avg_percent = (percent1 + percent2) / 2

        self.soilMoisture = avg_percent
        return avg_percent

    def readAll(self, pumpTimer=5):
        GroundTemp = self.readGroundTemp()
        TempAndHumidity = self.readHumidityAndTemp()
        GroundMoisture = self.getGroundMoisture()
        # ground temperature [0], air temperature [1], humidity [2], soil moisture [3]
        # Note: soil moisture [3] is now the averaged calibrated percentage (0-100)
        input_list = [GroundTemp, TempAndHumidity[0], TempAndHumidity[1], GroundMoisture]
        return input_list