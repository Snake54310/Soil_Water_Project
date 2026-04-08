from w1thermsensor import W1ThermSensor
import RPi.GPIO as GPIO
import time
import board
import adafruit_hts221
from adafruit_seesaw.seesaw import Seesaw
import sys   # for clean exit if needed

class Input_Output_Operations:
    def __init__(self, minimumUptime, maximumUptime):
        self.groundTemp = 0
        self.airtemp = 0
        self.humidity = 0
        self.soilMoisture = 0
        self.minUptime = minimumUptime
        self.maxUptime = maximumUptime

        # ====================== STEMMA SOIL SENSOR CALIBRATION ======================
        self.DRY_1 = 500        # raw value for sensor at 0x36 when completely dry
        self.WET_1 = 1016      # raw value for sensor at 0x36 when fully saturated
        self.DRY_2 = 500        # raw value for sensor at 0x37 when completely dry
        self.WET_2 = 1016      # raw value for sensor at 0x37 when fully saturated
        # =============================================================================

    def detect_stemma(self):
        """Returns True only if BOTH STEMMA sensors are detected on I2C."""
        try:
            i2c = board.I2C()
            ss1 = Seesaw(i2c, addr=0x36)
            _ = ss1.moisture_read()          # dummy read to test
            ss2 = Seesaw(i2c, addr=0x37)
            _ = ss2.moisture_read()
            return True
        except Exception:
            return False

    def readGroundTemp(self):
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(4, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            sensor = W1ThermSensor()
            temperature_in_c = sensor.get_temperature()
            self.groundTemp = temperature_in_c
            return temperature_in_c
        except Exception:
            self.groundTemp = 15.0
            return 15.0   # DS18B20 failed — continue anyway

    def readHumidityAndTemp(self):
        try:
            i2c = board.I2C()
            hts = adafruit_hts221.HTS221(i2c)
            temp_humid = [hts.temperature, hts.humidity]
            self.airtemp = temp_humid[0]
            self.humidity = temp_humid[1]
            return temp_humid
        except Exception:
            self.airtemp = 0.0
            self.humidity = 0.0
            return [0.0, 0.0]   # HTS221 failed — continue anyway

    def activatePump(self, pumpTimer=5):
        GPIO.setmode(GPIO.BCM)
        CONTROL_PIN = 22
        GPIO.setup(CONTROL_PIN, GPIO.OUT)
        try:
            GPIO.output(CONTROL_PIN, GPIO.HIGH)
            time.sleep(pumpTimer)
            GPIO.output(CONTROL_PIN, GPIO.LOW)
        finally:
            GPIO.cleanup()
        return

    def getGroundMoisture(self):
        """Returns calibrated soil moisture percentage (0-60) where:
        raw 300 ≈ 0%
        raw 1016 ≈ 60% (sensor's practical maximum)"""
        try:
            i2c_bus = board.I2C()

            # Sensor 1 at 0x36
            ss1 = Seesaw(i2c_bus, addr=0x36)
            raw1 = ss1.moisture_read()

            # Sensor 2 at 0x37
            ss2 = Seesaw(i2c_bus, addr=0x37)
            raw2 = ss2.moisture_read()

            # Calibration constants
            DRY_RAW = 500  # Update this after measuring completely dry soil
            MAX_RAW = 1016  # Sensor's practical maximum ≈ 60%

            def raw_to_percent(raw):
                if raw <= DRY_RAW:
                    return 0.0
                elif raw >= MAX_RAW:
                    return 100.0
                else:
                    # Linear mapping from 500 → 0% to 1016 → 100%
                    return ((raw - DRY_RAW) / (MAX_RAW - DRY_RAW)) * 100.0

            percent1 = raw_to_percent(raw1)
            percent2 = raw_to_percent(raw2)

            avg_percent = (percent1 + percent2) / 2.0
            self.soilMoisture = avg_percent
            return avg_percent
        except Exception as e:
            print(f"STEMMA read error: {e}")
            self.soilMoisture = 50.0 # if read fails, return basic value
            return 50.0

    def readAll(self, pumpTimer=5):
        GroundTemp = self.readGroundTemp()
        TempAndHumidity = self.readHumidityAndTemp()
        GroundMoisture = self.getGroundMoisture()
        # ground temperature [0], air temperature [1], humidity [2], soil moisture [3]
        input_list = [GroundTemp, TempAndHumidity[0], TempAndHumidity[1], GroundMoisture]
        return input_list