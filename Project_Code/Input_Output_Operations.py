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
        self.DRY_1 = 300        # raw value for sensor at 0x36 when completely dry
        self.WET_1 = 1800       # raw value for sensor at 0x36 when fully saturated
        self.DRY_2 = 300        # raw value for sensor at 0x37 when completely dry
        self.WET_2 = 1800       # raw value for sensor at 0x37 when fully saturated
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
            self.groundTemp = 0.0
            return 0.0   # DS18B20 failed — continue anyway

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
        """Returns averaged calibrated soil moisture percentage (0-100)."""
        try:
            i2c_bus = board.I2C()
            ss1 = Seesaw(i2c_bus, addr=0x36)
            raw1 = ss1.moisture_read()
            ss2 = Seesaw(i2c_bus, addr=0x37)
            raw2 = ss2.moisture_read()

            percent1 = ((raw1 - self.DRY_1) / (self.WET_1 - self.DRY_1)) * 100
            percent2 = ((raw2 - self.DRY_2) / (self.WET_2 - self.DRY_2)) * 100

            percent1 = max(0, min(100, percent1))
            percent2 = max(0, min(100, percent2))

            avg_percent = (percent1 + percent2) / 2
            self.soilMoisture = avg_percent
            return avg_percent
        except Exception:
            self.soilMoisture = 0.0
            return 0.0   # STEMMA failed — continue (though startup check already caught this)

    def readAll(self, pumpTimer=5):
        GroundTemp = self.readGroundTemp()
        TempAndHumidity = self.readHumidityAndTemp()
        GroundMoisture = self.getGroundMoisture()
        # ground temperature [0], air temperature [1], humidity [2], soil moisture [3]
        input_list = [GroundTemp, TempAndHumidity[0], TempAndHumidity[1], GroundMoisture]
        return input_list