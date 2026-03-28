# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time

import board

from adafruit_seesaw.seesaw import Seesaw

i2c_bus = board.I2C()  # uses board.SCL and board.SDA
# i2c_bus = board.STEMMA_I2C()  # For using the built-in STEMMA QT connector on a microcontroller

ss = Seesaw(i2c_bus, addr=0x37) # 0x37 for second sensor

while True:
    # read moisture level through capacitive touch pad
    touch = ss.moisture_read()

    # read temperature from the temperature sensor
    temp = ss.get_temp()

    print("temp: " + str(temp) + "  moisture: " + str(touch))
    time.sleep(1)


# Or Else:
'''
#include <Wire.h>

// I2C address of the soil sensor
const int soilSensorAddress = 0x36;

void setup() {
  Serial.begin(9600);
  Wire.begin(); // Join I2C bus
}

void loop() {
  Wire.beginTransmission(soilSensorAddress);
  Wire.write(0x0F); // Send command to read moisture
  Wire.endTransmission();
  Wire.requestFrom(soilSensorAddress, 2); // Request 2 bytes from sensor
  
  if (Wire.available() == 2) {
    int soilMoisture = Wire.read() << 8; // Read high byte
    soilMoisture |= Wire.read(); // Read low byte and combine with high byte
    
    Serial.print("Soil Moisture Level: ");
    Serial.println(soilMoisture);
  }
  
  delay(1000); // Wait for a second before reading again
}'''
