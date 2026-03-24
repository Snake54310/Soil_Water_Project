import RPi.GPIO as GPIO
import time

# Set up GPIO
GPIO.setmode(GPIO.BCM)          # or GPIO.BOARD for physical pin numbers
CONTROL_PIN = 22                # Change to your chosen GPIO pin. currently GPIO 22 (pin 15)
GPIO.setup(CONTROL_PIN, GPIO.OUT)

try:
    print("Turning MOSFET ON (load powered)...")
    GPIO.output(CONTROL_PIN, GPIO.HIGH)  # High = ON (module logic: high turns MOSFET on)
    time.sleep(5)

    print("Turning MOSFET OFF...")
    GPIO.output(CONTROL_PIN, GPIO.LOW)   # Low = OFF
    time.sleep(5)
    '''
    # Optional: PWM example for dimming (e.g., LED strip)
    pwm = GPIO.PWM(CONTROL_PIN, 1000)    # 1kHz frequency (module supports up to ~1kHz)
    pwm.start(0)                         # Start at 0% duty
    for duty in range(0, 101, 5):        # Fade in
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.1)
    pwm.stop()
    '''

finally:
    GPIO.cleanup()                       # Clean up on exit
