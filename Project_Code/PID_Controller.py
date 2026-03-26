# PID_Controller.py
# Pure PID controller for the AI-Regulated Smart Watering System
# Used by main script to compute pump pulse duty based on soil moisture error.
# No hard-coded intelligence — pure feedback only (LSTM will override gains later).

import time

class PID:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def compute(self, measurement):
        """Compute PID output (0.0–1.0) from current soil moisture reading."""
        error = self.setpoint - measurement
        now = time.time()
        dt = now - self.prev_time

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = max(min(self.integral, 100), -100)

        # Derivative
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        # PID output
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.prev_error = error
        self.prev_time = now

        # Clamp to valid pump duty range (0–1)
        return max(0.0, min(1.0, output))