#! /usr/bin/env python3
import serial
import time

# Set the serial port and baud rate
ser = serial.Serial('/dev/ttyACM1', 115200, timeout=1)
ser.write(b'A')
# Allow time for the serial connection to initialize
time.sleep(2)

# Function to read serial data
def read_serial():
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(line)

try:
    read_serial()
except KeyboardInterrupt:
    print("Serial reading stopped")
finally:
    ser.close()

