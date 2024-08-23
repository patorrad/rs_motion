#! /usr/bin/env python3
import serial
import time
import rospy
from std_msgs.msg import Float64MultiArray

# Set the serial port and baud rate
ser = serial.Serial('/dev/ttyACM1', 115200, timeout=1)
ser.write(b'A')
# Allow time for the serial connection to initialize
time.sleep(2)


if __name__ == "__main__":
    rospy.init_node('yaw', anonymous=True)
    pub = rospy.Publisher('/yaw', Float64MultiArray, queue_size=10)
    yaw = 0
    while not rospy.is_shutdown():
        #rospy.sleep(0.05)
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()

            # Split the line into parts
            parts = line.split()

            # Use a dictionary comprehension to extract the values
            values = {part.split(":")[0]: float(part.split(":")[1]) for part in parts}

            roll_value = values["Roll"]
            pitch_value = values["Pitch"]
            yaw_value = values["Yaw"]
        
            yaw_msg = Float64MultiArray()
            yaw_msg.data = [float(yaw_value)]
            pub.publish(yaw_msg)
    
    rospy.loginfo("Closing serial")
    ser.close()
