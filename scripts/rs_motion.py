#! /usr/bin/env python3

import pyrealsense2 as rs
import math
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
import numpy as np

from numpy import random
from ultralytics import YOLO
import cv2
import torch
import time
import sys
np.set_printoptions(threshold=sys.maxsize)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)
    
'''
Get angle of object from center of camera
Input
Output
hfov, vfov = 87, 58 # degrees for depth of D435
'''
def get_hor_angle(x, y, W=640, H=384, HFOV=87, VFOV=58):
    # Say you are using RGB camera and have set it to 720p resolution. W = 1280, H = 720
    # Assume the HFOV = 62 and VFOV=46 (Check values for your camera, assumed at random)
    # Center of camera in image
    # (W/2,H/2) = (W, H)

    # Horizontal Angle of say random pixel 
    x = ((x - W/2)/(W/2)) * (HFOV/2)
    # Vertical Angle of say random pixel
    y = ((y - H/2)/(H/2)) * (VFOV/2)

    # For a random pixel say (320,180) it will be:
    # H_Angle = ((320 - 640)/640)(62/2) = - 15.5 degree
    # V_Angle = ((180 - 360)/360)(46/2) = - 11.5 degree

    # You can get the euclidean angle by using simple (h-angle^2+v-angle^2)^0.5
    return x, y

def initialize_camera_imu():
    # start the frames pipe
    p = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    
    rospy.loginfo("Starting imu streaming...")
    prof = p.start(config)
    rospy.loginfo("Camera imu ready.")
    return p
    
def initialize_camera_color_depth():
    # start the frames pipe
    p = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    
    rospy.loginfo("Starting color and depth streaming...")
    prof = p.start(config)
    rospy.loginfo("Camera ready.")
    return p

class imu:
    def __init__(self):
        self.accel = None
        self.gyro = None
        self.ts = None    
        rospy.Subscriber("/car/car/camera/accel/sample", Imu, self.callback_accel)
        rospy.Subscriber("/car/car/camera/gyro/sample", Imu, self..callback_gyro)
        rospy.sleep(1)
        
    def callback_accel(self, msg):
        self.accel = msg.linear_acceleration
    
    def callback_gyro(self, msg):
        self.gyro = msg.angular_velocity
        self.ts = msg.header.stamp.secs

first = True
alpha = 0.98
totalgyroangleY = 0 

# Start yolo
model = YOLO("yolov8n.pt")
segmentation_model = YOLO("yolov8m-seg.pt")

device = 'cpu'
half = device 

if __name__ == "__main__":

    rospy.init_node('rs_motion', anonymous=True)
    pub_imu_angles = rospy.Publisher('imu_angles', Float64MultiArray, queue_size=10)
    pub_target_angles = rospy.Publisher('target_angles', Float64MultiArray, queue_size=10)
    
    imu = imu()
    print(imu.accel)
    print(imu.gyro)
    print(imu.ts)
    #pipeline_imu = initialize_camera_imu()
    #pipeline_color_depth = initialize_camera_color_depth()
    
    try:
      while not rospy.is_shutdown():
          #frames_imu = pipeline_imu.wait_for_frames()
          
          ## Gyro calculations
          #gather IMU data
          #accel = frames_imu[0].as_motion_frame().get_motion_data()
          #gyro = frames_imu[1].as_motion_frame().get_motion_data()
#          import pdb; pdb.set_trace()
          #<class 'pyrealsense2.pyrealsense2.vector'>
          #(Pdb) accel
          #x: -0.0643611, y: -9.94515, z: -0.450236
          #(Pdb) gyro
          #x: -5.27443e-06, y: -0.0034287, z: -1.18961e-05


          #ts = frames_imu.get_timestamp()
          accel = imu.accel
          gyro = imu.gyro
          ts = imu.ts

          #calculation for the first frame
          if (first):
              first = False
              last_ts_gyro = ts

              # accelerometer calculation
              accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
              accel_angle_x = math.degrees(math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
              accel_angle_y = math.degrees(math.pi)
  
              continue

          #calculation for the second frame onwards

          # gyrometer calculations
          dt_gyro = (ts - last_ts_gyro) / 1000
          last_ts_gyro = ts

          gyro_angle_x = gyro.x * dt_gyro
          gyro_angle_y = gyro.y * dt_gyro
          gyro_angle_z = gyro.z * dt_gyro

          dangleX = gyro_angle_x * 57.2958
          dangleY = gyro_angle_y * 57.2958
          dangleZ = gyro_angle_z * 57.2958

          totalgyroangleX = accel_angle_x + dangleX
          #totalgyroangleY = accel_angle_y + dangleY
          totalgyroangleY = accel_angle_y + dangleY + totalgyroangleY
          totalgyroangleZ = accel_angle_z + dangleZ

          #accelerometer calculation
          accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
          accel_angle_x = math.degrees(math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
          #accel_angle_y = math.degrees(math.pi)
          accel_angle_y = 0


          #combining gyrometer and accelerometer angles
          combinedangleX = totalgyroangleX * alpha + accel_angle_x * (1-alpha)
          combinedangleZ = totalgyroangleZ * alpha + accel_angle_z * (1-alpha)
          combinedangleY = totalgyroangleY

          data_to_send = Float64MultiArray()  # the data to be sent, initialise the array
          data_to_send.data = np.array([round(combinedangleX,2), round(combinedangleY,2), round(combinedangleZ,2)]) # assign the array with the value you want to send
          pub_imu_angles.publish(data_to_send)
          #print("Angle -  X: " + str(round(combinedangleX,2)) + "   Y: " + str(round(combinedangleY,2)) + "   Z: " + str(round(combinedangleZ,2)))
          
#          ## Yolo calculations
#          frames = pipeline_color_depth.wait_for_frames()
#          color_frame = frames.get_color_frame()
#          depth = frames.get_depth_frame()
#          
#          color_image = np.asanyarray(color_frame.get_data())
#          im0s = color_image
#          img = letterbox(im0s)[0]
#          img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#          img = np.ascontiguousarray(img)
#          img = torch.from_numpy(img).to(device)
#          img = img.half() if half else img.float()  # uint8 to fp16/32
#          img /= 255.0  # 0 - 255 to 0.0 - 1.0
#          if img.ndimension() == 3:
#              img = img.unsqueeze(0)
#        
#          depth_image = np.asanyarray(depth.get_data())
#          de0s = depth_image
#          deimg = letterbox(de0s)[0]
#          hfov, vfov = 87, 58 # degrees
#          results = model(img)
#          seg_results = segmentation_model(img)
#
#          boxes = results[0].boxes  # Boxes object for bounding box outputs
#          masks = results[0].masks  # Masks object for segmentation masks outputs
#          keypoints = results[0].keypoints  # Keypoints object for pose outputs
#          probs = results[0].probs  # Probs object for classification outputs
#          obb = results[0].obb  # Oriented boxes object for OBB outputs
#          # result.show()  # display to screen
#          # seg_results[0].show()
#          # Get angle
#          xyxy = seg_results[0].boxes.xyxy
#
#          if xyxy.size(0) == 0:
#              print("No objects detected")
#              continue
#          else:
#              for i in range(xyxy.size(0)):
#                  d1, d2 = int((int(xyxy[i][0])+int(xyxy[i][2]))/2), int((int(xyxy[i][1])+int(xyxy[i][3]))/2)
#                  x, y = get_hor_angle(d1, d2)
#                  print(f"Angle: {x}, {y}")
#          time.sleep(1.)
#            
#
#          for index, cls in enumerate(seg_results[0].boxes.cls):
#          
#              class_index = int(cls.cpu().numpy())
#              name = seg_results[0].names[class_index]    
#              mask = seg_results[0].masks.data.cpu().numpy()[index, :, :].astype(int)
#              obj = deimg[mask == 1]
#              obj = obj[~np.isnan(obj)]
#              avg_distance = np.mean(obj)/1000 if len(obj) else np.inf
#              print(f"Object: {name}, Distance: {avg_distance} m")
              
#          target_angles_to_send = Float64MultiArray()  # the data to be sent, initialise the array
#          target_angles_to_send.data = np.array([]) # assign the array with the value you want to send
#          pub_target_angles.pub(target_angles_to_send)

    finally:
      pass
      #pipeline_imu.stop()
      #pipeline_color_depth.stop()
