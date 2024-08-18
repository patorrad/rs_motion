from numpy import random
import pyrealsense2 as rs
import numpy as np
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

if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    ######################################################################
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    #####################################################################
    print("[INFO] Starting streaming...")
    pipeline.start(config)
    print("[INFO] Camera ready.")

    model = YOLO("yolov8n.pt")
    segmentation_model = YOLO("yolov8m-seg.pt")

    device = 'cpu'
    half = device 

    while True:
        
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth = frames.get_depth_frame()
        
        
        color_image = np.asanyarray(color_frame.get_data())
        im0s = color_image
        img = letterbox(im0s)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        depth_image = np.asanyarray(depth.get_data())
        de0s = depth_image
        deimg = letterbox(de0s)[0]
        hfov, vfov = 87, 58 # degrees

        # import pdb; pdb.set_trace()
        # deimg = deimg[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # deimg = np.ascontiguousarray(deimg)
        # deimg = torch.from_numpy(deimg).to(device)
        # deimg = deimg.half() if half else deimg.float()  # uint8 to fp16/32
        # deimg /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        
        results = model(img)
        seg_results = segmentation_model(img)
        # det_annotated = det_result[0].plot(show=False)
        # Process results list
        # for result, seg_result in zip(results, seg_results):
        boxes = results[0].boxes  # Boxes object for bounding box outputs
        masks = results[0].masks  # Masks object for segmentation masks outputs
        keypoints = results[0].keypoints  # Keypoints object for pose outputs
        probs = results[0].probs  # Probs object for classification outputs
        obb = results[0].obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        seg_results[0].show()
        # Get angle
        xyxy = seg_results[0].boxes.xyxy
        # import pdb; pdb.set_trace()
        if xyxy.size(0) == 0:
            print("No objects detected")
            continue
        else:
            for i in range(xyxy.size(0)):
                d1, d2 = int((int(xyxy[i][0])+int(xyxy[i][2]))/2), int((int(xyxy[i][1])+int(xyxy[i][3]))/2)
                x, y = get_hor_angle(d1, d2)
                print(f"Angle: {x}, {y}")
        time.sleep(1.)
            
        # if seg_results[0].boxes.cls.size(0) == 0:
        #     print("No objects detected")
        #     continue
        for index, cls in enumerate(seg_results[0].boxes.cls):
        
            class_index = int(cls.cpu().numpy())
            name = seg_results[0].names[class_index]
            # import pdb; pdb.set_trace()
            mask = seg_results[0].masks.data.cpu().numpy()[index, :, :].astype(int)
            obj = deimg[mask == 1]
            obj = obj[~np.isnan(obj)]
            avg_distance = np.mean(obj)/1000 if len(obj) else np.inf
            print(f"Object: {name}, Distance: {avg_distance} m")
        # import pdb; pdb.set_trace()
        # Process predictions
        # for i, det in enumerate(pred):  # per image
        #     seen += 1
        #     if webcam:  # batch_size >= 1
        #         p, im0, frame = path[i], im0s[i].copy(), dataset.count
        #         s += f'{i}: '
        #     else:
        #         p, s, im0 = path, '', im0s
        #         # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        #     p = Path(p)  # to Path
        #     save_path = str(save_dir / p.name)  # im.jpg
        #     txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        #     s += '%gx%g ' % im.shape[2:]  # print string
        #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     imc = im0.copy() if save_crop else im0  # for save_crop
        #     annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        #         # Print results
        #         for c in det[:, -1].unique():
        #             n = (det[:, -1] == c).sum()  # detections per class
        #             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             if save_txt:  # Write to file
        #                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #                 line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        #                 with open(f'{txt_path}.txt', 'a') as f:
        #                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

        #             if save_img or save_crop or view_img:  # Add bbox to image
        #                 c = int(cls)  # integer class
        #                 label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        #                 annotator.box_label(xyxy, label, color=colors(c, True))
        #                 d1, d2 = int((int(xyxy[0])+int(xyxy[2]))/2), int((int(xyxy[1])+int(xyxy[3]))/2)
        #                 zDepth = depth.get_distance(int(d1),int(d2))  # by default realsense returns distance in meters
        #                 tl = 3
        #                 tf = max(tl - 1, 1)  # font thickness
        #                 cv2.putText(im0, str(round((zDepth* 39.3701 ),2))+"in "+str(round((zDepth* 100 ),2))+" cm", (d1, d2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        #             if save_crop:
        #                 save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        #     # Stream results
        #     # im0 = annotator.result()
        #     # if view_img:
        # cv2.imshow(str(p), im0)
        # cv2.waitKey(1)  # 1 millisecond


