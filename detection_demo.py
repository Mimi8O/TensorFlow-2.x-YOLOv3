#================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

image_path   = "./IMAGES/street.jpg"
output_path  = "/kaggle/working/street_pred.jpg"  # 결과 이미지를 저장할 경로
video_path   = "./IMAGES/test.mp4"

yolo = Load_Yolo_model()
detect_image(yolo, image_path, output_path, input_size=YOLO_INPUT_SIZE, rectangle_colors=(255,0,0))
#detect_video(yolo, video_path, "", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0), realtime=False)

# 이미지 저장 확인
if os.path.exists(output_path):
    print(f"이미지가 성공적으로 저장되었습니다: {output_path}")
else:
    print("이미지가 저장되지 않았습니다.")
