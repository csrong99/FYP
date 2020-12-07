######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import tensorflow as tf
import sys
import math

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'
# VIDEO_NAME = 'with_static_bg_v2.mp4'
# VIDEO_NAME = 'with_static_bg_v1.mp4'
VIDEO_NAME = 'streaming.mp4'
#VIDEO_NAME = 'test.mov'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
# PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph_v3.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME,'labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
# video = cv2.VideoCapture(PATH_TO_VIDEO)
# video = cv2.VideoCapture('http://192.168.0.169:8090/camera.mjpeg')
# video = cv2.VideoCapture('http://192.168.0.169:8090/v6.h264') # aloevera01
# video = cv2.VideoCapture('http://192.168.0.169:8090/v7.h264') # aloevera02
video = cv2.VideoCapture('http://192.168.0.169:8090/v8.h264') # aloevera01 & 02
fps = video.get(5)

area_of_interest = []

def find_area_of_interest(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([110, 60, 0])
    upper_red = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Converting the image to black and white
    (_, res) = cv2.threshold(res, 90, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    contours_2d = np.vstack(cnt.squeeze())

    # get the all index for xmin xmax ymin ymax
    xmin_contour = contours_2d[np.argmin(contours_2d[:,0]), :][0]
    xmax_contour = contours_2d[np.argmax(contours_2d[:,0]), :][0]
    ymin_contour = contours_2d[np.argmin(contours_2d[:,1]), :][1]
    ymax_contour = contours_2d[np.argmax(contours_2d[:,1]), :][1]

    return [ymin_contour, xmin_contour, ymax_contour, xmax_contour]

while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    current_frame_num = video.get(1)

    # Number of aloe vera to be separated
    av_split = 2

    if(current_frame_num % math.floor(fps) == 0):
        # Press 'q' to quit
        # if cv2.waitKey(1) == ord('q'):
        #     break
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_expanded = np.expand_dims(frame_rgb, axis=0)

        if len(area_of_interest) == 0:
            area_of_interest = find_area_of_interest(frame)
        
        ymin, xmin, ymax, xmax = area_of_interest

        frame = frame[ymin:ymax, xmin:xmax]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Clearing all unused scores, boxes, classes
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        min_score_thresh = 0.9
        av_xy = []
        
        del_idx = [i  for i in range(scores.shape[0]) if scores[i] < min_score_thresh]
        boxes = np.delete(boxes, del_idx, axis=0)
        classes = np.delete(classes, del_idx, axis=0)
        scores = np.delete(scores, del_idx, axis=0)
        
        # Calculate the height and width of each of the box in boxes
        avs_hw = []
        height, width, channels = frame.shape

        # board size in cm
        board_h = 92
        board_w = 183
        
        height_per_pixel = board_h / height
        width_per_pixel = board_w / width

        # Split the frame to av_split 
        EXPAND_ALL_SIDE_PIXEL = 10
        avs_placement_label = []

        split_points = [int(width / av_split * i) for i in range(av_split)]
        for i in range(boxes.shape[0]):
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            threshold_point = (xmax + xmin) * 0.5
            the_col_place =  sum([1 for x in split_points if threshold_point > x ])
            avs_placement_label.append(the_col_place)



        for i in range(boxes.shape[0]):
            # Calculating the height of the aloe vera detected
            ymin, xmin, ymax, xmax = boxes[i]
            
            xmin = math.ceil(xmin * width) - EXPAND_ALL_SIDE_PIXEL
            xmax = math.ceil(xmax * width) + EXPAND_ALL_SIDE_PIXEL
            ymin = math.ceil(ymin * height) - EXPAND_ALL_SIDE_PIXEL 
            ymax = math.ceil(ymax * height) + EXPAND_ALL_SIDE_PIXEL

            xmin = 0 if xmin < 0 else xmin
            ymin = 0 if ymin < 0 else ymin

            # xmin = int(xmin * width) - EXPAND_ALL_SIDE_PIXEL
            # xmax = int(xmax * width) + EXPAND_ALL_SIDE_PIXEL
            # ymin = int(ymin * height) - EXPAND_ALL_SIDE_PIXEL
            # ymax = int(ymax * height) + EXPAND_ALL_SIDE_PIXEL 

            # av_width = (xmax - xmin) * width_per_pixel
            # av_height = (ymax- ymin) * height_per_pixel
            # avs_hw.append(tuple([av_height, av_width]))

            # To conver the detected to back and white
            av_frame = frame[ymin:ymax, xmin:xmax]

            av_gray = cv2.cvtColor(av_frame, cv2.COLOR_BGR2GRAY)

            # Converting the image to black and white
            (threst, av_black_white) = cv2.threshold(av_gray, 90, 255, cv2.THRESH_BINARY)
            av_black_white = cv2.bitwise_not(av_black_white)

            # av_black_white = av_black_white.copy()
            # perform morphology oepration using dilation
            # kernal_dilation = np.ones((5,5), np.uint8)
            # dilation = cv2.dilate(av_gray, kernal_dilation, iterations=12)
            # (ret, thresh) = cv2.threshold(dilation, 127, 255, 0)

            # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # cnt = contours[0]
            # max_area = cv2.contourArea(cnt)

            # for cont in contours:
            #     if cv2.contourArea(cont) > max_area:
            #         cnt = cont
            #         max_area = cv2.contourArea(cont)

            # cv2.drawContours(av_frame, [cnt], 0, (255, 255, 0), 3)

            # cv2.imshow('Aloe Vera 02', av_frame)

            # Test
            _, contours, hierarchy = cv2.findContours(av_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


            cnt = contours[0]
            max_area = cv2.contourArea(cnt)

            for cont in contours:
                if cv2.contourArea(cont) > max_area:
                    cnt = cont
                    max_area = cv2.contourArea(cont)

            cv2.drawContours(av_frame, [cnt], 0, (255, 255, 0), 3)


            # av_frame = cv2.drawContours(av_frame, contours, -1, (0,0,255), 3)


            #height = 1080
            #width = 1920

            contours_2d = np.vstack(contours).squeeze()

            # get the all index for xmin xmax ymin ymax
            xmin_contour = contours_2d[np.argmin(contours_2d[:,0]), :]
            xmax_contour = contours_2d[np.argmax(contours_2d[:,0]), :]
            ymin_contour = contours_2d[np.argmin(contours_2d[:,1]), :]
            ymax_contour = contours_2d[np.argmax(contours_2d[:,1]), :]

            # draw line to connect xmin xmax and ymin ymax

            xmin_t = min(contours_2d[:,0])
            xmax_t = max(contours_2d[:,0])
            ymin_t = min(contours_2d[:,1])
            ymax_t = max(contours_2d[:,1])
            y_mp = int(av_frame.shape[0] / 2)
            x_mp = int(av_frame.shape[1] / 2)

            av_width = (xmax_t - xmin_t) * width_per_pixel
            av_height = (ymax_t- ymin_t) * height_per_pixel
            avs_hw.append(tuple([av_height, av_width]))

            
            cv2.line(av_frame, (xmin_t, y_mp), (xmax_t, y_mp), (0,0,255), 5)
            cv2.line(av_frame, (x_mp, ymin_t), (x_mp, ymax_t), (0,0,255), 5)

            # plot the contour for xmin xmax ymin ymax

            cv2.circle(av_frame, tuple(xmin_contour), 2, (0,0,255), 5)
            cv2.circle(av_frame, tuple(xmax_contour), 2, (0,0,255), 5)
            cv2.circle(av_frame, tuple(ymin_contour), 2, (0,0,255), 5)
            cv2.circle(av_frame, tuple(ymax_contour), 2, (0,0,255), 5)

            av_black_white = cv2.resize(av_black_white, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            av_frame = cv2.resize(av_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Black and white image {}".format(i), av_black_white)

            cv2.imshow('Aloe Vera 02', av_frame)
            


        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            boxes,
            classes,
            scores,
            category_index,
            avs_hw,
            avs_placement_label=avs_placement_label,
            col_to_split=av_split,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=min_score_thresh,
            max_boxes_to_draw=50)
        
        frameResize = cv2.resize(frame, (800, 600))

        cv2.imshow('Object detector', frameResize)
        cv2.waitKey(10)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
