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

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'
VIDEO_NAME = 'streaming.mp4'
#VIDEO_NAME = 'test.mov'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

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
video = cv2.VideoCapture('C:\\Users\\Chuah\\Desktop\\FYP\\backup_out.mp4')

fps = video.get(5)

while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    current_frame_num = video.get(1)

    #Converts BGR color space of iamge to HSV color space
    #hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #brown = np.uint8([[[66, 100, 159]]]) 
    #hsvBrown = cv2.cvtColor(brown, cv2.COLOR_BGR2HSV)
    #print(hsvBrown)

    # define range of brown color in HSV
    #lower_brown = hsvBrown[0][0][0] - 10, 100, 100
    #upper_brown = hsvBrown[0][0][0] + 10, 255, 255

    #print(lower_brown)
    #print(upper_brown)

    # Threshold the HSV image to get only brown colors
    #mask = cv2.inRange(hsv, (1,100,100), (21,255,255))

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)

    if(current_frame_num % math.floor(fps) == 0):
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            # col_to_split=2,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.95)  

        #height = 1080
        #width = 1920
        #height, width, channels = frame.shape
        #ymin = boxes[0][0][0]
        #xmin = boxes[0][0][1]
        #ymax = boxes[0][0][2]
        #xmax = boxes[0][0][3]
                
        #chg var name, mp = midpoint
        #x_mp = (xmin + xmax) / 2 * width
        #y_mp = (ymin + ymax) / 2 * height
        #test = 5
        #label = 'Height: {:.2f} cm'.format(test)
        #cv2.putText(frame, label, (int(x_mp)-150, int(y_mp)),
        #        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),2)


        # All the results have been drawn on the frame, so it's time to display it.
        frameResize = cv2.resize(frame, (800, 400))
        #maskResize = cv2.resize(mask, (960, 540))
        #resResize = cv2.resize(res, (960, 540))
        #cropimg = frameResize.crop(0,0,250,250)
        #cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Object detector', 1280, 960)
        cv2.imshow('Object detector', frameResize)
        #cv2.imshow('mask',maskResize)
        #cv2.imshow('Mask',resResize)
        #cv2.imshow('hi', cropimg) 
        cv2.waitKey(10)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
