# Import packages
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
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

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'object_detection\\model'
VIDEO_NAME = 'streaming.mp4'
#VIDEO_NAME = 'test.mov'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph_v3.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, 'labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
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

area_of_interest = []
av_split = 2
EXPAND_ALL_SIDE_PIXEL = 10

# board size in cm
board_h = 92
board_w = 183


def find_area_of_interest(frame):
    """Locate the area of interest of the image

    Args:
        frame: uint8 numpy array with shape (img_height, img_width, 3)

    Returns:
        [tuple]: [ymin_contour, xmin_contour, ymax_contour, xmax_contour]
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([110, 60, 0])
    upper_red = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Converting the image to black and white
    (_, res) = cv2.threshold(res, 90, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    
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

def detect(frame):
    global area_of_interest
    if len(area_of_interest) == 0:
        area_of_interest = find_area_of_interest(frame)
    
    ymin, xmin, ymax, xmax = area_of_interest

    frame = frame[ymin:ymax, xmin:xmax]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, _) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Clearing all unused scores, boxes, classes
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes).astype(np.int32)
    min_score_thresh = 0.9
    
    del_idx = [i  for i in range(scores.shape[0]) if scores[i] < min_score_thresh]
    boxes = np.delete(boxes, del_idx, axis=0)
    classes = np.delete(classes, del_idx, axis=0)
    scores = np.delete(scores, del_idx, axis=0)
    
    # Calculate the height and width of each of the box in boxes
    avs_hw = []
    avs_result = {}
    height, width, _ = frame.shape

    height_per_pixel = board_h / height
    width_per_pixel = board_w / width

    avs_placement_label = []

    split_points = [int(width / av_split * i) for i in range(av_split)]

    for i in range(boxes.shape[0]):
        # Calculating the height of the aloe vera detected
        ymin, xmin, ymax, xmax = boxes[i]

        threshold_point = int( (xmax + xmin) * width) * 0.5 
        the_col_place =  sum([1 for x in split_points if threshold_point > x ])
        avs_placement_label.append(the_col_place)
        
        xmin = math.ceil(xmin * width) - EXPAND_ALL_SIDE_PIXEL
        xmax = math.ceil(xmax * width) + EXPAND_ALL_SIDE_PIXEL
        ymin = math.ceil(ymin * height) - EXPAND_ALL_SIDE_PIXEL 
        ymax = math.ceil(ymax * height) + EXPAND_ALL_SIDE_PIXEL

        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin

        # To conver the detected to back and white
        av_frame = frame[ymin:ymax, xmin:xmax]

        av_gray = cv2.cvtColor(av_frame, cv2.COLOR_BGR2GRAY)

        # Converting the image to black and white
        (_, av_black_white) = cv2.threshold(av_gray, 90, 255, cv2.THRESH_BINARY)
        av_black_white = cv2.bitwise_not(av_black_white)

        contours, _ = cv2.findContours(av_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


        cnt = contours[0]
        max_area = cv2.contourArea(cnt)

        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        cv2.drawContours(av_frame, [cnt], 0, (255, 255, 0), 3)

        contours_2d = np.vstack(contours).squeeze()

        xmin_t = min(contours_2d[:,0])
        xmax_t = max(contours_2d[:,0])
        ymin_t = min(contours_2d[:,1])
        ymax_t = max(contours_2d[:,1])

        
        av_width = (xmax_t - xmin_t) * width_per_pixel
        av_height = (ymax_t- ymin_t) * height_per_pixel
        avs_hw.append(tuple([av_height, av_width]))
        avs_result[the_col_place] = tuple([av_height, av_width])

        av_frame = cv2.resize(av_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

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
        line_thickness=2,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=50,
        skip_scores=True)
    
    frameResize = cv2.resize(frame, (800, 600))

    return frameResize, avs_result

def check_av_grow_condition(height):
    if height < 30:
        return "baby"
    elif height >= 30 and height < 60:
        return "normal"
    else:
        return "adult"

def check_av_health_condition(height, width):
    hw_ratio = height / width
    if hw_ratio > 1.0:
        return "Healthy"
    else:
        return "Not Healthy"
