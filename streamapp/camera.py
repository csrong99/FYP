from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
import tensorflow as tf
from object_detection import av_detection_analysis as avda
from datetime import datetime
from streamapp.templatetags import realtime_firebase as db
import os

curr_path = os.getcwd()

class VideoCamera(object):
	def __init__(self):
		# self.video = cv2.VideoCapture(0)
		self.result = {}
		# self.video = cv2.VideoCapture('http://192.168.0.169:8090/camera.mjpeg')
		# self.video = cv2.VideoCapture('http://192.168.0.169:8090/v6.h264') # aloevera01
		# self.video = cv2.VideoCapture('http://192.168.0.169:8090/v7.h264') # aloevera02
		# self.video = cv2.VideoCapture('http://192.168.0.169:8090/v8.h264') # aloevera01 & 02
		
		self.video = cv2.VideoCapture(curr_path + '\\streamapp\\v8.h264')

	def __del__(self):
		self.video.release()

	def get_frame(self):
		_, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		image, result = avda.detect(image)
		if len(result) != 0:
			for key in result.keys():
				w, h = result[key]
				if key not in self.result.keys():
					self.result[key] = [(w, h)]
				else:
					if len(self.result[key]) != 5:
						self.result[key].append(tuple([w, h]))
					else:
						self.result[key].pop(0)
						self.result[key].append(tuple([w, h]))

		_, jpeg = cv2.imencode('.jpg', image)

		return jpeg.tobytes()

	def update(self):
		plant_details = []

		now_datetime = datetime.now()
		now_datetime_str = now_datetime.strftime("%d/%m/%Y %H:%M:%S")
		
		for key in range(1, 3):
			if key in self.result.keys():
				while len(self.result[key]) < 5:
					continue
				height = sum([ h for h, _ in self.result[key]])
				width = sum([ w for _, w in self.result[key]])
				height = height / 5.0
				width = width / 5.0

				# return the status for the alov vera
				av_growth = avda.check_av_grow_condition(height)

				id = "av{:02}".format(key)
				av_dict = {
					'condition': av_growth,
					'datetime': now_datetime_str,
					'height': round(height,2),
					'width': round(width,2)
				}
			else:
				id = "av{:02}".format(key)
				av_dict = {
					'condition': "Null",
					'datetime': now_datetime_str,
					'height': 0.0,
					'width': 0.0
				}

			db.update_aloe_vera(id, av_dict)

		return plant_details
