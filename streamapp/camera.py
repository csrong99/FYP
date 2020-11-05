from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
import tensorflow as tf
from object_detection.av_detection import detect
from datetime import datetime
from streamapp.templatetags import realtime_firebase as db

class VideoCamera(object):
	def __init__(self):
		# self.video = cv2.VideoCapture(0)
		self.result = []
		# self.video = cv2.VideoCapture('http://192.168.0.169:8090/camera.mjpeg')
		# self.video = cv2.VideoCapture('C:\\Users\\Chuah\\Desktop\\FYP\\backup_out.mp4')

	def __del__(self):
		self.video.release()

	def get_frame(self):
		_, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		image, result = detect(image)
		if len(result) != 0 and len(self.result) != 5:
			self.result.append(result)
		elif len(self.result) == 5:
			self.result.pop(0)
			self.result.append(result)

		_, jpeg = cv2.imencode('.jpg', image)

		return jpeg.tobytes()

	def update(self):
		plant_details = []
		while True:
			if len(self.result) >= 5:
				for img in self.result[:5]:
					idx = 0
					for plant in img:
						h, w = plant
						if len(plant_details) < len(img):
							plant_details.append((h,w))
						else:
							h1, w1 = plant_details[idx] 
							plant_details[idx] = (h1 +h, w1+w)
						idx += 1
				break
		plant_details = [(h/5, w/5) for h, w in plant_details]

		now_datetime = datetime.now()
		now_datetime_str = now_datetime.strftime("%d/%m/%Y %H:%M:%S")

		av_dict = {}
		for i in range(len(plant_details)):
			id = "av{:02}".format(i+1)
			h,w = plant_details[i]
			cond, s_datetime = db.get_condition_days(id)
			duration_in_day = (now_datetime - s_datetime).days
			av_dict = {
				'condition': "{0} since {1} days".format(cond, duration_in_day),
				'datetime': now_datetime_str,
				'height': h,
				'width': w
			}
			db.update_aloe_vera(id, av_dict)

			
		return plant_details
