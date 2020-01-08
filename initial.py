import cv2 
from second import Identify
import numpy as np
import pandas as pd
from sklearn import svm
import tensorflow as tf
from tensorflow import keras



face_cascade = cv2.CascadeClassifier("haarcascade.xml")

video = cv2.VideoCapture(0)
check = True
while check == True:
	c ,frame = video.read()

	faces = face_cascade.detectMultiScale(frame , scaleFactor = 1.05 , minNeighbors = 5)

	for x,y,w,h in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
		font = cv2.FONT_HERSHEY_PLAIN
		cv2.putText(frame, "Press C to capture",(x,y-20),font,1,(255,255,255) )
		cropped = frame[y:y+h , x:x+w]

	key = cv2.waitKey(1)
	if key == ord('c'):
		check = False
		video.release()
		cv2.destroyAllWindows()
		ans = Identify(cropped)
	cv2.imshow('capturing',frame )

			



