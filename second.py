import cv2, time, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def Identify(gray):

	model = load_model('project1_model.h5')

	gray = cv2.resize(gray,(0,0),fx=0.5,fy=0.5)
	save_img(r"C:\Users\Hitesh khatana\Desktop\python programs\hitesh\output.jpg" ,gray , target_size = (32,32) )
	load = image.load_img(r"C:\Users\Hitesh khatana\Desktop\python programs\hitesh\output.jpg" , target_size = (32,32))
	arr = img_to_array(load)
	arr = arr.flatten()
	arr = arr.reshape(32,32,3)
	gray = array_to_img(arr)
	test_image =image.img_to_array(gray) 
	test_image =np.expand_dims(test_image, axis =0) 
	result = model.predict(test_image)
	print(result)

	with open(r"C:\Users\Hitesh khatana\Desktop\python programs\hitesh\testfile.txt" ,"r") as f:
		lines = f.readlines()
	li = list(lines)
	i = np.argmax(result[0])
	print(li[i])









	

	



	





		



