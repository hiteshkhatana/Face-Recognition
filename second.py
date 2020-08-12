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

	model = load_model('new_model.h5')

	test_image = cv2.imread(gray)
	result = model.predict(test_image)
	print(result)

	with open("testfile.txt" ,"r") as f:
		lines = f.readlines()
	li = list(lines)
	i = np.argmax(result[0])
	print(li[i])









	

	



	





		



