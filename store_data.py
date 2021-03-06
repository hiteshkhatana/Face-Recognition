import cv2 , os
import pandas as pd 
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img



print("************************************ Build your dataset ***************************************")
face_cascade = cv2.CascadeClassifier("haarcascade.xml")
quit = True
while quit == True:

	name = input("Enter your Name : ")

	with open("testfile.txt" ,"w") as f:
		f.write("\n")
		f.write(name)


	path = r"data\train"
	i_path = os.path.join(path , name)

	os.mkdir(i_path)

	video = cv2.VideoCapture(0)
	j = 40
	while j > 0:
		check ,frame = video.read()

		faces = face_cascade.detectMultiScale(frame , scaleFactor = 1.05 , minNeighbors = 5)

		for x,y,w,h in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
			font = cv2.FONT_HERSHEY_PLAIN
			cv2.putText(frame, "(%d photos remaining)"%j,(x,y-20),font,1,(255,255,255) )
			cropped = frame[y:y+h , x:x+w]

		key = cv2.waitKey(1)
		frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
				
		save_img("%s%s%d.jpg"%(i_path,name,j) ,cropped , target_size = (50,50) )
		j = j-1


		cv2.imshow('capturing',frame )
		if key == ord('q'):
			video.release()
			cv2.destroyAllWindows()

	choice = input("Do you want to add more Identity : if yes type 'y' ").lower()

	if choice == 'y':
		continue
	else:
		quit = False

