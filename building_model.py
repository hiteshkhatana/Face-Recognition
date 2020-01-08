#import libraries
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv(r"C:\Users\Hitesh khatana\Desktop\python programs\data3.csv")

columns_to_drop = ['Unnamed: 0','Identity']
factors = data.drop(columns_to_drop , axis = 1)
identity = data['Identity']


encoder = preprocessing.LabelEncoder()
#l1 = list(encoder.classes)
#print(l1)
identity = encoder.fit_transform(identity)

factors = np.array(factors)
factors = factors.flatten()
factors = factors.reshape(int(factors.shape[0]/3072),32,32,3)
x_train,x_test,y_train,y_test = train_test_split(factors,identity,test_size=.2, random_state = 6)

def model(X_train,X_test,y_train,y_test):


#normalizing the data
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train = X_train / 255.0
	X_test = X_test / 255.0

#one hot encoding the target outputs
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	class_num = y_test.shape[1]

#creating model
	model = Sequential()

#adding layers
	model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(256, kernel_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(128, kernel_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(class_num))
	model.add(Activation('softmax'))

	epochs = 10
	optimizer = 'adam'

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	model.fit(X_train,y_train, validation_data=(X_test, y_test), epochs=epochs )

	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Your model is built with Accuracy: %.2f%%" % (scores[1]*100))

#to save model

	from keras.models import load_model 
	model.save('project1_model.h5')


choice = input("Do you want to build model now : for yes -> 'y' :- ").lower()
if choice == 'y':
	model(x_train,x_test,y_train,y_test)

