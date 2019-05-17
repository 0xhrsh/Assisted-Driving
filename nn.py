import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import cv2
import math
import json
from pprint import pprint
from keras.preprocessing.image import ImageDataGenerator
import glob
from keras.utils import to_categorical


# This Code is used to train the NN for the first time
# using this the NN can be trained on a video file, the time intervals are calculated
# and are not advised to be tempered with without proper calculations

count = 0
videoFile = "c4e926af-43c5f8d3.mov"  # Rename the video file accordingly
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(5) #frame rate
print(frameRate)
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()


f=open('c4e926af-43c5f8d3.json','r') # Rename the File including timestamp marked data accordingly
x=f.read()
locations =dict(course=f.read(),timestamp=f.read(),speed=f.read())
gyro=dict(z=f.read(),y=f.read(),x=f.read(),timestamp=f.read())
data_location=[]
i=0
data = json.loads(x)
i=0
data_gyro=np.ones(41)
j=0
while i<len(data["gyro"]):
	
	temp=data["gyro"][i]["y"]
	if temp>0.1:
		data_gyro[j]=1
	elif temp<-0.1:
		data_gyro[j]=-1
	else:
		data_gyro[j]=0	
	i+=50
	j+=1

images=list(glob.iglob('*.jpg'))
n=len(images)
X=np.zeros((n,200,66,3))
i=0
for image in images:
	img=cv2.imread(image)
	img=cv2.resize(img,(66,200))
	img=np.expand_dims(img, axis=0)
	X[i]=img
	i+=1
X=np.array(X/255)


data_gyro=to_categorical(data_gyro)


# load json and create model
# json_file = open('weights.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# load weights into new model

#model.load_weights("weights.h5")
#print("Loaded model from disk")


# Defining Model here, the model is inspired from NVIDIA's End to End deep learning paper
# read the paper to understand the intention
# It is a sequential model with 5 convulational layers and a NN with 4 dense layers

model=Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(200,66,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(36,(5,5),activation="relu",padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(48,(5,5),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(1164,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(10,activation='tanh'))
model.add(Dense(2,activation='sigmoid'))
print('Compiling the model...')
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X,data_gyro,epochs=30,batch_size=15)
print("Predicting Scores")
scores=model.evaluate(X,data_gyro)
print(scores[1]*100)



# serialize model to JSON
model_json = model.to_json()
with open("weights.json", "w+") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("weights.h5")
print("Saved model to disk")
 
# later...
 

 
# # evaluate loaded model on test data
# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# scores=loaded_model.evaluate(X,data_gyro)
# #score = loaded_model.evaluate(X, data_gyro, verbose=0)
# print(scores[1]*100)