import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import cv2
import math
import json
from pprint import pprint
from keras.preprocessing.image import ImageDataGenerator
import glob
f=open('c4e97226-7be7eef5.json','r')
x=f.read()
locations =dict(course=f.read(),timestamp=f.read(),speed=f.read())
gyro=dict(z=f.read(),y=f.read(),x=f.read(),timestamp=f.read())
data_location=[]
i=0
data = json.loads(x)
#while i<len(data["locations"]):
#	data_location.append([data["locations"][i]["speed"],data["locations"][i]["timestamp"],data["locations"][i]["course"]])
#	i+=1
i=0
data_gyro=[]
while i<len(data["gyro"]):
	data_gyro.append([data["gyro"][i]["z"],data["gyro"][i]["y"],data["gyro"][i]["x"]])
	i+=50
#print("Locations")
#pprint(data_location)

train_dataset=ImageDataGenerator(rescale=1./255, horizontal_flip=False)
#train_dataset=ImageDataGenerator
#training_set=train_dataset.flow_from_directory('ASStech/images',target_size=(66,200),batch_size=16,class_mode='binary')








# images=list(glob.iglob('*.jpg'))
# n=len(images)
# X=np.zeros((n,66,200,3))
# i=0
# for image in images:
# 	img=cv2.imread(image)
# 	#image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	img=cv2.resize(img,(200,66))
# 	print(np.shape(img))
# 	X[i]=img
# 	print(np.shape(X[i]))
# 	i+=1
# X=np.array(X/255)
# print("gyro")
# print(len(data_gyro))
model=Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(66,200,3),activation='relu'))

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
model.add(Dense(100,activation='sigmoid'))
model.add(Dropout(0.15))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(3,activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
count = 0
videoFile = "c4e97226-7be7eef5.mov"
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

#model.fit_generator(X[:],data_gyro,epochs=20,batch_size=5)
#scores=model.evaluate(X,data_gyro)
print(scores[1]*100)