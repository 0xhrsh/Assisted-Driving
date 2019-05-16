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
a='c4eb96a1-81c4450c'
count = 0
videoFile = a+'.mov'
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

images=list(glob.iglob('*.jpg'))
n=len(images)
X=np.zeros((n,200,66,3))
i=0
for image in images:
	img=cv2.imread(image)
	#image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img=cv2.resize(img,(66,200))
	#print(np.shape(img))
#	cv2.imshow("IMAGE",img)
	img=np.expand_dims(img, axis=0)
	X[i]=img
#	cv2.waitKey(0)
	#print(np.shape(X[i]))
	i+=1
X=np.array(X/255)
#print("gyro")
#print(len(data_gyro))


a=a+'.json'
f=open(a,'r')
x=f.read()
locations =dict(course=f.read(),timestamp=f.read(),speed=f.read())
gyro=dict(z=f.read(),y=f.read(),x=f.read(),timestamp=f.read())
data_location=[]
i=0
data = json.loads(x)
i=0
data_gyro=np.ones(n)
j=0
while i<len(data["gyro"]):
	
	temp=data["gyro"][i]["y"]
	if temp>0:
		data_gyro[j]=1
	if temp<0:
		data_gyro[j]=0	
	i+=50
	j+=1


data_gyro=to_categorical(data_gyro)

#load json and create model
json_file = open('weights.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

json_file = open('weights.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
loaded_model.fit(X,data_gyro,epochs=30,batch_size=15)
scores=loaded_model.evaluate(X,data_gyro)
print(scores[1]*100)

# serialize model to JSON
model_json = loaded_model.to_json()
with open("weights.json", "w+") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save_weights("weights.h5")
print("Saved model to disk")