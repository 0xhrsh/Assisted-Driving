import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import cv2
import math
import json
from pprint import pprint
from keras.preprocessing.image import ImageDataGenerator
import glob
from keras.utils import to_categorical


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
#loading model


count = 0
videoFile = ""
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
#print("Locations")
#pprint(data_gyro)

# train_dataset=ImageDataGenerator(rescale=1./255, horizontal_flip=False)
# #train_dataset=ImageDataGenerator
# training_set=train_dataset.flow_from_directory('images',target_size=(66,200),batch_size=16,class_mode='binary')








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
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
loaded_model.fit(X,data_gyro,epochs=10,batch_size=7)


score = loaded_model.evaluate(X, data_gyro, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



#make a prediction
ynew = loaded_model.predict_classes(X)
# show the inputs and predicted outputs
for i in range(len(X)):
	print("X=%s, Predicted=%s" % (X[i], ynew[i]))