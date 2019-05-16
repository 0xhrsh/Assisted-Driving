import cv2
import math
import os
import numpy as np
import time
def track(img):
	n=3
	var=(n//2) +1
	test=np.ones([n,n])
	black=np.zeros([n,n])
	X=0
	Y=0
	wid=len(img[0])
	hig=len(img)
	i=0
	count=0
	while i< wid-n+1:
		j=0
		while j<hig-n+1:
			#compare=
			if (np.dot(test,img[i:i+n,j:j+n])==black).all():
				X+=i+var
				Y+=j+var
				count+=1
			j+=1
		i+=1
	if count==0:
		return None
	else:
		return([X//count,Y//count])

def filter_face(img,gray_img,face):

	return (img[face[0]:face[0]+int(6*face[2]/8),face[1]:face[1]+face[3]],
			gray_img[face[0]:face[0]+int(6*face[2]/8),face[1]:face[1]+face[3]])



def remove_brows(img):
	hig=len(img)
	wid=len(img[0])
	return img[int(0.12*hig):,int(0.6*wid):int(0.94*wid)]

def face_bb(img,face):
	cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(123,20,12),2)


def draw(eyes,images):
	#Yavg=int((eyes[0][1]+eyes[1][1])/2)
	#print(Yavg)
	n=len(eyes)
	if n < 2:
		print("machine")
	else:
		print("human")
	for i in range(0,n):
		cv2.circle(images[i],(eyes[i][0],eyes[i][1]),5,(0,0,150),-1)

def refine(img,TH):
	img =cv2.medianBlur(img, 5)
	
	_,img = cv2.threshold(img, TH,255, cv2.THRESH_BINARY)
	img =cv2.erode(img,None,iterations=2)
	img =cv2.dilate(img,None,iterations=4)
	return img

# face cascade
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#cascade for the eye
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
#cascade is the group of filters that are applied on the given image

cap=cv2.VideoCapture(0)
cv2.namedWindow("Image")
#cv2.createTrackbar('Threshold','image',0,255,None)
Threshold=69
while True:
	_,image=cap.read()
	# Reading the image
	#image=cv2.imread("HARSH.jpg")
	#image=cv2.imread("test_4.jpg")
	gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # will convert the image to greyscale (obviously)
	# this tells the faces in the image, first argument is the gray scale image,
	# second is the reduction in size for each frame/ image and third tells the min neighbours each rectangle should
	# have to retain it
	faces=face_cascade.detectMultiScale(gray_image,1.3,5)
	#this draws the rectangle aroung the face
	gray_faces=[]
	face_img=[]
	eyes_cod=[]
	eye_images=[]
	for face in faces:
		face_bb(image,face)
		face_img,gray_face=filter_face(image,gray_image,face)
		eyes=eye_cascade.detectMultiScale(face_img,1.3,5)
		i=0
		for eye in eyes:
			cv2.rectangle(face_img,(eye[0],eye[1]),(eye[0]+eye[2],eye[1]+eye[3]),(0,255,0),1)
			eye_image=face_img[eye[1]:eye[1]+int(eye[3]),eye[0]:eye[0]+int(eye[2])]
			gray_eye_image=gray_face[eye[1]:eye[1]+int(eye[3]),eye[0]:eye[0]+int(eye[2])]
			gray_eye_image =refine(gray_eye_image,Threshold)
		
			# Uncomment the following part to remove the influence of eyebrows  (BUGS PRESENT!!!)

			#browless_gray_eye_image=remove_brows(gray_eye_image)
			#eyes_cod.append(track(browless_gray_eye_image))
		
			eyes_cod.append(track(gray_eye_image))
			eye_images.append(eye_image)
		draw(eyes_cod,eye_images)
		cv2.imshow("Image",image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()