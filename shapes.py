import os 
import sys
import re
import time
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import math
import random

#root directory of the project
ROOT_DIR=os.path.abspath("../../")
#import mask rcnn
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.model import modellib
from mrcnn import visualize
from mrcnn.model import log


#directory to save logs and trained model
MODEL_DIR=os.path.join(ROOT_DIR,"logs")
#Local path to trained weights file
COCO_MODEL_PATH=os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
#download COCO trained weights from releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)

class ShapesConfig(Config):
	#configuration for training on the toy shapes dataset
#give the config a name
	NAME="shapes"
#no of gpu is 1 and images per gpu is 8
    GPU_COUNT=1
    IMAGES_PER_GPU=8
#give size to images
    IMAGE_MAX_DIM=128
    IMAGE_MIN_DIM=128
#setting the anchor scales
    RPN_ANCHOR_SCALES=(8,16,32,64,128)
#train ROIS per image
    TRAIN_ROIS_PER_IMAGE=32
#using a small epoch
    STEPS_PER_EPOCH=100
#use small validation steps since the epoch is small	
    VALIDATION_STEPS=5
#making an object for config class
config=ShapesConfig()
#for displaying the configuration
config.display()
class ShapesDataset(utils.Dataset):
	#regenerates the shapes synthetic dataset.the dataset consists of simple triangles, circles and rectangles placed randomly on a blank surface
  
  	def load_shapes(self,count,width,height):
	  		# generate classes
    	 	self.add_class("shapes",1,"square")
     		self.add_class("shapes",2,"circle")
     		self.add_class("shapes",3,"triangle")

    	 	#taking random height and random width and random background color to create a circle/triangle/rectangle, you guessed it, RANDOMLY!

     		for i in range(count):
     			bg_color,shapes=self.random_image(height,width)
     			self.add_image("shapes",imageid=i,path=None.shapes=shapes,color=color,height=height,width=width)

	def load_image(self,image_id)
		  # creates an image for shapes to be placed in
   		info=self.info[image_id]
   		bg_color=np.array(info['bg_color']).reshape([1,1,3])
   		image=np.ones([info['height'],info['width'],3],dtype=uint8)
   		image=image*bg_color.astype(np.uint8)
  		for shape,color,dims in info['shapes']:
   			# images are added to the blank image created by the given funtion
   			image=self.draw_shape(image,shape,dims,color)
   		return image # image is the final image (an image the shapes are embedded in)

 	def image_reference(self,image_id):
 		info=self.info[image_id]
 		if info["source"]=="shapes":
 			return info["shapes"]
 		else:
 			super(self._class_)image.reference(self,image_id)

	def load_mask(self,image_id):
  		# creates the mask that is the correct one
  		info=self.image_info[image_id]
  		shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),shape, dims, 1)
        
        # Handle occlusions (So that the masks don't overlap)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

  	def draw_shape(self,image,shape,dims,color):
  		#draw the shape from the given specifications
  		x,y,s=dims
  		if shape=='square'
  	  		cv2.rectangle(image,(x-s,y-s),(x+s,y+s),color,-1)
  	  	elif shape=="circle":
  	  		cv2.circle(image,(x,y),s,color.-1)
  	  	elif shape=="triangle":
  	  		points=np.array([[(x,y-s),(x-s/math.sin(math.radians(60)),y+s),(x+s/math.sin(math.radians(60)).y+s)]],dtype=np.uint.32)
  	  		cv2.fillPoly(image,points,color)
    	return image
  
  	def random_shape(self,height,width):
  		#generate random specification of the shapes that lie within the 
  		#given height and width boundaries
  		#also determines the shape of the shape
  		shape=random.choice(["square","circle","triangle"])
  		color=tuple([random.randint(0,255) for _ in range(3)])
  	 	buffer=20
    	y=random.randint(buffer,height-buffer-1)
   		x=random.randint(buffer,width-buffer-1)
    	s=random.randint(buffer,height//4)
    	return color,shape,(x,y,s)

  	def random_image(self,height,width):
  		#creates random specifications of an image with multiple shapes
  		#returns the background color of the image
  		bg_color=np.array([random.randint(0,255) for _in range(3)])
  		shapes=[]
  		boxes=[]
  	  	N = random.randint(1, 4)
  	  	# decides the number of the shapes in the image (yes....randomly)
        for _ in range(N):
        	shape, color, dims = self.random_shape(height, width)
        	shapes.append((shape, color, dims))
            x, y, s = dims
           	boxes.append([y-s, x-s, y+s, x+s])\

        # Apply non-max suppression with 0.3 threshold to avoid
        # shapes covering each other (occlusion)
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

# Preparing Different Datasets

#Forming an object for the Dataset
dataset_train=ShapesDataset()
# forming dataset for the algo to train on
dataset_train.load_shapes(500,config,IMAGE_SHAPE[0],config.IMAGE_SHAPE[1])
dataset_train.prepare()
# Preparing dataset for validation as well, just like above (we are not here to spoonfeed you)
validation_train=ShapesDataset()
validation_train.load_shapes(500,config,IMAGE_SHAPE[0],config.IMAGE_SHAPE[1])
validation_train.prepare()
# Displaying the images with the masks
image_ids=np.random.choice(dataset_train.image_ids,4)
for image_id in image_ids:
	image=dataset_train.load_image(image_id)
	mask,class_ids=dataset_train.load_mask(image_id)
	visualize.display_top_masks(image,mask,class_ids,dataset_train.class_names)


# Create model in training mode

model=modellib.MaskRCNN(mode="training",config=config,model_dir=MODEL_DIR)
#which weights to start with
init_with = "coco"  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
#Training the model
model.train(dataset_train,dataset_val,learning_rate=config.LEARNING_RATE,epochs=1,layers='heads')
# Fine Tuning the model
model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE / 10,epochs=2,layers="all")

class InferenceConfig(ShapesConfig):
	GPU_COUNT=1 # We change the Infrastucture now because we are no longer training hence the needs are different
	IMAGES_PER_GPU=1
inference_config=InferenceConfig()
model=modellib.MaskRCNN(mode="inference",config=inference_config,model_dir=MODEL_DIR)
#get path to saved weights
model_path=model.find_last()
#load trained weights
print("Loading weights from",model_path)
model.load_weights(model_path,by_name=True)
#test on a random image
image_id=random.choice(dataset.val_image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,image_id, use_mini_mask=False)

# Save The logs
log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))
#testing with output giving also the class_ids
results=model.detect([original_image],verbose=1)
r=results[0]
visualize.display_instances(original_image,r['rois'],r['masks'],r['cass_ids'],
	dataset_val.class_names,r['scores'],ax=get_ax())
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))
