import os
import time
#import re
import maplotlib
import matplotlib.pyplot as plt 
import sys
import cv2
import numpy as np 
import random
import math

# root directory 
ROOT_DIR= os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.model import modellib
from mrcnn import visualize
from mrcnn.model import log

# importing mscoco configurations
import coco

#saving logs and trained models
COCO_MODEL_DIR = os.path.join(ROOT_DIR,"logs")

#COCO Weights
COCO_WEIGHTS_FILE=os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
if not os.path.exists(COCO_WEIGHTS_FILE):
	utils.download_trained_weights(COCO_WEIGHTS_FILE)

#Directory of the images
IMAGE_DIR=os.path.join(ROOT_DIR,"images")

# We will be running on inference mode here
class _config_(coco.CocoConfig):
	# since we are running on config mode here we will run one picture at a time
	GPU_COUNT=1
	IMAGES_PER_GPU=1
# Call the above function to implement these assigned features
config=_config_()

# inferencing it
model=modellib.MaskRCNN(mode="inference",model_dir=COCO_MODEL_DIR,config=config)
# Loading weights
model.load_weights(path=COCO_WEIGHTS_FILE,by_name=True)

class_names=['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
# Taking an image to test upon
image=skimage.io.imread(IMAGE_DIR,test.jpeg)
results=model.detect([image],verbose=1)
r=results[0]
visualize.display_instances(image,r['rois'],r['masks'],r['class_id'],class_names,r['scores'])