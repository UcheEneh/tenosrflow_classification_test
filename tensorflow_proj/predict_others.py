import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

#Used to predict the class for new dataset

# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path

image_size=128
num_channels=3
images = []

        #READ AND RESIZE IMAGE USING OPENCV
# Reading the image using OpenCV
image = cv2.imread(filename)

# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 

#The input to the network is of shape [None, image_size, image_size, num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size, image_size, num_channels)

#Now, restore saved trained model
sess = tf.Session()

