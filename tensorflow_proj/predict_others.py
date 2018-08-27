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
        
        #PREPROCESSING: READ AND RESIZE IMAGE USING OPENCV
# Reading the image using OpenCV
image = cv2.imread(filename)

# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)     #Convert to RGB

#The input to the network is of shape [None, image_size, image_size, num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size, image_size, num_channels)

#Now, restore saved trained model
sess = tf.Session()

#Step 1: Recreate the network graph
saver = tf.train.import_meta_graph('dogs-cats-model.meta')

#Step 2: Now load the weights saved using the restore method
saver.restore(sess, tf.train.latest_checkpoint('./'))

#Accessing the default graph which we have restored
graph = tf.get_default_graph()

#Now, let's get hold of he operation that we can process to get the output.
#In the original network, y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

#Feed image to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, len(os.listdir('training_data'))))


#Create the feed_dict required to be fed to calculate new y_pred
feed_dict_testing = {x: x_batch, y_true:y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)

#result is of this format: [probability_of_sunflower]
print(result) 

