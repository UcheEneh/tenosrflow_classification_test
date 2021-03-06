import dataset
import tensorflow as tf
import numpy as np

import time
from datetime import timedelta

import math
import random
import os

#Add seeds so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

                    #DEFINE PARAMETERS
batch_size = 32

#Prepare input data
classes = os.listdir('training_data')   #this folder would be created later    #[cats, dogs]
num_classes = len(classes)  #2

#20% for validation
validation_size = 0.2
img_size = 128  #image reshaped to 128x128
num_channels = 3
train_path = 'training_data'


#Load all the training and validation images and labels into memory using openCV
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

#Labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 128


                    #DEFINE METHODS FOR NETWORK
#initialize weights as normal distributions
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    #define weights and bias of filter using functions above which would be trained 
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(size=num_filters)
    
    # Creating the conv layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    #SAME means we shall 0 pad the input such a way that output x,y dimensions are same as that of input.
    #stride: [batch_stride, x_stride, y_stride, depth_stride] 
    #batch_stride and depth_stride are always 1 as you don't want to skip images in your batch and along the depth.
    
    #Add biases after convolution 
    layer += biases 
    
    #We shall use max-pooling
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1], #shape of pooling
                           strides=[1, 2, 2, 1], 
                           padding='SAME')  
    
    #Output of pooling is fed to ReLU 
    layer = tf.nn.relu(layer)
    
    return layer

def create_flatten_layer(layer):
    #The output of a conv layer is a multi-dimensional Tensor. We want to convert this into a one-dim tensor.  
    #We simply use the reshape operation to create a single dimensional tensor as defined below:
    
    #We know that the shape of the layer will be [batch_size, img_size, img_size, num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()             #note: layer is of tf datatype, not np
    
    #Number of features will be img_height*img_width*num_channels. 
    #But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()      # = [img_size * img_size * num_channels]
    
    # Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])
    
    return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    #Define trainable weights and biases for FC layer
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    
    layer = tf.matmul(input, weights) + biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer 


                        #CREATE NETWORK
layer_conv1 = create_convolutional_layer(input = x, 
                                         num_input_channels = num_channels, 
                                         conv_filter_size = filter_size_conv1, 
                                         num_filters = num_filters_conv1)

#Remember: number of filters become the new number of channels after each conv layer
layer_conv2 = create_convolutional_layer(input = layer_conv1, 
                                         num_input_channels = num_filters_conv1, 
                                         conv_filter_size = filter_size_conv2, 
                                         num_filters = num_filters_conv2)

layer_conv3 = create_convolutional_layer(input = layer_conv2, 
                                         num_input_channels = num_filters_conv2, 
                                         conv_filter_size = filter_size_conv3, 
                                         num_filters = num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input = layer_flat, 
                            num_inputs = layer_flat.get_shape()[1:4].num_elements(), 
                            num_outputs = fc_layer_size, 
                            use_relu = True)

layer_fc2 = create_fc_layer(input = layer_fc1, 
                            num_inputs = fc_layer_size, 
                            num_outputs = num_classes, 
                            use_relu = False)

#Softmax
y_pred = tf.nn.softmax(layer_fc2, name = 'y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)

sess.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=layer_fc2)

cost = tf.reduce_mean(cross_entropy)    #Computes mean of elements across dims of a tensor (and returns a single element, since axis not defined)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)   #Matrix of True for the indexes of values that are equal and False otherwise
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #tf.cast: Casts a tensor to a new type(in this case, bool to float32).

sess.run(tf.global_variables_initializer())     #why a second time????????????



def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = sess.run(accuracy, feed_dict=feed_dict_train)
    val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
    
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:>.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


    
total_iterations = 0
saver = tf.train.Saver()    #

def train(num_iteration):
    
    global total_iterations
    
    for i in range(total_iterations, total_iterations + num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
        
        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        
        feed_dict_val = {x: x_valid_batch,
                        y_true: y_valid_batch}
        
        sess.run(optimizer, feed_dict=feed_dict_tr)
        
        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = sess.run(cost, feed_dict = feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))        
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(sess, './' + 'dogs-cats-model')
            
    total_iterations += num_iteration

                #RUN NETWORK
train(num_iteration = 3000)

