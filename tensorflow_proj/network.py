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
from tensorflow import import set_ran






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
    
    #stride: [batch_stride x_stride y_stride depth_stride] 
    #batch_stride and depth_stride are always 1 as you don’t want to skip images in your batch and along the depth. 
    
    #Add biases after convolution 
    layer += biases 
    
    #We shall use max-pooling
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1], #shape of pooling
                           strides=[1, 2, 2, 1], 
                           padding='SAME')
    
    #Output of poolinf ios fed to ReLU 
    layer = tf.nn.relu(layer)
    
    return layer

def create_flatten_layer(layer):
    
    #The Output of a convolutional layer is a multi-dimensional Tensor. 
    #We want to convert this into a one-dimensional tensor. This is done in the Flattening layer. 
    #We simply use the reshape operation to create a single dimensional tensor as defined below:
    
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
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