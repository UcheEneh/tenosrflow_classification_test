"""Builds the CIFAR-10 network.
    
    Summary of available functions:
    # Compute input images and labels for training. If you would like to run evaluations, use inputs() instead.
    inputs, labels = distorted_inputs()
    
    # Compute inference on the model inputs to make a prediction.
    predictions = inference(inputs)
    
    # Compute the total loss of the prediction with respect to the labels.
    loss = loss(predictions, labels)
    
    # Create a graph to run one step of training with respect to the loss.
    train_op = train(loss, global_step)
"""

# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

#Basic model params
tf.app.flags.DEFINE_integer('')

