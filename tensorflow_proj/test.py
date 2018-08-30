"""
import tensorflow as tf

hello = tf.constant('Hello Tensorflow!')

sess = tf.Session()
print(sess.run(hello))
"""

import os

classes = os.listdir('training_data')
num_classes = len(classes)

print(classes)
print("") 
print(num_classes)