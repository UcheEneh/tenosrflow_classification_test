"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    
    """Reads and parses examples from CIFAR10 data files.
    
      Recommendation: if you want N-way read parallelism, call this function N times.  This will give you N independent Readers reading different
      files & positions within those files, which will give better mixing of examples.
      
      Args:
        filename_queue: A queue of strings with the filenames to read from.
      
      Returns:
        An object representing a single example, with the following fields:
          height: number of rows in the result (32)
          width: number of columns in the result (32)
          depth: number of color channels in the result (3)
          key: a scalar string Tensor describing the filename & record number
            for this example.
          label: an int32 Tensor with the label in the range 0..9.
          uint8image: a [height, width, depth] uint8 Tensor with the image data
      """
    
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    
    # Every record consists of a label followed by the image, with a fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes
    
    
    # Read a record, getting filenames from the filename_queue. 
    # No header or footer in the CIFAR-10 format, so we leave header_bytes and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
    result.key, value = reader.read(filename_queue)
    
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)
    
    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)    #input, start, end, cast to int32
    
    # The remaining bytes after the label represent the image, which we reshape from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),    #input, start, end
                                                            [result.depth, result.height, result.width])    #cast to this
    
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    
    return result


def _generate_image_and_label_batch(iimage, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.
    
    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain in the queue that provides batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    
    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    
    # Create a queue that shuffles the examples, and then read 'batch_size' images + labels from the example queue.
    