"""Utilities for parsing PTB text files."""

""" https://blog.csdn.net/u012436149/article/details/52828782 """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
#import 

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()      #.read(): Returns the contents of a file as a string.
                                                                #split(): e.g returns: ['hello', 'world', 'a', 'b', 'c']
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()
        
def _build_vocab(filename):
    data = _read_words(filename)
    
    counter = collections.Counter(data)     #output a dictionary: key is word, value is the number of times this word appears
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))  #counter.items() will return a list of tuples, 
                                                #tuple is (key, value), in descending order of value, in ascending order of keys
                                                
    words, _ = list(zip(*count_pairs))  #put it in a list    #e.g: { ("hello":12), ("world":11), ...
                                        #key: (words: "hello"), value: (_: 12)
    word_to_id = dict(zip(words, range(len(words)))) #create a dict in (value descending, key ascending), using id btw (0-len(words))
    #e.g: { ("hello", 0), ("world", 1), ...
    # dic: {key: word, value: id}     
    #hello appears 12 times and is the most appearing, so it gets id of 0,...
    
    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    
    return [word_to_id[word] for word in data if word in word_to_id]    #return only words from data in word_to_id dict

def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".
    
      Reads PTB text files, converts strings to integer ids,
      and performs mini-batching of the inputs.
      
      The PTB dataset comes from Tomas Mikolov's webpage:
      http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
      
      Args:
        data_path: string path to the directory where simple-examples.tgz has
          been extracted.
      
      Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
    """
    
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    
    word_to_id = _build_vocab(train_path)   #for full vocabulary
    train_data = _file_to_word_ids(train_path, word_to_id)      #remember: it returns only words in the word_to_id (full vocab)
                                                                #basically: train_data has same length as full vocab (word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    
    return train_data, valid_data, test_data, vocabulary

def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.
        
        This chunks up raw_data into batches of examples and returns Tensors that
        are drawn from these batches.
        
        Args:
          raw_data: one of the raw data outputs from ptb_raw_data.
          batch_size: int, the batch size.
          num_steps: int, the number of unrolls.
          name: the name of this operation (optional).
        
        Returns:
          A pair of Tensors, each shaped [batch_size, num_steps]. The second element
          of the tuple is the same data time-shifted to the right by one.
        
        Raises:
          tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.    
    """
    
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        #raw_data: dict of tuples for a dataset e.g. train_data
        #e.g: { ("hello", 0), ("world", 1), ...
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])
        #note: (batch_size * batch_len) was used instead of data_len incase of floating points
        # i.e. in case there are less values in data_len than in (batch_size * batch_len)
        
        epoch_size = (batch_len - 1) // num_steps
        #num_steps = num of rolls the batches would be inputed: (basically, number of inputs for the lstm)
        #NOT SURE EXACTLY. CHECK LATER
        
        
        
    
    
      