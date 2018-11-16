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
                                                
    words, _ = list(zip(*count_pairs))  #put it in a list
    word_to_id = dict(zip(words, range(len(words)))) #create a dict in (value descending, key ascending), using id btw (0-len(words))
    #e.g: { ((hello:12), 0), ((world:11), 1), ...
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
  
  