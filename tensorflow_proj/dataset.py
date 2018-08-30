#dataset is a class created to read the input dataset. 
#This is a simple python code that reads images from the provided training and testing dataset folders.

import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(train_path, image_size, classes):    #classes = [dogs, cats]  #list
    images = []
    labels = []
    img_names = []
    cls = []
    
                    #READ AND RESIZE IMAGE USING OPENCV
    print('Going to read training images')
    for fields in classes:   #fields = cats, dogs
        
        #Understanding index
        """
        classes = ['cats', 'dogs']
        
        print "Index for dogs : ", classes.index( 'dogs' ) 
        >>> Index for dogs :  1 
        """
        
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)     #makes a list of every imagepath of class "field: i.e cat or dog" (that ends with g i.e .jpg)
        
        for fl in files:
            #Read the images and set label
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)     #convert to RGB
            
            images.append(image)    #add it to list of all images in this field (i.e cat or dog)
            label = np.zeros(len(classes))
            label[index] = 1.0      #so when we train cats, the label for cats = 1 and dogs = 0, and vice versa
            
            labels.append(label)
            flbase = os.path.basename(fl)   #???  #Return the base name of pathname fl. This is the second element of the pair returned by passing fl to the function split(). 
                                            #Note that the result of this function is different from the Unix basename program; where basename for '/foo/bar/' returns 'bar', 
                                            #the basename() function returns an empty string ('').
                                            
                                            #in this case e.g.: cat.110.jpg
                            
            img_names.append(flbase)
            cls.append(fields)
            
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    
    print(images.shape)
    print(labels.shape)
    print(img_names.shape)
    print(cls.shape)

    return images, labels, img_names, cls


class DataSet(object):
    
    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]    #gives number of images used in the batch
        
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def img_names(self):
        return self._img_names
    
    @property
    def cls(self):
        return self._cls
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_done(self):
        return self._epochs_done
    
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:   #images.shape[0] #if one epoch is completed, reset for next epoch
            #Update after each epoch
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples #_num_examples = images.shape[0] which is the total number of images in the data 

        end = self._index_in_epoch
        
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]
    
def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object):      #make class DataSets usable in this method but do nothing (pass) after initialization
        pass                    ### NOTE. DIFFERENT FROM DataSet
    
    data_sets = DataSets()
    
    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    #Shuffle data so that we don't have an array of all cats first then dogs
    #shuffle: Shuffle arrays or sparse matrices in a consistent way (so images and their specific labels, img_name and cls would still be together after shuffle)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)        
    
    if isinstance(validation_size, float):      #if validation_size is of type float
        validation_size = int(validation_size * images.shape[0])
        
    #Validation images: last 20% of images
    validation_images = images[:validation_size]    
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]
    
    #Train images: first 80% of images
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]
    
    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
    
    return data_sets
