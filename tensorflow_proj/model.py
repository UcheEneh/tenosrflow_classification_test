classes = ['dogs', 'cats']
num_classes = len(classes)

train_path = 'training_data'

#validation split
validation_size = 0.2

#batch_size
batch_size = 16

data =dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

#dataset is a class created to read the input data. 
#This is a simple python code that reads images from the provided training and testing data folders.
