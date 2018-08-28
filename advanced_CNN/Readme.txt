CIFAR-10 classification is a common benchmark problem in machine learning. 
The problem is to classify RGB 32x32 pixel images across 10 categories:

"airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck."


Goal:
The goal of this tutorial is to build a relatively small convolutional neural network (CNN) for recognizing images

Model Architecture:
The model follows the architecture AlexNet

This model achieves a peak performance of about 86% accuracy within a few hours of training time on a GPU. 
It consists of 1,068,298 learnable parameters and requires about 19.5M multiply-add operations to compute inference on a single image.


Code Organization:

File                            Purpose
------------------------------------------------------------------------------------------
cifar10_input.py                Reads the native CIFAR-10 binary file format.
cifar10.py                      Builds the CIFAR-10 model.
cifar10_train.py                Trains a CIFAR-10 model on a CPU or GPU.
cifar10_multi_gpu_train.py      Trains a CIFAR-10 model on multiple GPUs.
cifar10_eval.py                 Evaluates the predictive performance of a CIFAR-10 model.


CIFAR-10 Model:

1. Model inputs: inputs() and distorted_inputs() add operations that read and preprocess CIFAR images for evaluation and training, respectively.
2. Model prediction: inference() adds operations that perform inference, i.e. classification, on supplied images.
3. Model training: loss() and train() add operations that compute the loss, gradients, variable updates and visualization summaries.


						Model Inputs
Reading input:
To read binary files in which each record is a fixed number of bytes, use tf.FixedLengthRecordReader with the tf.decode_raw operation. 
The decode_raw op converts from a string to a uint8 tensor.

For example, the CIFAR-10 dataset uses a file format where each record is represented using a fixed number of bytes: 
	- 1 byte for the label followed by 3072 bytes of image data. 
Once you have a uint8 tensor, standard operations can slice out each piece and reformat as needed