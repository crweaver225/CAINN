# Required imports to run this code. Some of these dependencies may require installing via pip
from enum import Enum
import ctypes
import sys
import gc
import operator
from Neural_Network import Neural_Network
from Neural_Network import Activation_Function
from Neural_Network import Loss
import numpy as np
from numpy import genfromtxt
from PIL import Image

# Load all the image data and convert into number between 1 and zero
my_data = np.loadtxt('mnist_train.csv', delimiter=',')
train_labels = my_data[:,:1]
train_data = my_data[:,1:]
train_data = train_data.astype('float32')
train_data /= 255

lr = np.arange(10)

# There are 60,000 images in the data set. We will test on the last 10,000 and train on the first 50,000
test_data = train_data[50000:]
test_labels = train_labels[50000:]
test_labels_one_hot = (lr==test_labels).astype(np.float)

train_data = train_data[:50000].tolist()
train_labels = train_labels[:50000].tolist()

train_labels_one_hot = (lr==train_labels).astype(np.float)

# Initialize our neural network and set the architecture for this task
nn = Neural_Network()
nn.add_input_layer(784)
nn.add_fully_connected_layer(80, Activation_Function.Sigmoid)
nn.add_fully_connected_layer(40, Activation_Function.Relu)
nn.add_output_layer(10, Activation_Function.SoftMax)

# Build the network, set the learning rate and tell the network where to save
nn.build()
nn.setLearningRate(0.1)
nn.set_filepath('mnist-net.json')
nn.print_loss_every_iterations(1)

# Train the network with mini batch sizes of 64, for 5 iterations using cross entropy as our loss function
nn.train(train_data, train_labels_one_hot, 64, 5, Loss.CrossEntropy)
nn.save_network()

#Once training is complete, this will one through the test set and see how accurate our network is by comparing its answers to the correct labels
index = 0
matches = 0
for data in test_data:
    output = nn.execute(test_data[index])
    outputIndex, value = max(enumerate(output), key=operator.itemgetter(1))
    if outputIndex == test_labels[index][0]:
        matches += 1
    index += 1
match_percentage = matches / index
print("percent correct: ")
print(match_percentage)

