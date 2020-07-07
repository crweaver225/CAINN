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

my_data = np.loadtxt('mnist_train.csv', delimiter=',')
train_labels = my_data[:,:1]
train_data = my_data[:,1:]
train_data = train_data.astype('float32')
train_data /= 255

lr = np.arange(10)

test_data = train_data[40000:]
test_labels = train_labels[40000:]
test_labels_one_hot = (lr==test_labels).astype(np.float)

train_data = train_data[:40000].tolist()
train_labels = train_labels[:40000].tolist()

train_labels_one_hot = (lr==train_labels).astype(np.float)

nn = Neural_Network()

nn.add_input_layer(784)
nn.add_fully_connected_layer(80, Activation_Function.Sigmoid)
nn.add_fully_connected_layer(40, Activation_Function.Relu)
nn.add_output_layer(10, Activation_Function.SoftMax)

nn.build()
nn.setLearningRate(0.25)
nn.set_filepath('mnist-net.json')
nn.print_loss_every_iterations(1)

nn.train(train_data, train_labels_one_hot, 64, 5, Loss.CrossEntropy)
nn.save_network()

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
