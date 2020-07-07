# C.A.I.N.N.

## Udacity Capstone Project
This is my Udacity C++ capstone project. I choose option 1. This project is its own Neural Network builder inspired by the deep learning library Keras. The project, while written in C++, has a python wrapper around it so that users can access the project as an library in a python project. Please see the About section below for more information on the exact purpose of this project. 

### Requirements to run
- Python 3.0 or higher
- Cmake (https://cmake.org/install/)

This project must be compiled to rub by following these steps. 

- Create a "build" directory within the CAINN directory and CD into it
- within the build directory, run cmake ..
- within the build directory, run make

After that, you should be compiled and ready for run. This project does utilize one third party dependency, nlohmann/json. Cmake will automatically download and install this dependency when running the steps above in a way that is cross-platform compliant.

### Testing the project
I have provided a couple of different tests to demonstrate how the deep learning project works. The main one to focus on is in the file Udacity-temp.py. This file will initialize my neural network project and train it to convert temperatures from celsius to fahrenheit. The file is extensively commented for details on how this works.By running "python Udacity-temp.py" in a CLI, you will see the network train and once training is complete, it will then be feed some new temperatures in celsius and output its converted answer. The output will let you see both the correct answer and what the network says the answer is for comparison. Please be aware, with neural network like this, the desire is to get very close to the correct answer, but rarely will a neural network be 100% correct. 

My project will also save the network to disk. In order to test this part, you can also run the file "python Udacity-temp-load.py". This will load the network from disk and execute a conversion of provided celcius temperatures. Again, this will output both the correct answer and the networks answer for comparison. 

An optional test you can run if you want, is the Udacity-mnist.py file which takes a dataset of images that are themselves handwritten digits and trains the network to recognize what number each image has (more information provided here: https://en.wikipedia.org/wiki/MNIST_database). This file is again extensively commented on for more details, at the end of training, the file will output how accurate the network is by testing itself against 10,000 images that it has not previously seen during training. Depending on your computer, this training can take up to 5 minutes, but should have an accuracy above 80%. In order to run this file though, you must have the following dependencies on your machine:
- numpy
- PIL
- You will also need to download the mnist dataset. You can do so by select this link: http://www.pjreddie.com/media/files/mnist_train.csv. Please be sure to move this dataset into the CAINN directory.

### Rubric requirements
#### The project reads data from a file and process the data, or the program writes data to a file. 
File Network_Saver.cpp contains two methods. One saves a file in JSON format, the other reads the json format and converts back into a neural network object.
#### Inheritance hierarchies are logical. Composition is used instead of inheritance when appropriate. Abstract classes are composed of pure virutal functions. Override functions are specified.
The Input_layer, fully_connected_layer, output_layer classes all inherit from the Neural_layer class which is the only class the main Neural_Network class interacts with. All these children classes implement all pure virtual functions and override functions when unique implementation is required.
#### Templates generalize functions in the project
In Tensor.cpp, the matmul function starting at line 101 accepts a templated function that it later executes for data manipulation in a different function on line 96.
#### The project follows the Rule of 5.
In Tensor.cpp, the class implements the rule of 5.
#### The project uses smart pointers. The project does not use raw pointers.
Smart pointers are almost exclusively used. This can be seen most extensively in Neural_Layer.h lines 16 through 20. I am using a raw pointer for the Tensor class (Tensor.h line 22). But this is because operations on this array need to be as fast as possible in order to cut the train time on the program down. Even the slightest hit to performance that accessing the raw pointer via a smart pointer can have will drastically increase the train time.
#### The project uses multithreading
In Tensor.cpp, starting at line 104, the project spins up threads to reduce calculation time on this very expensive function. I have to use this one carefully as all threads are operating on the same array, but with proper dimensionality calculations, this can be done and it cuts down run time significantly.

## About
This project is designed to be a neural network trainer inspired by Tensorflow and Keras. The project exists as an external library coded in c++ to be performant and efficient. CAINN will do the following:
- Graph a neural network to user specifications
- Train the neural network using stochastic gradient descent
- Execute the neural network
- Save the neural network in JSON format

## Example
The library can be accessed using Python 2.7 or higher. Just import the library like so:
```
from Neural_Network import Neural_Network
from Neural_Network import Activation_Function
from Neural_Network import Loss
```

Once imported a network can be setup to train as simply as this:
```
nn = Neural_Network()
nn.add_input_layer(1)
nn.add_fully_connected_layer(20, Activation_Function.Sigmoid)
nn.add_fully_connected_layer(40, Activation_Function.Relu)
nn.add_fully_connected_layer(10, Activation_Function.Relu)
nn.add_output_layer(1, Activation_Function.Relu)
nn.build()
nn.setLearningRate(0.01)
nn.train(train_data,train_results, 10 ,500, Loss.MSE)
```
Training the network is then as simple as calling this function. The four parameters are:
- the training examples
- the training targets
- the mini batch size
- how many iterations
- the loss function
```
nn.train(train_data,train_results, 10 ,500, Loss.MSE)
```

## Other features
by setting the neural networks file path, you can either call on the network to be saved at anytime
```
nn.set_filepath('temp-net.json')
nn.save_network()
```
or you can tell the network to save after any iteration where the loss is less than previous iterations.
```
nn.save_best_automatically(True)
```
In order to track the progress of your training, you can set how often the network outputs the loss of any given iteration
```
nn.print_loss_every_iterations(100)
```
You can at anytime load an old network by calling this function with the correct file path
```
nn.load_network('temp-net.json')
```
After loading or training a network, you can then execute your network at any time with input of the same dimension as was used during the initial building of the network.
```
result = nn.execute([5.0])
```

## Supported types

### This project currently supports three types of layers
- Input layer
    - add_input_layer()
- Fully connected layer
    - add_fully_connected_layer()
- Output layer
    - add_output_layer()

### This project currently supports five types of activation functions
- Sigmoid
- Relu
- Leaky Relu
- Pass
- Softmax

### This project currently supports three kinds of loss fuctions
- Mean squared error
    Loss.MSE
- Absolute error
    Loss.ASE
- Cross Entropy
    Loss.CrossEntropy

## Highlights
- Utilizes multi-threading for speed and efficiency.
- A clean and simple pythonic interface
- Memory efficient 
- Modern C++ memory management practices 
- Stores networks in JSON which is widely adaptable to most enviornments for use
