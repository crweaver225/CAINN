from enum import Enum

#Import the Neural Network c++ library in
from Neural_Network import Neural_Network
from Neural_Network import Activation_Function
from Neural_Network import Loss

# This neural network will train to learn how to convert celsius temperatures to fahrenheit. We will pass in a list of temperatures in celsius
# and its corresponding fahrenheit conversion and have the network learn how to do the conversion itself.

# This is the celsius temperatures the network will train on. Since the network can take multiple values as inputs,
# the training data must be an array within the train_data array.
train_data = [[-10.0],[0.0],[8.0],[15.0],[22.0],[38.0],[50.0],[12.0],[19.0],[44.0],[61.0],[70.0],[11.0]]
# This is the fahrenheit answers for each each item in train_data
train_results = [[14.0],[32.0],[46.0],[59.0],[72.0],[100.0],[122.0],[53.6],[66.2],[111.2],[141.8],[158.0],[51.8]]

# After importing the Neural Network library, you can now initialize the neural network object
nn = Neural_Network()

# Before training the network, you must first tell the network what structure it should have.
# You can add either an input layer, fully connected layer, or output layer. Each layer is initialized
# with how many neurons are in the layer and the activiation function it will use
# Feel free to change this if you want, but this structure will work well for the task at hand
nn.add_input_layer(1)
nn.add_fully_connected_layer(20, Activation_Function.Sigmoid)
nn.add_fully_connected_layer(40, Activation_Function.Relu)
nn.add_fully_connected_layer(10, Activation_Function.Relu)
nn.add_output_layer(1, Activation_Function.Relu)

# You must always build the network after the structure is finalized
nn.build()
# This is an important value for how quickly the network learns. This rate is good for this task.
nn.setLearningRate(0.001)
# If you wish to save the network after finishing, use this method to set the file path and file name
nn.set_filepath('temp-net.json')
# As the network trains, you can get updates to the terminal for how training is going. This number states how ofter you might get an update
nn.print_loss_every_iterations(100)

# This will tell the network to start training. We pass in the training inputs and its corresponding answers.
# The third parameter is how many training example will be handled at the same time. This speeds up training
# but somtimes at the cost of precision. The fourth parameter is the loss function the network will use.
# As the network trains, the lower the loss, the better.
nn.train(train_data,train_results, 1,5000, Loss.MSE)

# Now that the network has completed training, we will now pass a couple of temperatures into the network
# and see how well it performs at converting from celcius to fahrenheit
print("Network converts 8.0 to:", nn.execute([5.0])[0],". Correct value was: 41")
print("Network converts 20.0 to:", nn.execute([20.0])[0],". Correct value was: 68")
print("Network converts 29.0 to:", nn.execute([29.0])[0],". Correct value was: 84.2")
print("Network converts 42.0 to:", nn.execute([42.0])[0],". Correct value was: 107.6")

# The last step is we will save the network to the file path pointed out earlier
nn.save_network()






