from enum import Enum

#Import the Neural Network c++ library in
from Neural_Network import Neural_Network
from Neural_Network import Activation_Function
from Neural_Network import Loss

# We previously trained a neural network to convert temperatures in celcius to fahrenheit and then saved that network.
# Here we can demonstrate loading up that network and using it for conversions

# After importing the Neural Network library, you can now initialize the neural network object
nn = Neural_Network()

# Call load_network on the Neural_Network object with the file path of the saved network.
#This will load up the network and build it so that it is ready to execute
nn.load_network('temp-net.json')

# We can now pass values into the network and they should execute just as they did previously
print("Network converts 8.0 to:", nn.execute([5.0])[0],". Correct value was: 41")
print("Network converts 20.0 to:", nn.execute([20.0])[0],". Correct value was: 68")
print("Network converts 29.0 to:", nn.execute([29.0])[0],". Correct value was: 84.2")
print("Network converts 42.0 to:", nn.execute([42.0])[0],". Correct value was: 107.6")







