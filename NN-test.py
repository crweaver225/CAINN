from enum import Enum
import ctypes
import sys
import gc
from Neural_Network import Neural_Network
from Neural_Network import Activation_Function

nn = Neural_Network()
nn.add_input_layer([1])
nn.add_fully_connected_layer(2, Activation_Function.Sigmoid)
#nn.add_fully_connected_layer(500, Activation_Function.Relu)
#nn.add_fully_connected_layer(100, Activation_Function.Relu)
nn.add_output_layer(1, Activation_Function.Relu)

nn.build()
#nn.setLearningRate(0.001)
#nn.train([[0.2], [0.4]], [[0.4],[0.8]],1,100)
#nn.train([[0.2],[0.3],[0.4]], [[0.4],[0.6],[0.8]],2,2)
#nn.train([[-10.0],[0.0],[8.0],[15.0],[22.0],[38.0]], [[14.0],[32.0],[46.0],[59.0],[72.0],[100.0]], 6,2000)

#nn.execute([0.2])
#nn.execute([0.3])
#nn.execute([0.4])
#nn.execute([0.5])

#nn.execute([8.0]) #46
#nn.execute([15.0]) #59
#nn.execute([20.0]) #68
#nn.execute([38.0]) #100
#nn.execute([50.0]) #122



