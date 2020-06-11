from enum import Enum
import ctypes
import sys
import gc
from Neural_Network import Neural_Network
from Neural_Network import Activation_Function

nn = Neural_Network()
nn.add_input_layer([1])
nn.add_fully_connected_layer(2, Activation_Function.Relu)
nn.add_output_layer(1, Activation_Function.Relu)

nn.build()
nn.train([[8]], [[16]],1,10)
#nn.execute([0.1])

gc.collect()
sys.exit()