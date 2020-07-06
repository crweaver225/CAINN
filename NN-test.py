from enum import Enum
import ctypes
import sys
import gc
from Neural_Network import Neural_Network
from Neural_Network import Activation_Function
from Neural_Network import Loss

nn = Neural_Network()

nn.add_input_layer(1)
nn.add_fully_connected_layer(20, Activation_Function.Sigmoid)
nn.add_fully_connected_layer(40, Activation_Function.Relu)
nn.add_fully_connected_layer(10, Activation_Function.Relu)
nn.add_output_layer(1, Activation_Function.Relu)

nn.build()
nn.setLearningRate(0.001)
nn.set_filepath('temp-net.json')
#nn.save_best_automatically(True)
#nn.stop_training_automatically(True)
nn.print_loss_every_iterations(1000)
nn.train([[-10.0],[0.0],[8.0],[15.0],[22.0],[38.0],[50.0],[12.0],[19.0],[44.0]], [[14.0],[32.0],[46.0],[59.0],[72.0],[100.0],[122.0],[53.6],[66.2],[111.2]], 1,50000, Loss.MSE)

#nn.load_network('temp-net.json')
print(nn.execute([8.0])) #46
print(nn.execute([15.0])) #59
print(nn.execute([20.0])) #68
print(nn.execute([38.0])) #100
print(nn.execute([50.0])) #122
print(nn.execute([12.0])) # 53.6


#nn.train([[0.2], [0.4]], [[0.4],[0.8]],1,90000)
#nn.train([[0.2],[0.3],[0.4]], [[0.4],[0.6],[0.8]],2,2)
#print(nn.execute([0.2]))
#print(nn.execute([0.3]))
#print(nn.execute([0.4]))
#print(nn.execute([0.5]))
#nn.save_network('temp-net.json')





