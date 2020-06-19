from enum import Enum
import ctypes
from enum import Enum
import ctypes

class Activation_Function(Enum):
    Sigmoid = 0
    Pass = 1
    Relu = 2
    Leaky_Relu = 3
    Tanh = 4
    SoftMax = 5
    Maxpool = 6
    Flatten = 7

class Layer_Types(Enum):
    Fully_Connected = "Fully_Connected"
    Convoluted = "Convoluted"
    Recurrent = "Recurrent"
    Maxpool = "Maxpool"
    Flatten = "Flatten"
    Input = "Input"

lib = ctypes.cdll.LoadLibrary('./build/libNeural_Network.so')

class Neural_Network(object):
    def __init__(self):
        lib.Neural_Network_new.argtypes = [ctypes.c_void_p]
        lib.Neural_Network_new.restype = ctypes.c_void_p


        lib.Neural_Network_build.argtypes =  [ctypes.c_void_p]
        lib.Neural_Network_build.restype = ctypes.c_void_p

        lib.Neural_Network_add_input_layer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.Neural_Network_add_input_layer.restype = ctypes.c_void_p

        lib.Neural_Network_add_fully_connected_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.Neural_Network_add_fully_connected_layer.restype = ctypes.c_void_p

        lib.Neural_Network_add_output_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.Neural_Network_add_output_layer.restype = ctypes.c_void_p

        lib.Neural_Network_execute.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.Neural_Network_execute.restype = ctypes.c_void_p

        lib.Neural_Network_set_learning_rate.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.Neural_Network_set_learning_rate.restype = ctypes.c_void_p

        lib.Neural_Network_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),ctypes.c_int,ctypes.c_int,ctypes.c_int]
        lib.Neural_Network_train.restype = ctypes.c_void_p

        self.obj = lib.Neural_Network_new(1)

    def setLearningRate(self, learning_rate):
        lib.Neural_Network_set_learning_rate(self.obj, learning_rate)

    def build(self):
        lib.Neural_Network_build(self.obj)

    def add_input_layer(self, dimensions):
        int_pointers = (ctypes.c_int * len(dimensions))(*dimensions)
        lib.Neural_Network_add_input_layer(self.obj, int_pointers, len(dimensions))

    def add_fully_connected_layer(self, neurons, activation_function):
        #ac_string = activation_function.value
        #str_pointer = ctypes.c_char_p(ac_string.encode('utf-8'))
        lib.Neural_Network_add_fully_connected_layer(self.obj, neurons, activation_function.value)

    def add_output_layer(self, neurons, activation_function):
        lib.Neural_Network_add_output_layer(self.obj, neurons, activation_function.value)

    def execute(self, input):
        float_pointers = (ctypes.c_float * len(input))(*input)
        lib.Neural_Network_execute(self.obj, float_pointers)

    def train(self, input, targets, batch_size, epochs):
        cpp_inputs = []
        cpp_targets = []
        for index in range(0,len(input)):
            cpp_inputs.append((ctypes.c_float * len(input[index]))(*input[index]))
            cpp_targets.append((ctypes.c_float * len(targets[index]))(*targets[index]))
        cpp_inputs_pointer = (ctypes.POINTER(ctypes.c_float) * len(input))(*cpp_inputs)
        cpp_inputs_targets = (ctypes.POINTER(ctypes.c_float) * len(input))(*cpp_targets)
        lib.Neural_Network_train(self.obj, cpp_inputs_pointer, cpp_inputs_targets, batch_size, epochs, len(input))
