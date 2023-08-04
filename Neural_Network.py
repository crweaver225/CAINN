from enum import Enum
import ctypes


class Loss(Enum):
    MSE = 0
    ASE = 1
    CrossEntropy = 2


class Activation_Function(Enum):
    Sigmoid = 0
    Pass = 1
    Relu = 2
    Leaky_Relu = 3
    Tanh = 4
    SoftMax = 5
    Maxpool = 6
    Flatten = 7
    Global_Maxpool = 8


class Layer_Types(Enum):
    Fully_Connected = "Fully_Connected"
    Convoluted = "Convoluted"
    Recurrent = "Recurrent"
    Maxpool = "Maxpool"
    Global_Maxpool = "Global_Maxpool"
    Flatten = "Flatten"
    Input = "Input"
    Dropout = "Dropout"


lib = ctypes.cdll.LoadLibrary('./build/libNeural_Network.so')


class Neural_Network(object):
    def __init__(self):
        lib.Neural_Network_new.argtypes = [ctypes.c_void_p]
        lib.Neural_Network_new.restype = ctypes.c_void_p

        lib.Neural_Network_build.argtypes = [ctypes.c_void_p]
        lib.Neural_Network_build.restype = ctypes.c_void_p

        lib.Neural_Network_add_input_layer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.Neural_Network_add_input_layer.restype = ctypes.c_void_p

        lib.Neural_Network_add_fully_connected_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.Neural_Network_add_fully_connected_layer.restype = ctypes.c_void_p

        lib.Neural_Network_add_dropout_layer.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.Neural_Network_add_dropout_layer.restype = ctypes.c_void_p

        lib.Neural_Network_add_flatten_layer.argtypes = [ctypes.c_void_p]
        lib.Neural_Network_add_flatten_layer.restype = ctypes.c_void_p

        lib.Neural_Network_add_embedding_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.Neural_Network_add_embedding_layer.restype = ctypes.c_void_p

        lib.Neural_Network_add_output_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.Neural_Network_add_output_layer.restype = ctypes.c_void_p

        lib.Neural_Network_execute.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.Neural_Network_execute.restype = ctypes.POINTER(ctypes.c_float)

        lib.Neural_Network_set_learning_rate.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.Neural_Network_set_learning_rate.restype = ctypes.c_void_p

        lib.Neural_Network_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                             ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_int]
        lib.Neural_Network_train.restype = ctypes.c_void_p

        lib.Neural_Network_set_filepath.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.Neural_Network_set_filepath.restype = ctypes.c_void_p

        lib.Neural_Network_save_network.argtypes = [ctypes.c_void_p]
        lib.Neural_Network_save_network.restype = ctypes.c_void_p

        lib.Neural_Network_load_network.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
        lib.Neural_Network_load_network.restype = ctypes.c_void_p

        lib.Neural_Network_output_dimensions.argtypes = [ctypes.c_void_p]
        lib.Neural_Network_output_dimensions.restype = ctypes.c_int

        lib.Neural_Network_save_best_automatically.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.Neural_Network_save_best_automatically.restype = ctypes.c_void_p

        lib.Neural_Network_stop_training_automatically.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.Neural_Network_stop_training_automatically.restype = ctypes.c_void_p

        lib.Neural_Network_print_loss_every_iteartions.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.Neural_Network_print_loss_every_iteartions.restype = ctypes.c_void_p

        lib.Neural_Network_shuffle_training_date_per_epoch.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.Neural_Network_shuffle_training_date_per_epoch.restype = ctypes.c_void_p

        lib.Neural_Network_apply_l2_regularization.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.Neural_Network_apply_l2_regularization.restype = ctypes.c_void_p

        self.obj = lib.Neural_Network_new(1)

    def __del__(self):
        del self.obj

    def setLearningRate(self, learning_rate):
        lib.Neural_Network_set_learning_rate(self.obj, learning_rate)

    def build(self):
        lib.Neural_Network_build(self.obj)

    def add_input_layer(self, dimensions):
        int_pointers = (ctypes.c_int * len(dimensions))(*dimensions)
        lib.Neural_Network_add_input_layer(self.obj, int_pointers, len(dimensions))

    def add_fully_connected_layer(self, neurons, activation_function):
        lib.Neural_Network_add_fully_connected_layer(self.obj, neurons, activation_function.value)

    def add_dropout_layer(self, dropped):
        lib.Neural_Network_add_dropout_layer(self.obj, dropped)

    def add_flatten_layer(self):
        lib.Neural_Network_add_flatten_layer(self.obj)

    def add_embedding_layer(self, unique_words_size, output):
        lib.Neural_Network_add_embedding_layer(self.obj, unique_words_size, output)

    def add_output_layer(self, neurons, activation_function):
        lib.Neural_Network_add_output_layer(self.obj, neurons, activation_function.value)

    def execute(self, input):
        float_pointers = (ctypes.c_float * len(input))(*input)
        array_pointer = lib.Neural_Network_execute(self.obj, float_pointers)
        output_size = lib.Neural_Network_output_dimensions(self.obj)
        final_array = [array_pointer[i] for i in range(output_size)]
        return final_array

    def train(self, input, targets, batch_size, epochs, loss_function=Loss.MSE):
        cpp_inputs = []
        cpp_targets = []
        for index in range(0, len(input)):
            cpp_inputs.append((ctypes.c_float * len(input[index]))(*input[index]))
            cpp_targets.append((ctypes.c_float * len(targets[index]))(*targets[index]))
        cpp_inputs_pointer = (ctypes.POINTER(ctypes.c_float) * len(input))(*cpp_inputs)
        cpp_inputs_targets = (ctypes.POINTER(ctypes.c_float) * len(input))(*cpp_targets)
        lib.Neural_Network_train(self.obj, cpp_inputs_pointer, cpp_inputs_targets, batch_size, epochs,
                                 loss_function.value, len(input))

    def set_filepath(self, path):
        cString = ctypes.create_string_buffer(len(path))
        path_string_ptr = ctypes.c_char_p(path.encode('utf-8'))
        lib.Neural_Network_set_filepath(self.obj, path_string_ptr)

    def save_network(self):
        lib.Neural_Network_save_network(self.obj)

    def load_network(self, path):
        s = ctypes.create_string_buffer(len(path))
        path_string_ptr = ctypes.c_char_p(path.encode('utf-8'))
        lib.Neural_Network_load_network(self.obj, len(s), path_string_ptr)

    def save_best_automatically(self, activate):
        lib.Neural_Network_save_best_automatically(self.obj, activate)

    def turn_on_l2_regularization(self, activate):
        lib.Neural_Network_apply_l2_regularization(self.obj, activate)

    def stop_training_automatically(self, activate):
        lib.Neural_Network_stop_training_automatically(self.obj, activate)

    def print_loss_every_iterations(self, iterations):
        lib.Neural_Network_print_loss_every_iteartions(self.obj, iterations)

    def shuffle_training_data(self, activate):
        lib.Neural_Network_shuffle_training_date_per_epoch(self.obj, activate)
