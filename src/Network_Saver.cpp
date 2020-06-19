#include "Network_Saver.h"


void Network_Saver::save_network(Neural_Network *neural_network, std::string &path) {

    
    this->network_layers = new int[neural_network->neural_layers.size()];
    this->activation_functions = new int[neural_network->neural_layers.size()];
    this->neurons = new int[neural_network->neural_layers.size()];
    int index = 0;
    for (std::shared_ptr<Neural_Layer> x : neural_network->neural_layers) {
        if (dynamic_cast<Input_layer*>(x.get()) != nullptr) {
            std::cout << "Layer is an input layer" << std::endl;
            this->network_layers[index] = 1;
        } else if (dynamic_cast<Fully_Connected_Layer*>(x.get()) != nullptr) {
            std::cout << "Layer is an fully connected layer" << std::endl;
            this->network_layers[index] = 2;
        } else if (dynamic_cast<Output_Layer*>(x.get()) != nullptr) {
            std::cout << "Layer is an output layer" << std::endl;
            this->network_layers[index] = 3;
        }
        this->activation_functions[index] = x.get()->returnActivationFunctionType();
        this->neurons[index] = x.get()->output_dimensions()[1];
        index += 1;
    }
}

void Network_Saver::load_network(std::string &path) {
 
}

