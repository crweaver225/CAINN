#include "Network_Saver.h"


void Network_Saver::SaveNetwork(Neural_Network *neural_network, std::string &path) {

   std::vector<int> network_layers;
   std::vector<int> activation_functions;
   std::vector<int> neurons;
   std::vector<std::vector<float>> bias;
   std::vector<std::vector<float>> weights;

    for (std::shared_ptr<Neural_Layer> x : neural_network->_neuralLayers) {
        if (dynamic_cast<Input_layer*>(x.get()) != nullptr) {
            network_layers.push_back(1);
        } else if (dynamic_cast<Fully_Connected_Layer*>(x.get()) != nullptr) {
            network_layers.push_back(2);
            bias.push_back(std::vector<float>(x.get()->_bias.get(), x.get()->_bias.get() + x.get()->_dimensions.back()));
            std::vector<int> weight_shape = x.get()->_weights.get()->Shape();
            weights.push_back(std::vector<float>(x.get()->_weights.get()->ReturnData() , x.get()->_weights.get()->ReturnData() + (weight_shape[1] * weight_shape[2])));
        } else if (dynamic_cast<Output_Layer*>(x.get()) != nullptr) {
            network_layers.push_back(3);
            std::vector<int> weight_shape = x.get()->_weights.get()->Shape();
            weights.push_back(std::vector<float>(x.get()->_weights.get()->ReturnData() , x.get()->_weights.get()->ReturnData() + (weight_shape[1] * weight_shape[2])));
        }
        activation_functions.push_back(x.get()->ReturnActivationFunctionType());
        neurons.push_back(x.get()->OutputDimensions().back());
    }

    json network_json;
    network_json["layer_type"] = network_layers;
    network_json["activation_functions"] = activation_functions;
    network_json["neurons"] = neurons;
    network_json["bias"] = bias;
    network_json["weights"] = weights;

    std::ofstream file(path);
    file << network_json;
}

void Network_Saver::LoadNetwork(Neural_Network *neural_network, std::string &path) {

    neural_network->_neuralLayers.clear();

    std::ifstream i(path);
    json network_json;
    i >> network_json;

    std::vector<int> network_layers = network_json["layer_type"].get<std::vector<int>>();
    std::vector<int> activation_functions = network_json["activation_functions"].get<std::vector<int>>();
    std::vector<int> neurons = network_json["neurons"].get<std::vector<int>>();
    std::vector<std::vector<float>> bias =  network_json["bias"].get<std::vector<std::vector<float>>>();
    std::vector<std::vector<float>> weights = network_json["weights"].get<std::vector<std::vector<float>>>();

    for (int layer = 0; layer < network_layers.size(); layer++) {
        if (network_layers[layer] == 1) {
            neural_network->AddInputLayer(neurons[layer]);
        } else if (network_layers[layer] == 2) {
            neural_network->AddFullyConnectedLayer(neurons[layer], activation_functions[layer]);
        } else if (network_layers[layer] == 3) {
            neural_network->AddOutputLayer(neurons[layer], activation_functions[layer]);
        }
    }
    neural_network->Build();

    int bias_index = 0;
    int weight_index = 0;
    for (int layer = 0; layer < network_layers.size(); layer++) {
        if (network_layers[layer] == 2) {
            neural_network->_neuralLayers[layer].get()->SetBias(bias[bias_index].data());
            bias_index++;
            neural_network->_neuralLayers[layer].get()->_weights.get()->SetData(weights[weight_index].data());
            weight_index++;
        } else if (network_layers[layer] == 3) {
            neural_network->_neuralLayers[layer].get()->_weights.get()->SetData(weights[weight_index].data());
            weight_index++;
        }
    }
}

