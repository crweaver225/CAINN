#include "Network_Saver.h"


void Network_Saver::SaveNetwork(Neural_Network *neural_network, std::string &path) {

    std::vector<int> network_layers;
    std::vector<int> activation_functions;
    std::vector<int> neurons;
    std::vector<std::vector<float>> bias;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<int>> dimensions;
    std::vector<float> dropped;

    for (Neural_Layer* x : neural_network->network()) {
        if (dynamic_cast<Input_layer*>(x) != nullptr) {
            network_layers.push_back(1);
            dimensions.push_back(x->_dimensions.vector());
            dropped.push_back(0.0f);
        } else if (dynamic_cast<Fully_Connected_Layer*>(x) != nullptr) {
            network_layers.push_back(2);
            bias.push_back(std::vector<float>(x->_bias.get(), x->_bias.get() + x->_dimensions.columns));
            Dimensions weight_dimensions = x->_weights->dimensions();
            weights.push_back(std::vector<float>(x->_weights.get()->ReturnData() , x->_weights.get()->ReturnData() + (weight_dimensions.rows * weight_dimensions.columns)));
            dimensions.push_back(x->_dimensions.vector());
            dropped.push_back(0.0f);
        } else if (dynamic_cast<Output_Layer*>(x) != nullptr) {
            network_layers.push_back(3);
            Dimensions weight_shape = x->_weights->dimensions();
            weights.push_back(std::vector<float>(x->_weights.get()->ReturnData() , x->_weights.get()->ReturnData() + (weight_shape.rows * weight_shape.columns)));
            dimensions.push_back(x->_dimensions.vector());
            dropped.push_back(0.0f);
        } else if (dynamic_cast<Embedding_Layer*>(x) != nullptr) {
            network_layers.push_back(4);
            Dimensions weight_shape = x->_weights->dimensions();
            weights.push_back(std::vector<float>(x->_weights.get()->ReturnData() , x->_weights.get()->ReturnData() + (weight_shape.rows * weight_shape.columns)));
            dimensions.push_back(x->_dimensions.vector());
            dropped.push_back(0.0f);
        } else if (dynamic_cast<Flatten_Layer*>(x) != nullptr) {
            network_layers.push_back(5);
            dimensions.push_back(x->_dimensions.vector());
            dropped.push_back(0.0f);
        } else if (dynamic_cast<Dropout_Layer*>(x) != nullptr) {
            Dropout_Layer *dropout_layer = dynamic_cast<Dropout_Layer*>(x);
            network_layers.push_back(7);
            dropped.push_back(dropout_layer->returnPercentageDropped());
        }
        activation_functions.push_back(x->ReturnActivationFunctionType());
        neurons.push_back(x->ReturnDimensions().columns);
    }

    json network_json;
    network_json["layer_type"] = network_layers;
    network_json["activation_functions"] = activation_functions;
    network_json["neurons"] = neurons;
    network_json["bias"] = bias;
    network_json["weights"] = weights;
    network_json["dimensions"] = dimensions;
    network_json["dropped"] = dropped;

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
    std::vector<std::vector<int>> dimensions = network_json["dimensions"].get<std::vector<std::vector<int>>>();
    std::vector<float> dropped = network_json["dropped"].get<std::vector<float>>();\

    for (int layer = 0; layer < network_layers.size(); layer++) {
        if (network_layers[layer] == 1) {
            neural_network->AddInputLayer(&dimensions[layer][0], dimensions[layer].size());
        } else if (network_layers[layer] == 2) {
            neural_network->AddFullyConnectedLayer(neurons[layer], activation_functions[layer]);
        } else if (network_layers[layer] == 3) {
            neural_network->AddOutputLayer(neurons[layer], activation_functions[layer]);
        } else if (network_layers[layer] == 4) {
            neural_network->AddEmbeddingLayer(weights[1].size(), neurons[layer]);
        } else if (network_layers[layer] == 5) {
            neural_network->AddFlattenLayer();
        }   else if (network_layers[layer] == 7) {
            neural_network->AddDropoutLayer(dropped[layer]);
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
        } else if (network_layers[layer] == 4) {
            neural_network->_neuralLayers[layer].get()->_weights.get()->SetData(weights[weight_index].data());
            weight_index++;
        } else if (network_layers[layer] == 8) {
            neural_network->_neuralLayers[layer].get()->_weights.get()->SetData(weights[weight_index].data());
            weight_index++;
        }
    }
}


