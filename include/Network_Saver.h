#ifndef NETWORK_SAVER_H_
#define NETWORK_SAVER_H_

#include <typeinfo>
#include <fstream>
#include <cstring>
#include "Neural_Network.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
    
class Neural_Network;

class Network_Saver {
  //  std::vector<int> network_layers;
  //  std::vector<int> activation_functions;
   // std::vector<int> neurons;
   // std::vector<std::vector<float>> bias;
   // std::vector<std::vector<float>> weights;

public:
    void save_network(Neural_Network *neural_network, std::string &path);
    void load_network(Neural_Network *neural_network, std::string &path);
};

#endif /* NETWORK_SAVER_H_ */