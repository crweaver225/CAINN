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

public:
    void SaveNetwork(Neural_Network *neural_network, std::string &path);
    void LoadNetwork(Neural_Network *neural_network, std::string &path);
};

#endif /* NETWORK_SAVER_H_ */