#ifndef NETWORK_SAVER_H_
#define NETWORK_SAVER_H_

#include <typeinfo>
#include <fstream>
#include <cstring>
#include "Neural_Network.h"
    
class Neural_Network;

class Network_Saver {
    int *network_layers;
    int *activation_functions;
   // float *weights;
  //  float *bias;
    int *neurons;
public:
    void save_network(Neural_Network *neural_network, std::string &path);
    void load_network(std::string &path);
};

#endif /* NETWORK_SAVER_H_ */