#include <iostream>
#include <vector>
#include <memory>
#include <ctime>
#include "Input_Layer.h"
#include "Fully_Connected_Layer.h"
#include "Output_Layer.h"

#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

class Neural_Network {

private:
    std::vector<std::shared_ptr<Neural_Layer>> neural_layers;
    void backpropogate();
    void clearGradients();
    float calculateL2();

public:

    Neural_Network();
    ~Neural_Network();
    Neural_Network(const Neural_Network &neural_network);
    Neural_Network& operator = (const Neural_Network &neural_network);
    Neural_Network(Neural_Network &&neural_network);
    Neural_Network& operator = (Neural_Network &&neural_network);

    void addInputLayer(int *dimensions, int dimension);
    void addFullyConnectedLayer(int neurons, int activation_function);
    void addOutputLayer(int neurons, int activation_function);
    void build();
    void execute(float *input);
    void train(float **input, float **targets, int batch_size, int epochs, int input_size);
};

extern "C" {
  Neural_Network* Neural_Network_new() {return new Neural_Network();}
  void Neural_Network_build(Neural_Network* neural_nerwork) {return neural_nerwork->build();}
  void Neural_Network_add_input_layer(Neural_Network* neural_nerwork, int *dimensions, int dimension) {return neural_nerwork->addInputLayer(dimensions, dimension);}
  void Neural_Network_add_fully_connected_layer(Neural_Network* neural_nerwork, int neurons, int activation_function) {return neural_nerwork->addFullyConnectedLayer(neurons, activation_function);}
  void Neural_Network_add_output_layer(Neural_Network* neural_nerwork, int neurons, int activation_function) {return neural_nerwork->addOutputLayer(neurons, activation_function);}
  void Neural_Network_execute(Neural_Network* neural_nerwork, float* input) {return neural_nerwork->execute(input);}
  void Neural_Network_train(Neural_Network* neural_nerwork, float **input, float **targets, int batch_size, int epochs, int input_size) {return neural_nerwork->train(input,targets,batch_size,epochs, input_size);}
}

#endif /* NEURAL_NETWORK_H_ */