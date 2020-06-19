#include "Neural_Network.h"

#ifdef __cplusplus
extern "C" {
  #endif
  Neural_Network* Neural_Network_new() {return new Neural_Network();}
  void Neural_Network_set_learning_rate(Neural_Network* neural_network, float learning_rate) {return neural_network->setLearningRate(learning_rate);}
  void Neural_Network_build(Neural_Network* neural_nerwork) {return neural_nerwork->build();}
  void Neural_Network_add_input_layer(Neural_Network* neural_nerwork, int *dimensions, int dimension) {return neural_nerwork->addInputLayer(dimensions, dimension);}
  void Neural_Network_add_fully_connected_layer(Neural_Network* neural_nerwork, int neurons, int activation_function) {return neural_nerwork->addFullyConnectedLayer(neurons, activation_function);}
  void Neural_Network_add_output_layer(Neural_Network* neural_nerwork, int neurons, int activation_function) {return neural_nerwork->addOutputLayer(neurons, activation_function);}
  void Neural_Network_execute(Neural_Network* neural_nerwork, float* input) {return neural_nerwork->execute(input);}
  void Neural_Network_train(Neural_Network* neural_nerwork, float **input, float **targets, int batch_size, int epochs, int input_size) {return neural_nerwork->train(input,targets,batch_size,epochs, input_size);}
#ifdef __cplusplus
}
#endif