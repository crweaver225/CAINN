#include "Neural_Network.h"

#ifdef __cplusplus
extern "C" {
  #endif
  Neural_Network* Neural_Network_new() {return new Neural_Network();}
  void Neural_Network_print_loss_every_iteartions(Neural_Network* neural_nerwork, int iterations){return neural_nerwork->set_print_loss_ever_iterations(iterations);}
  void Neural_Network_set_filepath(Neural_Network* neural_network, const char* path){return neural_network->set_filepath(path);} 
  void Neural_Network_stop_training_automatically(Neural_Network* neural_network, bool activate) {return neural_network->stop_training_automatically(activate);}
  void Neural_Network_save_best_automatically(Neural_Network* neural_nerwork, bool activate) {return neural_nerwork->save_best_automatically(activate);}
  const int Neural_Network_output_dimensions(Neural_Network* neural_network) {return neural_network->output_dimensions();}
  void Neural_Network_load_network(Neural_Network* neural_network, size_t len,const char* path) {return neural_network->load_network(len,path);}
  void Neural_Network_save_network(Neural_Network* neural_network) {return neural_network->save_network();}
  void Neural_Network_set_learning_rate(Neural_Network* neural_network, float learning_rate) {return neural_network->setLearningRate(learning_rate);}
  void Neural_Network_build(Neural_Network* neural_nerwork) {return neural_nerwork->build();}
  void Neural_Network_add_input_layer(Neural_Network* neural_nerwork, int dimension) {return neural_nerwork->addInputLayer(dimension);}
  void Neural_Network_add_fully_connected_layer(Neural_Network* neural_nerwork, int neurons, int activation_function) {return neural_nerwork->addFullyConnectedLayer(neurons, activation_function);}
  void Neural_Network_add_output_layer(Neural_Network* neural_nerwork, int neurons, int activation_function) {return neural_nerwork->addOutputLayer(neurons, activation_function);}
  const float * Neural_Network_execute(Neural_Network* neural_nerwork, float* input) {return neural_nerwork->execute(input);}
  void Neural_Network_train(Neural_Network* neural_nerwork, float **input, float **targets, int batch_size, int epochs, int loss_function, int input_size) {return neural_nerwork->train(input,targets,batch_size,epochs,loss_function,input_size);}
#ifdef __cplusplus
}
#endif