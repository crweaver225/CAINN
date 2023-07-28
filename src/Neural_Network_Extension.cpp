#include "Neural_Network.h"

#ifdef __cplusplus
extern "C" {
  #endif

  Neural_Network* Neural_Network_new() {return new Neural_Network();}
  void Neural_Network_print_loss_every_iteartions(Neural_Network* neural_nerwork, int iterations){return neural_nerwork->SetPrintLossEverIterations(iterations);}
  void Neural_Network_set_filepath(Neural_Network* neural_network, const char* path){return neural_network->SetFilepath(path);} 
  void Neural_Network_stop_training_automatically(Neural_Network* neural_network, bool activate) {return neural_network->StopTrainingAutomatically(activate);}
  void Neural_Network_apply_l2_regularization(Neural_Network* neural_network, bool activate) {return neural_network->TurnOnRegularization(activate);}
  void Neural_Network_save_best_automatically(Neural_Network* neural_nerwork, bool activate) {return neural_nerwork->SaveBestAutomatically(activate);}
  void Neural_Network_shuffle_training_date_per_epoch(Neural_Network *neural_network, bool activate) { return neural_network->SetShuffleDataFlag(activate);}
  const int Neural_Network_output_dimensions(Neural_Network* neural_network) {return neural_network->OutputDimensions();}
  void Neural_Network_load_network(Neural_Network* neural_network, size_t len,const char* path) {return neural_network->LoadNetwork(len,path);}
  void Neural_Network_save_network(Neural_Network* neural_network) {return neural_network->SaveNetwork();}
  void Neural_Network_set_learning_rate(Neural_Network* neural_network, float learning_rate) {return neural_network->SetLearningRate(learning_rate);}
  void Neural_Network_build(Neural_Network* neural_nerwork) {return neural_nerwork->Build();}
  void Neural_Network_add_convolutional_layer(Neural_Network* neural_nerwork, int kernels, int kernel_size, int stride) {return neural_nerwork->AddConvolutionalLayer(kernels, kernel_size, stride);}
  void Neural_Network_add_maxpool_layer(Neural_Network* neural_nerwork, int kernel_size, int stride) {return neural_nerwork->AddMaxpoolLayer(kernel_size, stride);}
  void Neural_Network_add_input_layer(Neural_Network* neural_nerwork, int* dimension, int size) {return neural_nerwork->AddInputLayer(dimension, size);}
  void Neural_Network_add_fully_connected_layer(Neural_Network* neural_nerwork, int neurons, int activation_function) {return neural_nerwork->AddFullyConnectedLayer(neurons, activation_function);}
  void Neural_Network_add_dropout_layer(Neural_Network* neural_network, float dropped) {return neural_network->AddDropoutLayer(dropped);}
  void Neural_Network_add_flatten_layer(Neural_Network* neural_network) {return neural_network->AddFlattenLayer();}
  void Neural_Network_add_embedding_layer(Neural_Network* neural_network, int unique_words_length, int output) {return neural_network->AddEmbeddingLayer(unique_words_length, output);}
  void Neural_Network_add_output_layer(Neural_Network* neural_nerwork, int neurons, int activation_function) {return neural_nerwork->AddOutputLayer(neurons, activation_function);}
  const float * Neural_Network_execute(Neural_Network* neural_nerwork, float* input) {return neural_nerwork->Execute(input);}
  void Neural_Network_train(Neural_Network* neural_nerwork, float **input, float **targets, int batch_size, int epochs, int loss_function, int input_size) {return neural_nerwork->Train(input,targets,batch_size,epochs,loss_function,input_size);}
#ifdef __cplusplus
}
#endif