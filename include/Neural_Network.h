#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <iostream>
#include <vector>
#include <memory>
#include <ctime>
#include <limits>
#include "Input_Layer.h"
#include "Fully_Connected_Layer.h"
#include "Output_Layer.h"
#include "Network_Saver.h"


class Neural_Network {

private:
    std::vector<std::shared_ptr<Neural_Layer>> neural_layers;
    void backpropogate();
    void clearGradients();
    const float calculateL2() const;
    bool save_if_best = false;
    bool stop_automatically = false;
    float best_loss;
    int print_loss_every_iterations;
    std::string filePath;
    friend class Network_Saver;

public:

    Neural_Network();
    ~Neural_Network();
    Neural_Network(const Neural_Network &neural_network);
    Neural_Network& operator = (const Neural_Network &neural_network);
    Neural_Network(Neural_Network &&neural_network);
    Neural_Network& operator = (Neural_Network &&neural_network);

    void addInputLayer(int dimension);
    void addFullyConnectedLayer(int neurons, int activation_function);
    void addOutputLayer(int neurons, int activation_function);
    void build();
    void setLearningRate(float learning_rate);
    const float* execute(float *input);
    void train(float **input, float **targets, int batch_size, int epochs, int loss_function, int input_size);
    void save_network();
    void load_network(size_t len, const char* path);
    const int output_dimensions() const;
    void save_best_automatically(bool activate);
    void stop_training_automatically(bool activate);
    void set_filepath(const char* path);
    void set_print_loss_ever_iterations(int iteration);
};

#endif /* NEURAL_NETWORK_H_ */
