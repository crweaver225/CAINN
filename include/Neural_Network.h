#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <iostream>
#include <vector>
#include <memory>
#include <ctime>
#include <limits>
#include "Input_Layer.h"
#include "Fully_Connected_Layer.h"
#include "Dropout_Layer.h"
#include "Output_Layer.h"
#include "Network_Saver.h"
#include "Flatten_Layer.h"
#include "Embedding_Layer.h"

class Neural_Network {

private:
    std::vector<std::shared_ptr<Neural_Layer>> _neuralLayers;
    void Backpropogate();
    void ClearGradients();
    const float CalculateL2() const;
    void randomizeDropout();
    bool _saveIfBest = false;
    bool _stopAutomatically = false;
    float _bestLoss;
    bool _droppoutLayerExists = false;
    int _printLossEveryIterations;
    std::string _filePath;
    friend class Network_Saver;

public:

    void AddInputLayer(int dimension);
    void AddFullyConnectedLayer(int neurons, int activation_function);
    void AddDropoutLayer(float dropped);
    void AddOutputLayer(int neurons, int activation_function);
    void AddFlattenLayer();
    void AddEmbeddingLayer(int unique_words_length, int output);
    void Build();
    void SetLearningRate(float learning_rate);
    const float* Execute(float *input);
    void Train(float **input, float **targets, int batch_size, int epochs, int loss_function, int input_size);
    void SaveNetwork();
    void LoadNetwork(size_t len, const char* path);
    const int OutputDimensions() const;
    void SaveBestAutomatically(bool activate);
    void StopTrainingAutomatically(bool activate);
    void SetFilepath(const char* path);
    void SetPrintLossEverIterations(int iteration);
};

#endif /* NEURAL_NETWORK_H_ */
