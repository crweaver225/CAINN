#include <vector>
#include <iostream>
#include <memory>
#include "Tensor.h"
#include "Activation_Functions.h"

#ifndef NEURAL_LAYER_H_
#define NEURAL_LAYER_H_

class Neural_Layer {

private:
    friend class Network_Saver;
    void setBias(float *data);
protected:
    std::unique_ptr<Tensor> weights;
    std::unique_ptr<Tensor> gradient;
    std::unique_ptr<float> bias;
    std::shared_ptr<Neural_Layer> previous_layer;
    std::vector<int> dimensions;
    Activation_Function activation_function;
    Loss loss_function;

    const Tensor* previous_layer_output();
    Tensor* previous_layer_gradient();

    float* generateBiasValues(int size);

    void buildGradient();
    
    auto returnActivationFunction() -> void (*)(float*, float*, int, int);
    auto returnActivationFunctionDerivative() -> void (*)(float*, float*, int);


public:
    Neural_Layer(std::vector<int> dimensions, Activation_Function activation_function);
    ~Neural_Layer();
    Neural_Layer(const Neural_Layer &neural_layer);
    Neural_Layer& operator = (const Neural_Layer &neural_layer);
    Neural_Layer(Neural_Layer &&neural_layer);
    Neural_Layer& operator=(Neural_Layer &&neural_layer);

    std::unique_ptr<Tensor> output_results;

    const float returnL2() const;

    virtual void build(std::shared_ptr<Neural_Layer> previous_layer) = 0;
    virtual void forward_propogate() = 0;
    virtual void backpropogate() = 0;

    void clearGradient();

    virtual void printMetaData();
    const std::vector<int>& output_dimensions();
    virtual void setBatchDimensions(int batch_size);
    void setActiveDimensions(int batch_size);

    // functions for input layer
    virtual void addInput(float *input);
    virtual void addInputInBatches(const int dimensions, float **input);

    //functions for output layer
    virtual void setLossFunction(Loss loss);
    virtual void training(bool train);
    virtual void calculateError(float **target, float regularization);
    virtual void printError();
    virtual void printFinalResults();
    virtual void resetLoss();
    virtual float returnLoss() const;

    // for network saver
    Activation_Function returnActivationFunctionType() const;
};

#endif /* NEURAL_LAYER_H_ */