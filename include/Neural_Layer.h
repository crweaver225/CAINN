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
    void SetBias(float *data);
protected:
    std::unique_ptr<Tensor> _weights;
    std::unique_ptr<Tensor> _gradient;
    std::unique_ptr<float> _bias;
    std::shared_ptr<Neural_Layer> _previousLayer;
    std::vector<int> _dimensions;
    Activation_Function _activationFunction;
    Loss _lossFunction;

    const Tensor* PreviousLayerOutput();
    Tensor* PreviousLayerGradient();

    float* GenerateBiasValues(int size);

    void BuildGradient();
    
    auto ReturnActivationFunction() -> void (*)(float*, float*, int, int);
    auto ReturnActivationFunctionDerivative() -> void (*)(float*, float*, int);


public:
    Neural_Layer(std::vector<int> dimensions, Activation_Function activation_function);
    ~Neural_Layer();
    Neural_Layer(const Neural_Layer &neural_layer) = delete;
    Neural_Layer& operator = (const Neural_Layer &neural_layer) = delete;
    Neural_Layer(Neural_Layer &&neural_layer);
    Neural_Layer& operator=(Neural_Layer &&neural_layer);

    std::unique_ptr<Tensor> _outputResults;

    const float ReturnL2() const;

    virtual void Build(std::shared_ptr<Neural_Layer> previousLayer) = 0;
    virtual void ForwardPropogate() = 0;
    virtual void Backpropogate() = 0;

    void ClearGradient();

    virtual void PrintMetaData();
    virtual const std::vector<int>& OutputDimensions();
    virtual void SetBatchDimensions(int batch_size);
    void SetActiveDimensions(int batch_size);

    // functions for input layer
    virtual void AddInput(float *input);
    virtual void AddInputInBatches(const int dimensions, float **input);

    //functions for output layer
    virtual void SetLossFunction(Loss loss);
    virtual void Training(bool train);
    virtual void CalculateError(float **target, float regularization);
    virtual void PrintError();
    virtual void PrintFinalResults();
    virtual void ResetLoss();
    virtual float ReturnLoss() const;

    // for network saver
    Activation_Function ReturnActivationFunctionType() const;
};

#endif /* NEURAL_LAYER_H_ */