#include <vector>
#include <iostream>
#include <memory>
#include "Tensor.h"
#include "Dimensions.h"
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
    std::unique_ptr<Tensor> _output;
    std::unique_ptr<float> _bias;

    Tensor const* _input;

    Dimensions _previousLayer_Dimensions;
    Dimensions _dimensions;

    Activation_Function _activationFunction;
    Loss _lossFunction;

    float* GenerateBiasValues(int size);

    void BuildGradient();
    
    auto ReturnActivationFunction() -> void (*)(float*, float*, int, int);
    auto ReturnActivationFunctionDerivative() -> void (*)(float*, float*, int);


public:
    Neural_Layer(Dimensions dimensions, Activation_Function activation_function);
    ~Neural_Layer();
    Neural_Layer(const Neural_Layer &neural_layer) = delete;
    Neural_Layer& operator = (const Neural_Layer &neural_layer) = delete;
    Neural_Layer(Neural_Layer &&neural_layer) noexcept;
    Neural_Layer& operator=(Neural_Layer &&neural_layer) noexcept;

    
    const Dimensions ReturnDimensions() const;
    const float ReturnL2() const;

    virtual void Build(Neural_Layer const* previousLayer) = 0;
    virtual Tensor const* ForwardPropogate(Tensor const* input) = 0;
    virtual Tensor * Backpropogate(Tensor* gradient) = 0;

    void ClearGradient();

    virtual void PrintMetaData();

    virtual void SetBatchDimensions(int batch_size);
    virtual void SetActiveDimensions(int batch_size);
    virtual void Training(bool train);

    virtual void PrintOutput();

    // for network saver
    Activation_Function ReturnActivationFunctionType() const;
};

#endif /* NEURAL_LAYER_H_ */