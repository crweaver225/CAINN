#include "Neural_Layer.h"
#include <math.h>

Neural_Layer::Neural_Layer(Dimensions dimensions, Activation_Function activation_function) {
    this->_dimensions = dimensions;
    this->_activationFunction = activation_function;
}

Neural_Layer::~Neural_Layer() { }

Neural_Layer::Neural_Layer(Neural_Layer &&neural_layer)  noexcept  {
    _output = std::move(neural_layer._output);
    _weights = std::move(neural_layer._weights);
    _gradient = std::move(neural_layer._gradient);
    _bias = std::move(neural_layer._bias);
    _dimensions = std::move(neural_layer._dimensions);
    _activationFunction = neural_layer._activationFunction;
}

Neural_Layer& Neural_Layer::operator=(Neural_Layer &&neural_layer) noexcept  {
    return *this;
}

void Neural_Layer::PrintMetaData() {
    std::cout<<"Generic neural layer: "<<_dimensions.dimensions <<std::endl;
}

const Dimensions Neural_Layer::ReturnDimensions() const {
    return _dimensions;
}

float* Neural_Layer::GenerateBiasValues(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1,0.1);
    float* bias_layer = new float[size];
    for (int i = 0; i < size; ++i) {
        bias_layer[i] = dis(gen);
    }
    return bias_layer;
}

void Neural_Layer::SetBias(float *data) {
    float * new_bias = new float[_dimensions.columns];
    for (int b = 0; b < _dimensions.columns; b++) {
        new_bias[b] = data[b];
    }
    this->_bias.reset(new_bias);
}

void Neural_Layer::Training(bool train) { 
    if (train) {
        BuildGradient();
        _output->optimizeForTraining();
    } else {
        _gradient.reset();
        _output->optimizeForInference();
    }
}

auto Neural_Layer::ReturnActivationFunction() -> void (*)(float*,float*, int, int) {
    if (_activationFunction == Activation_Function::Sigmoid) {
        return Activation_Functions::sigmoid;
    } else if (_activationFunction == Activation_Function::Relu) {
        return Activation_Functions::relu;
    } else if (_activationFunction == Activation_Function::Leaky_Relu) {
        return Activation_Functions::leaky_relu;
    } else if (_activationFunction == Activation_Function::SoftMax) {
        return Activation_Functions::softmax;
    } else {
        return Activation_Functions::pass;
    }
}

auto Neural_Layer::ReturnActivationFunctionDerivative() -> void (*)(float*, float*, int) {
    if (_activationFunction == Activation_Function::Sigmoid) {
        return Activation_Functions::sigmoid_d;
    } else if (_activationFunction == Activation_Function::Relu) {
        return Activation_Functions::relu_d;
    } else if (_activationFunction == Activation_Function::Leaky_Relu) {
        return Activation_Functions::leaky_relu_d;
    } else if (_activationFunction == Activation_Function::SoftMax) {
        return Activation_Functions::softmax_d;
    } else {
        return Activation_Functions::pass_d;
    }
}

void Neural_Layer::SetActiveDimensions(int batch_size) {
    this->_output->SetActiveDimension(batch_size);
}

void Neural_Layer::SetBatchDimensions(int batch_size) {
    _dimensions.dimensions = batch_size;
    _previousLayer_Dimensions.dimensions = batch_size;
    _output = std::make_unique<Tensor>(batch_size, _output->NumberOfChannels(), _output->NumberOfRows(), _dimensions.columns);
}

const float Neural_Layer::ReturnL2() const {
    return _weights.get()->SumTheSquares();
}

void Neural_Layer::ClearGradient() {
    _gradient.get()->ResetTensor();
}

void Neural_Layer::BuildGradient() {
    _gradient = std::make_unique<Tensor>(Tensor(_previousLayer_Dimensions.dimensions,
                                                _previousLayer_Dimensions.channels,
                                                _previousLayer_Dimensions.rows,
                                                _previousLayer_Dimensions.columns));
    _gradient->optimizeForTraining();
}

Activation_Function Neural_Layer::ReturnActivationFunctionType() const {
    return _activationFunction;
}

void Neural_Layer::PrintOutput() {
    _output->Print();
}