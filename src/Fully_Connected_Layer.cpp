#include "Fully_Connected_Layer.h"
#include <ctime>

Fully_Connected_Layer::Fully_Connected_Layer(std::vector<int> dimensions, Activation_Function af) : Neural_Layer(dimensions, af) {}

Fully_Connected_Layer& Fully_Connected_Layer::operator=(Fully_Connected_Layer &&fully_connected_layer) {
    if (this == &fully_connected_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(fully_connected_layer));
    return *this;
 }

Fully_Connected_Layer::~Fully_Connected_Layer() {}

void Fully_Connected_Layer::PrintMetaData() {
    std::cout<<"fully connected layer: ("<<_previousLayer.get()->OutputDimensions().back()<<","<<_dimensions.back()<<")"<<std::endl;
}

void Fully_Connected_Layer::Build(std::shared_ptr<Neural_Layer> previous_layer) {
    this->_previousLayer = previous_layer;
    this->_weights = std::unique_ptr<Tensor>(new Tensor(previous_layer.get()->OutputDimensions().back(), _dimensions.back()));
    this->_weights->AssignRandomValues();
    this->_bias = std::unique_ptr<float>(GenerateBiasValues(_dimensions.back()));
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(1, _dimensions.back()));
}

void Fully_Connected_Layer::ForwardPropogate() {
    _outputResults.get()->Matmul(*PreviousLayerOutput(), *_weights.get(), _bias.get(), ReturnActivationFunction());
}

void Fully_Connected_Layer::Backpropogate() {
    _gradient.get()->ApplyDerivative(*_outputResults.get(), ReturnActivationFunctionDerivative());
    PreviousLayerGradient()->UpdateGradients(*_gradient.get(), *_weights.get());
    _weights->UpdateWeights(*_gradient.get(), *PreviousLayerOutput());
    const float *gradient_data = _gradient.get()->ReturnData();
    for (int b = 0; b < _gradient.get()->Shape()[2]; ++b) {
        _bias.get()[b] -= (_bias.get()[b] * gradient_data[b]) * Tensor::_learningRate;
    }
}