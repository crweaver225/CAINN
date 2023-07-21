#include "Fully_Connected_Layer.h"
#include <ctime>

Fully_Connected_Layer::Fully_Connected_Layer(Dimensions dimensions, Activation_Function af) : Neural_Layer(dimensions, af) {}

Fully_Connected_Layer::Fully_Connected_Layer(Fully_Connected_Layer &&fully_connected_layer)  noexcept : Neural_Layer{std::move(fully_connected_layer)} {}

Fully_Connected_Layer& Fully_Connected_Layer::operator=(Fully_Connected_Layer &&fully_connected_layer) noexcept {
    if (this == &fully_connected_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(fully_connected_layer));
    return *this;
 }

Fully_Connected_Layer::~Fully_Connected_Layer() {}

void Fully_Connected_Layer::PrintMetaData() {
    std::cout<<"fully connected layer: ("
            <<_previousLayer_Dimensions.columns
            <<", "
            <<_dimensions.columns
            <<")\n";
}

void Fully_Connected_Layer::Build(Neural_Layer const* previousLayer) {
    _previousLayer_Dimensions = previousLayer->ReturnDimensions();
    this->_weights = std::unique_ptr<Tensor>(new Tensor(_previousLayer_Dimensions.columns, _dimensions.columns));
    this->_weights->AssignRandomValues();
    this->_bias = std::unique_ptr<float>(GenerateBiasValues(_dimensions.columns));
    this->_output = std::make_unique<Tensor>(Tensor(1, _dimensions.columns));
}

Tensor const* Fully_Connected_Layer::ForwardPropogate(Tensor const* input) {
    _input = input;
    _output.get()->Matmul(*_input, *_weights.get(), _bias.get(), ReturnActivationFunction());
    return _output.get();
}

Tensor* Fully_Connected_Layer::Backpropogate(Tensor* gradient) {

    gradient->ApplyDerivative(*_output, ReturnActivationFunctionDerivative());

    const float *gradient_data = gradient->ReturnData();
    for (int b = 0; b < gradient->NumberOfRows(); ++b) {
        _bias.get()[b] -= (_bias.get()[b] * gradient_data[b]) * Tensor::_learningRate;
    }

    _gradient->UpdateGradients(*gradient, *_weights);

    _weights->UpdateWeights(*gradient, *_input);
    
    return _gradient.get();
}