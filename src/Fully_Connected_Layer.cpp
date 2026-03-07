#include "Fully_Connected_Layer.h"
#include <ctime>

Fully_Connected_Layer::Fully_Connected_Layer(Dimensions dimensions, Activation_Function af) : Neural_Layer(dimensions, af) {}

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

void Fully_Connected_Layer::Training(bool train) {
    if (train) {
        BuildGradient();
        _output->optimizeForTraining();
        _adamState.emplace(_weights->NumberOfElements());
        _adamStateBias.emplace(_dimensions.columns);
    } else {
        _gradient.reset();
        _output->optimizeForInference();
        _adamState.reset();
        _adamStateBias.reset();
    }
}

Tensor const* Fully_Connected_Layer::ForwardPropogate(Tensor const* input) {
    _input = input;
    _output.get()->Matmul(*_input, *_weights.get(), _bias.get(), ReturnActivationFunction());
    return _output.get();
}

Tensor* Fully_Connected_Layer::Backpropogate(Tensor* gradient) {

    gradient->ApplyDerivative(*_output, ReturnActivationFunctionDerivative());

    const float *gradient_data = gradient->ReturnData();
    int batch_size = _output->ReturnActiveDimension();
    
    AdamState& biasAdam = _adamStateBias.value();
    
    for (int b = 0; b < gradient->NumberOfRows(); ++b) {
        biasAdam.gradientAccumulation[b] += gradient_data[b] / batch_size; // average over batch
    }

    biasAdam.t++;

   const float bias_correction1 = 1.0f - std::pow(biasAdam.beta1, biasAdam.t);
   const float bias_correction2 = 1.0f - std::pow(biasAdam.beta2, biasAdam.t);

    for (int b = 0; b < gradient->NumberOfRows(); ++b) {
        const float g = biasAdam.gradientAccumulation[b]; // averaged gradient

        biasAdam.m[b] = biasAdam.beta1 * biasAdam.m[b] + (1.0f - biasAdam.beta1) * g;
        biasAdam.v[b] = biasAdam.beta2 * biasAdam.v[b] + (1.0f - biasAdam.beta2) * g * g;

        const float m_hat = biasAdam.m[b] / bias_correction1;
        const float v_hat = biasAdam.v[b] / bias_correction2;

       _bias.get()[b] += Tensor::_learningRate * m_hat / (std::sqrt(v_hat) + biasAdam.epsilon);

       biasAdam.gradientAccumulation[b] = 0.0f; // reset
    }
    
    // update weights
    _gradient->UpdateGradients(*gradient, *_weights);
    _weights->UpdateWeights(*gradient, *_input, _adamState.value());

    return _gradient.get();
}
