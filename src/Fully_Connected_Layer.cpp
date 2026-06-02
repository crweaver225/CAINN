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
    int active_dim = _output->ReturnActiveDimension();
    int num_outputs = _dimensions.columns;

    AdamState& biasAdam = _adamStateBias.value();

    for (int col = 0; col < num_outputs; col++) {
        float sum = 0.0f;
        for (int d = 0; d < active_dim; d++) {
            sum += gradient_data[d * num_outputs + col];
        }
        biasAdam.gradientAccumulation[col] += sum / active_dim;
    }

    biasAdam.t++;

    const float bias_correction1 = 1.0f - std::pow(biasAdam.beta1, biasAdam.t);
    const float bias_correction2 = 1.0f - std::pow(biasAdam.beta2, biasAdam.t);

    for (int col = 0; col < num_outputs; col++) {
        const float g = biasAdam.gradientAccumulation[col];

        biasAdam.m[col] = biasAdam.beta1 * biasAdam.m[col] + (1.0f - biasAdam.beta1) * g;
        biasAdam.v[col] = biasAdam.beta2 * biasAdam.v[col] + (1.0f - biasAdam.beta2) * g * g;

        const float m_hat = biasAdam.m[col] / bias_correction1;
        const float v_hat = biasAdam.v[col] / bias_correction2;

        _bias.get()[col] += Tensor::_learningRate * m_hat / (std::sqrt(v_hat) + biasAdam.epsilon);

        biasAdam.gradientAccumulation[col] = 0.0f;
    }
    
    // update weights
    _gradient->UpdateGradients(*gradient, *_weights);
    _weights->UpdateWeights(*gradient, *_input, _adamState.value());

    return _gradient.get();
}
