#include "Convolution_Layer.h"

Convolution_Layer::Convolution_Layer(int kernels, int kernel_size, int stride) : Neural_Layer({1,1,1,1}, Activation_Function::Relu) {
    _stride = stride;
    _kernels = kernels;
    _kernel_size = kernel_size;
}

void Convolution_Layer::PrintMetaData() {
    std::cout<<"convolutional layer: ("
            <<"["<<_dimensions.channels<<","<<_dimensions.rows<<","<<_dimensions.columns<<"]"
            <<std::endl;
}

void Convolution_Layer::Build(Neural_Layer const* previousLayer) {
    _previousLayer_Dimensions = previousLayer->ReturnDimensions();

    int output_depth = _kernels;
 
    int output_rows = ((_previousLayer_Dimensions.rows - _kernel_size) / _stride) + 1;
    int output_columns = ((_previousLayer_Dimensions.columns - _kernel_size) / _stride) + 1;

    _dimensions = Dimensions{ 1, output_depth, output_rows, output_columns };

    _weights = std::unique_ptr<Tensor>(new Tensor(output_depth, _previousLayer_Dimensions.channels, _kernel_size, _kernel_size));
    _weights->AssignRandomValues();

    _bias = std::unique_ptr<float>(GenerateBiasValues(output_depth));

    _output = std::make_unique<Tensor>(Tensor(output_depth, output_rows, output_columns));
}

void Convolution_Layer::Training(bool train) {
    if (train) {
        BuildGradient();
        _output->optimizeForTraining();
        _adamState.emplace(_weights->NumberOfElements());
        _adamStateBias.emplace(_dimensions.channels);
    } else {
        _gradient.reset();
        _output->optimizeForInference();
        _adamState.reset();
        _adamStateBias.reset();
    }
}

Tensor const* Convolution_Layer::ForwardPropogate(Tensor const* input) {
    _input = input;
    _output.get()->Convolve(*_input, *_weights, _stride, _bias.get());
    return _output.get();
}

Tensor* Convolution_Layer::Backpropogate(Tensor* gradient) {
    gradient->ApplyDerivative(*_output, ReturnActivationFunctionDerivative());

    AdamState& biasAdam = _adamStateBias.value();
    const int active_dimension = _output->ReturnActiveDimension();
    const float *gradient_data = gradient->ReturnData();

    for (int oc = 0; oc < _dimensions.channels; oc++) {
        float sum = 0.0f;
        for (int b = 0; b < active_dimension; b++) {
            for (int y = 0; y < _dimensions.rows; y++) {
                for (int x = 0; x < _dimensions.columns; x++) {
                    sum += gradient->valueAt(b,oc,y,x);
                }
            }
        }
        biasAdam.gradientAccumulation[oc] = sum / active_dimension;
    }

    biasAdam.t ++;
    const float bias_correction1 = 1.0f - std::pow(biasAdam.beta1, biasAdam.t);
    const float bias_correction2 = 1.0f - std::pow(biasAdam.beta2, biasAdam.t);

    for (int oc = 0; oc < _dimensions.channels; ++oc) {
        const float g = biasAdam.gradientAccumulation[oc];

        biasAdam.m[oc] = biasAdam.beta1 * biasAdam.m[oc] + (1.0f - biasAdam.beta1) * g;

        biasAdam.v[oc] = biasAdam.beta2 * biasAdam.v[oc] + (1.0f - biasAdam.beta2) * g * g;

        const float m_hat = biasAdam.m[oc] / bias_correction1;
        const float v_hat = biasAdam.v[oc] / bias_correction2;

        _bias.get()[oc] -= Tensor::_learningRate * m_hat / (std::sqrt(v_hat) + biasAdam.epsilon);

        biasAdam.gradientAccumulation[oc] = 0.0f;
    }
        
    _gradient.get()->Backward(*gradient, *_weights, _stride);
    _weights->UpdateKernel(*_input, *gradient, _stride, _adamState.value());
    return _gradient.get();
}

void Convolution_Layer::SetBatchDimensions(int batch_size) {
    _dimensions.dimensions = batch_size;
    _previousLayer_Dimensions.dimensions = batch_size;
    _output = std::unique_ptr<Tensor>(new Tensor(batch_size,_dimensions.channels, _dimensions.rows, _dimensions.columns));
}


