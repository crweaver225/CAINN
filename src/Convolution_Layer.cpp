#include "Convolution_Layer.h"

Convolution_Layer::Convolution_Layer(int kernels, int kernel_size, int stride) : Neural_Layer({1,1,1,1}, Activation_Function::Relu) {
    _stride = stride;
    _kernels = kernels;
    _kernel_size = kernel_size;
}

Convolution_Layer::Convolution_Layer(Convolution_Layer &&convolution_layer)  noexcept : Neural_Layer{std::move(convolution_layer)} {
    this->_stride = convolution_layer._stride;
    this->_kernels = convolution_layer._kernels;
    this->_kernel_size = convolution_layer._kernel_size;
}

Convolution_Layer& Convolution_Layer::operator=(Convolution_Layer &&convolution_layer) noexcept {
    if (this == &convolution_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(convolution_layer));
    return *this;
 }

Convolution_Layer::~Convolution_Layer() {}

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
    
    this->_weights = std::unique_ptr<Tensor>(new Tensor(output_depth, _previousLayer_Dimensions.channels, _kernel_size, _kernel_size));
    this->_weights->AssignRandomValues();
    this->_output = std::make_unique<Tensor>(Tensor(output_depth, output_rows, output_columns));
}

Tensor const* Convolution_Layer::ForwardPropogate(Tensor const* input) {
    _input = input;
    _output.get()->Convolve(*_input, *_weights, _stride);
    return _output.get();
}

Tensor* Convolution_Layer::Backpropogate(Tensor* gradient) {
    gradient->ApplyDerivative(*_output, ReturnActivationFunctionDerivative());
    _weights->UpdateKernel(*_input, *gradient, _stride);
    _gradient.get()->Backward(*gradient, *_weights, _stride);
    return _gradient.get();
}

void Convolution_Layer::SetBatchDimensions(int batch_size) {
    _dimensions.dimensions = batch_size;
    _output = std::unique_ptr<Tensor>(new Tensor(batch_size,_dimensions.channels, _dimensions.rows, _dimensions.columns));
}
/*
void Convolution_Layer::Training(bool train)  {
     if (train) {
        this->_gradient = std::make_unique<Tensor>(Tensor(_dimensions.dimensions,_dimensions.channels, _dimensions.rows, _dimensions.columns));
    } else {
        this->_gradient.reset();
    }
}
*/
