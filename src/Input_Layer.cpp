#include "Input_Layer.h"

Input_layer::Input_layer(Dimensions dimensions) : Neural_Layer(dimensions, Activation_Function::Pass) {}

Input_layer::~Input_layer() {}

Input_layer::Input_layer(Input_layer &&input_layer)  noexcept : Neural_Layer{std::move(input_layer)} {}

Input_layer& Input_layer::operator=(Input_layer &&input_layer) noexcept  {
    if (this == &input_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(input_layer));
    return *this;
}

void Input_layer::PrintMetaData() {
    std::cout<<"Input layer: ("<<_dimensions.channels<<","<<_dimensions.rows<<","<<_dimensions.columns<<")"<<std::endl;
}

void Input_layer::Build(Neural_Layer const *previousLayer) {
    _previousLayer_Dimensions = previousLayer->ReturnDimensions();
    _output = std::make_unique<Tensor>(Tensor(_dimensions.channels, _dimensions.rows,_dimensions.columns));
    _inputArray = std::unique_ptr<float>(new float[_dimensions.channels * _dimensions.rows * _dimensions.columns]);
}

Tensor const* Input_layer::ForwardPropogate(Tensor const* input) {
    return input;
}

Tensor const* Input_layer::AddInput(float *input) {
    _output.get()->SetData(input);
    return _output.get();
}

Tensor const* Input_layer::AddInputInBatches(const int dimensions, float **input) {
   int input_size = _dimensions.channels * _dimensions.rows * _dimensions.columns;
   for (int i = 0; i < dimensions; ++i) {
       for (int k = 0; k < input_size; ++k) {
           _inputArray.get()[(i * input_size) + k] = input[i][k];
       }
   }
    _output.get()->SetData(_inputArray.get());
    return _output.get();
}

void Input_layer::SetBatchDimensions(int batch_size) {
    _dimensions.dimensions = batch_size;
    _previousLayer_Dimensions.dimensions = batch_size;
    _output = std::make_unique<Tensor>(Tensor(batch_size, _dimensions.channels, _dimensions.rows, _dimensions.columns));
    _inputArray = std::unique_ptr<float>(new float[batch_size * _dimensions.channels * _dimensions.rows * _dimensions.columns]);
}
