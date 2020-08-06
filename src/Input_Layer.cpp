#include "Input_Layer.h"

Input_layer::Input_layer(std::vector<int> dimensions) : Neural_Layer(dimensions, Activation_Function::Pass) {}

Input_layer::~Input_layer() {}

Input_layer& Input_layer::operator=(Input_layer &&input_layer) {
    if (this == &input_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(input_layer));
    return *this;
}

void Input_layer::PrintMetaData() {
    std::cout<<"Input layer: (1,"<<_dimensions[0]<<")"<<std::endl;
}

void Input_layer::Build(std::shared_ptr<Neural_Layer> previous_layer) {
    _outputResults = std::unique_ptr<Tensor>( new Tensor(1, _dimensions.front(),_dimensions.back()));
    _inputArray = std::unique_ptr<float>(new float[OutputDimensions().back() * this->_dimensions[0]]);
}

void Input_layer::AddInput(float *input) {
    _outputResults.get()->SetData(input);
}

void Input_layer::AddInputInBatches(const int dimensions, float **input) {
    int input_size = OutputDimensions().back();
    for (int i = 0; i < dimensions; ++i) {
        for (int k = 0; k < input_size; ++k) {
            _inputArray.get()[(i * input_size) + k] = input[i][k];
        }
    }
    _outputResults.get()->SetData(_inputArray.get());
}

void Input_layer::SetBatchDimensions(int batch_size) {
    _dimensions.front() = batch_size;
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(batch_size, _dimensions[1], _dimensions[2]));
    _inputArray = std::unique_ptr<float>(new float[OutputDimensions().back() * batch_size]);
}

