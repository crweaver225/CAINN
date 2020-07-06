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

void Input_layer::printMetaData() {
    std::cout<<"Input layer: (1,"<<dimensions[0]<<")"<<std::endl;
}

void Input_layer::build(std::shared_ptr<Neural_Layer> previous_layer) {
    output_results = std::unique_ptr<Tensor>( new Tensor(1, dimensions.front(),dimensions.back()));
    input_array = std::unique_ptr<float>(new float[output_dimensions().back() * this->dimensions[0]]);
}

void Input_layer::addInput(float *input) {
    output_results.get()->setData(input);
}

void Input_layer::addInputInBatches(const int dimensions, float **input) {
    int input_size = output_dimensions().back();
    for (int i = 0; i < dimensions; ++i) {
        for (int k = 0; k < input_size; ++k) {
            input_array.get()[(i * input_size) + k] = input[i][k];
        }
    }
    output_results.get()->setData(input_array.get());
}

void Input_layer::setBatchDimensions(int batch_size) {
    dimensions.front() = batch_size;
    this->output_results = std::unique_ptr<Tensor>(new Tensor(batch_size, dimensions[1], dimensions[2]));
    input_array = std::unique_ptr<float>(new float[output_dimensions().back() * batch_size]);
}

