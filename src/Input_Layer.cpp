#include "Input_Layer.h"

Input_layer::Input_layer(std::vector<int> dimensions) : Neural_Layer(dimensions, Activation_Function::Pass) {
    std::cout<<"Input layer constructor called"<<std::endl;
}

Input_layer::~Input_layer() {
    std::cout<<"Input layer deconstructor called"<<std::endl;
}

Input_layer::Input_layer(const Input_layer &input_layer) : Neural_Layer( input_layer.dimensions, input_layer.activation_function) {
    std::cout<<"Input layer copy constructor called"<<std::endl;
}

Input_layer& Input_layer::operator=(const Input_layer &input_layer) {
    std::cout<<"Input layer copy assignment operator called"<<std::endl;
}

Input_layer::Input_layer(Input_layer &&input_layer) : Neural_Layer{std::move(input_layer)}{
    std::cout<<"Input layer move constructor called"<<std::endl;
}

Input_layer& Input_layer::operator=(Input_layer &&input_layer) {
    std::cout<<"Input layer move assignment operator called"<<std::endl;
}

void Input_layer::printMetaData() {
    std::cout<<"Input layer: (1,"<<dimensions[0]<<")"<<std::endl;
}

void Input_layer::build(std::shared_ptr<Neural_Layer> previous_layer) {
  //  std::cout<<"Input layer build called"<<std::endl;
    output_results = std::unique_ptr<Tensor>( new Tensor(1, dimensions.front(),dimensions.back()));
}

void Input_layer::addInput(float *input) {
    output_results.get()->setData(input);
}

void Input_layer::addInputInBatches(const int dimensions, float **input) {
    int input_size = output_dimensions().back();
    float *input_array = new float[this->dimensions[0] * input_size];
    memset(input_array, 0.0f, this->dimensions[0] * input_size * sizeof(float));
    for (int i = 0; i < dimensions; ++i) {
        for (int k = 0; k < input_size; ++k) {
            input_array[(i * input_size) + k] = input[i][k];
        }
    }
    output_results.get()->setData(input_array);
}

void Input_layer::setBatchDimensions(int batch_size) {
    dimensions.front() = batch_size;
    this->output_results = std::unique_ptr<Tensor>(new Tensor(batch_size, dimensions[1], dimensions[2]));
}

