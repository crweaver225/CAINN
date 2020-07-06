#include "Neural_Layer.h"

Neural_Layer::Neural_Layer(std::vector<int> dimensions, Activation_Function activation_function) {
    this->dimensions = dimensions;
    this->activation_function = activation_function;
}

Neural_Layer::~Neural_Layer() {}

Neural_Layer::Neural_Layer(Neural_Layer &&neural_layer) {
    std::cout<<"Neural Layer move constructor called"<<std::endl;
    output_results = std::move(neural_layer.output_results);
    weights = std::move(neural_layer.weights);
    gradient = std::move(neural_layer.gradient);
    bias = std::move(neural_layer.bias);
    previous_layer = std::move(neural_layer.previous_layer);
    dimensions = std::move(neural_layer.dimensions);
    activation_function = neural_layer.activation_function;
}

Neural_Layer& Neural_Layer::operator=(Neural_Layer &&neural_layer) {
    std::cout<<"Neural Layer move assignment operator called"<<std::endl;
}

void Neural_Layer::printMetaData() {
    std::cout<<"Generic neural layer: "<<dimensions[0]<<std::endl;
}

const std::vector<int>& Neural_Layer::output_dimensions() {
    return dimensions;
}

const Tensor* Neural_Layer::previous_layer_output() {
    return previous_layer.get()->output_results.get();
}

float* Neural_Layer::generateBiasValues(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1,0.1);
    float* bias_layer = new float[size];
    for (int i = 0; i < size; ++i) {
        bias_layer[i] = dis(gen);
    }
    return bias_layer;
}

void Neural_Layer::setBias(float *data) {
    float * new_bias = new float[dimensions.back()];
    for (int b = 0; b < dimensions.back(); b++) {
        new_bias[b] = data[b];
    }
    this->bias.reset(new_bias);
}

void Neural_Layer::setLossFunction(Loss loss) {
    this->loss_function = loss;
}

void Neural_Layer::training(bool train) { 
    if (train) {
        buildGradient();
    } else {
        gradient.reset();
    }
}

auto Neural_Layer::returnActivationFunction() -> void (*)(float*,float*, int, int) {
        if (activation_function == Activation_Function::Sigmoid) {
        return Activation_Functions::sigmoid;
    } else if (activation_function == Activation_Function::Relu) {
        return Activation_Functions::relu;
    } else if (activation_function == Activation_Function::SoftMax) {
        return Activation_Functions::softmax;
    } else {
        return Activation_Functions::pass;
    }
}

auto Neural_Layer::returnActivationFunctionDerivative() -> void (*)(float*, float*, int) {
    if (activation_function == Activation_Function::Sigmoid) {
        return Activation_Functions::sigmoid_d;
    } else if (activation_function == Activation_Function::Relu) {
        return Activation_Functions::relu_d;
    } else if (activation_function == Activation_Function::SoftMax) {
        return Activation_Functions::softmax_d;
    } else {
        return Activation_Functions::pass_d;
    }
}

void Neural_Layer::setActiveDimensions(int batch_size) {
    this->output_results.get()->setActiveDimension(batch_size);
}

void Neural_Layer::setBatchDimensions(int batch_size) {
    dimensions.front() = batch_size;
    this->output_results = std::unique_ptr<Tensor>(new Tensor(batch_size, previous_layer.get()->output_results.get()->shape()[1], dimensions.back()));
}

const float Neural_Layer::returnL2() const {
    return weights.get()->sumTheSquares();
}

void Neural_Layer::clearGradient() {
    gradient.get()->resetTensor();
}

void Neural_Layer::buildGradient() {
    this->gradient = std::unique_ptr<Tensor>(new Tensor(this->dimensions.front(), 1, this->dimensions.back()));
}

Activation_Function Neural_Layer::returnActivationFunctionType() const {
    return activation_function;
}

Tensor* Neural_Layer::previous_layer_gradient() {
    return previous_layer.get()->gradient.get();
}

void Neural_Layer::addInput(float *input) {
  std::cout<<"attempting to add input to a non-input neural layer. aborting..."<<std::endl;
  exit(0);
}

void Neural_Layer::addInputInBatches(const int dimensions, float **input) {
    std::cout<<"attempting to add input as batches to a non-input neural layer. aborting..."<<std::endl;
    exit(0);
}

void Neural_Layer::calculateError(float **target, float regularization) {
    std::cout<<"attempting to calculate the rror from a non-output neural layer. aborting..."<<std::endl;
    exit(0);
}

void Neural_Layer::printError() {
    std::cout<<"attempting to print error from a non-output neural layer...aborting..."<<std::endl;
    exit(0);
}

void Neural_Layer::resetLoss() {
    std::cout<<"Attempting to reset the loss of a non-output neural layer...aborting..."<<std::endl;
    exit(0);
}

float Neural_Layer::returnLoss() const {
    std::cout<<"Attempting to access the loss of a non-output neural layer...aborting..."<<std::endl;
    exit(0);
}

void Neural_Layer::printFinalResults() {
    std::cout<<"Final results: "<<std::endl;
    output_results->print();
}
