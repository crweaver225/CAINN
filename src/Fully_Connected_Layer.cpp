#include "Fully_Connected_Layer.h"

Fully_Connected_Layer::Fully_Connected_Layer(std::vector<int> dimensions, Activation_Function activation_function) : Neural_Layer(dimensions, activation_function) {
    std::cout<<"Fully connected layer constructor called"<<std::endl;
}

Fully_Connected_Layer::Fully_Connected_Layer(const Fully_Connected_Layer &fully_connected_layer) : Neural_Layer(fully_connected_layer.dimensions, fully_connected_layer.activation_function) {
    std::cout<<"Fully connected layer copy constructor called"<<std::endl;
 }

Fully_Connected_Layer& Fully_Connected_Layer::operator=(const Fully_Connected_Layer &fully_connected_layer) { 
    std::cout<<"Fully connected layer copy assignment operator called"<<std::endl;
}

Fully_Connected_Layer::Fully_Connected_Layer(Fully_Connected_Layer &&fully_connected_layer) : Neural_Layer{std::move(fully_connected_layer)} { 
    std::cout<<"Fully connected layer move constructor called"<<std::endl;
}

Fully_Connected_Layer& Fully_Connected_Layer::operator=(Fully_Connected_Layer &&fully_connected_layer) {
    std::cout<<"Fully connected layer move assignment operator called"<<std::endl;
 }

Fully_Connected_Layer::~Fully_Connected_Layer() {
    std::cout<<"Fully connected layer destructor called"<<std::endl;
}

void Fully_Connected_Layer::printMetaData() {
    std::cout<<"fully connected layer: ("<<previous_layer.get()->output_dimensions().back()<<","<<dimensions[0]<<")"<<std::endl;
}

void Fully_Connected_Layer::build(std::shared_ptr<Neural_Layer> previous_layer) {
    this->previous_layer = previous_layer;
    this->weights = std::unique_ptr<Tensor>(new Tensor(previous_layer.get()->output_dimensions().back(), dimensions[0]));
    this->weights->assignRandomValues();
    this->bias = std::unique_ptr<float>(generateBiasValues(dimensions[0]));
}

void Fully_Connected_Layer::forward_propogate() {
    output_results = std::unique_ptr<Tensor>(previous_layer_output()->matmul(*weights.get(), bias.get(), returnActivationFunction()));
}

void Fully_Connected_Layer::backpropogate() {
    gradient.get()->applyDerivative(*output_results.get(), returnActivationFunctionDerivative());
    previous_layer_gradient()->updateGradients(*gradient.get(), *weights.get());
    const float *gradient_data = gradient.get()->returnData().get();
    for (int b = 0; b < gradient.get()->shape().get()[2]; ++b) {
        bias.get()[b] += (bias.get()[b] * gradient_data[b]) * 0.001;
    }
}