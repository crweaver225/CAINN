#include "Fully_Connected_Layer.h"

Fully_Connected_Layer::Fully_Connected_Layer(std::vector<int> dimensions, Activation_Function af) : Neural_Layer(dimensions, af) {
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
    std::cout<<"fully connected layer: ("<<previous_layer.get()->output_dimensions().back()<<","<<dimensions.back()<<")"<<std::endl;
}

void Fully_Connected_Layer::build(std::shared_ptr<Neural_Layer> previous_layer) {
    std::cout<<"fully connected layer build called"<<std::endl;
    this->previous_layer = previous_layer;
    this->weights = std::unique_ptr<Tensor>(new Tensor(previous_layer.get()->output_dimensions().back(), dimensions.back()));
    this->weights->assignRandomValues();
    this->bias = std::unique_ptr<float>(generateBiasValues(dimensions.back()));
}

void Fully_Connected_Layer::forward_propogate() {
    //std::cout<<"fully connected fp beginning"<<std::endl;
    //previous_layer_output()->printShape();
    output_results = std::unique_ptr<Tensor>(previous_layer_output()->matmul(*weights.get(), bias.get(), returnActivationFunction()));
  //  std::cout<<"fully connected fp ended with result "<<std::endl;
}

void Fully_Connected_Layer::backpropogate() {
  //  std::cout<<"Fully connected layer backprop beginning"<<std::endl;
  //  std::cout<<"output results: ";
  //  output_results.get()->print();
    gradient.get()->applyDerivative(*output_results.get(), returnActivationFunctionDerivative());
    previous_layer_gradient()->updateGradients(*gradient.get(), *weights.get());
    weights->updateWeights(*gradient.get(), *previous_layer_output());
    const float *gradient_data = gradient.get()->returnData();

  //  std::cout<<"----bias values updated----"<<std::endl;
    for (int b = 0; b < gradient.get()->shape()[2]; ++b) {
        bias.get()[b] -= (bias.get()[b] * gradient_data[b]) * Tensor::learning_rate;
    //    std::cout<<bias.get()[b]<<" * "<<gradient_data[b]<<" * 0.1 = "<<(bias.get()[b] * gradient_data[b]) * 0.001<<std::endl;
    }
  //  std::cout<<"----bias values finished----"<<std::endl;
   // std::cout<<"Fully connected layer backprop ended"<<std::endl;
}