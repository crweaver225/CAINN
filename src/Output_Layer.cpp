#include "Output_Layer.h"

Output_Layer::Output_Layer(std::vector<int> dimensions, Activation_Function activiation_function) : Neural_Layer(dimensions, activation_function) {
    std::cout<<"output layer constructor called"<<std::endl;

}

Output_Layer::~Output_Layer() {
    std::cout<<"output layer deconstructor called"<<std::endl;

}

Output_Layer::Output_Layer(const Output_Layer &output_layer) : Neural_Layer(output_layer.dimensions, output_layer.activation_function) {
    std::cout<<"output layer copy constructor called"<<std::endl;

}

Output_Layer& Output_Layer::operator=(const Output_Layer &output_layer) {
    std::cout<<"output layer copy assignment operator called"<<std::endl;

}

Output_Layer::Output_Layer(Output_Layer &&output_layer) : Neural_Layer{std::move(output_layer)}  {
    std::cout<<"output layer move constructor called"<<std::endl;

}

Output_Layer& Output_Layer::operator=(Output_Layer &&output_layer) {
    std::cout<<"output layer move assigment operator called"<<std::endl;

}

void Output_Layer::build(std::shared_ptr<Neural_Layer> previous_layer) {
    std::cout<<"Output layer build called"<<std::endl;
    this->previous_layer = previous_layer;
    this->weights = std::unique_ptr<Tensor>(new Tensor(previous_layer.get()->output_dimensions().back(), dimensions[0]));
    this->weights->assignRandomValues();
    this->bias = std::unique_ptr<float>(new float[dimensions[0]]);
    memset(bias.get(), 0.0f, dimensions[0]);
}

void Output_Layer::printMetaData() {
    std::cout<<"output layer: ("<<previous_layer->output_dimensions().back()<<","<<dimensions[0]<<") "<<std::endl;
}
   
void Output_Layer::forward_propogate() {
    output_results = std::unique_ptr<Tensor>(previous_layer_output()->matmul(*weights.get(), bias.get(), returnActivationFunction()));
}

void Output_Layer::printFinalResults() {
    std::cout<<"Final results: ";
    output_results->print();
    std::cout<<std::endl;
}

void Output_Layer::calculateError(float **target, float regularization) {
    loss = 0.0f;
    int output_size = output_dimensions().back();
    int dimensions = output_results.get()->shape().get()[0];
    error = std::unique_ptr<float>( new float[output_size * dimensions]);
    std::shared_ptr<float> output = output_results.get()->returnData();
    for (int d = 0; d < dimensions; ++d) {
        int current_dimensions = d * output_size;
        for (int i_o = 0; i_o < output_size; ++ i_o) {
            loss += abs(target[d][i_o] - output.get()[dimensions]);
            error.get()[current_dimensions + i_o] = target[d][i_o] - output.get()[dimensions];
        }
    }
    loss = loss / dimensions;
}

void Output_Layer::printError() {
    int output_size = output_dimensions().back();
    std::shared_ptr<float> output = output_results.get()->returnData();
    std::cout<<"Outcome: ";
    for (int i = 0; i < output_size; ++i) {
        std::cout<<output.get()[i]<< " ";
    }
    std::cout<<", Error: ";
    for (int i = 0; i < output_size; ++i) {
        std::cout<<error.get()[i]<<" ";
    }
    std::cout<<", Loss: ";
    for (int i = 0; i < output_size; ++i) {
        std::cout<<loss<<" ";
    }
    std::cout<<std::endl;
}
  
void Output_Layer::backpropogate() {
    gradient.get()->updateTensor(error.get());
    previous_layer_gradient()->updateGradients(*gradient.get(), *weights.get());
    weights->updateWeights(*gradient.get(), *output_results.get());
}