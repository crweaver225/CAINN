#include "Output_Layer.h"

Output_Layer::Output_Layer(std::vector<int> dimensions, Activation_Function af) : Neural_Layer(dimensions, af) {}

Output_Layer::~Output_Layer() {}

Output_Layer::Output_Layer(Output_Layer &&output_layer) : Neural_Layer{std::move(output_layer)} {}

Output_Layer& Output_Layer::operator=(Output_Layer &&output_layer) {
     if (this == &output_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(output_layer));
    loss = output_layer.loss;
    error = std::move(output_layer.error);
    return *this;
}

void Output_Layer::build(std::shared_ptr<Neural_Layer> previous_layer) {
    this->previous_layer = previous_layer;
    this->weights = std::unique_ptr<Tensor>(new Tensor(previous_layer.get()->output_dimensions().back(), dimensions.back()));
    this->weights->assignRandomValues();
    this->bias = std::unique_ptr<float>(new float[dimensions.back()]);
    memset(bias.get(), 0.0f, dimensions.back() * sizeof(float));
    this->output_results = std::unique_ptr<Tensor>(new Tensor(1, previous_layer_output()->shape()[1], dimensions.back()));
}

void Output_Layer::training(bool train) {
    if (train) {
        error = std::unique_ptr<float>( new float[dimensions.front() * dimensions.back()]);
        buildGradient();
    } else {
        error.reset();
        gradient.reset();
    }
}

void Output_Layer::resetLoss() {
    this->loss = 0.0f;
    this->batches_in_iteration = 1;
}

float Output_Layer::returnLoss() const {
    return (loss / batches_in_iteration);
}

void Output_Layer::printMetaData() {
    std::cout<<"output layer: ("<<previous_layer->output_dimensions().back()<<","<<dimensions.back()<<") "<<std::endl;
}

void Output_Layer::forward_propogate() {
    output_results.get()->matmul(*previous_layer_output(), *weights.get(), bias.get(), returnActivationFunction());
}

void Output_Layer::printFinalResults() {
    std::cout<<"Final results: ";
    output_results->print();
    std::cout<<std::endl;
}

void Output_Layer::calculateError(float **target, float regularization) {
    float temp_loss = 0.0f;
    batches_in_iteration ++;
    auto lf = returnLossFunction();
    int output_size = dimensions.back();
    int dimension = dimensions.front();
    int active_dimension = output_results->returnActiveDimension();
    const float *output = output_results->returnData();
    for (int d = 0; d < active_dimension; d++) {
        int current_dimensions = d * output_size;
        temp_loss += lf(output, target[d], current_dimensions, output_size);
        for (int i_o = 0; i_o < output_size; ++ i_o) {
            error.get()[current_dimensions + i_o] = (target[d][i_o] - output[current_dimensions + i_o]) - regularization;
        }
    }
    loss += temp_loss / active_dimension;
}

auto Output_Layer::returnLossFunction() -> float (*)(const float*, float*, int, int) {
    if (loss_function == Loss::MSE) {
        return Loss_Function::mse;
    } else if (loss_function == Loss::ASE) {
        return Loss_Function::ase;
    } else {
        return Loss_Function::crossentropy;
    }
}

void Output_Layer::printError() {
    float averaged_loss = loss / batches_in_iteration;
    std::cout<<", Loss: "<<averaged_loss<<std::endl;
}
  
void Output_Layer::backpropogate() {
    gradient.get()->updateTensor(error.get());
    gradient.get()->applyDerivative(*output_results.get(), returnActivationFunctionDerivative());
    previous_layer_gradient()->updateGradients(*gradient.get(), *weights.get());
    weights->updateWeights(*gradient.get(), *previous_layer_output());
}