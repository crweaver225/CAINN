#include "Neural_Layer.h"

Neural_Layer::Neural_Layer(std::vector<int> dimensions, Activation_Function activation_function) {
    this->_dimensions = dimensions;
    this->_activationFunction = activation_function;
}

Neural_Layer::~Neural_Layer() {}

Neural_Layer::Neural_Layer(Neural_Layer &&neural_layer) {
    std::cout<<"Neural Layer move constructor called"<<std::endl;
    _outputResults = std::move(neural_layer._outputResults);
    _weights = std::move(neural_layer._weights);
    _gradient = std::move(neural_layer._gradient);
    _bias = std::move(neural_layer._bias);
    _previousLayer = std::move(neural_layer._previousLayer);
    _dimensions = std::move(neural_layer._dimensions);
    _activationFunction = neural_layer._activationFunction;
}

Neural_Layer& Neural_Layer::operator=(Neural_Layer &&neural_layer) {
    std::cout<<"Neural Layer move assignment operator called"<<std::endl;
}

void Neural_Layer::PrintMetaData() {
    std::cout<<"Generic neural layer: "<<_dimensions[0]<<std::endl;
}

const std::vector<int>& Neural_Layer::OutputDimensions() {
    return _dimensions;
}

const Tensor* Neural_Layer::PreviousLayerOutput() {
    return _previousLayer.get()->_outputResults.get();
}

float* Neural_Layer::GenerateBiasValues(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1,0.1);
    float* bias_layer = new float[size];
    for (int i = 0; i < size; ++i) {
        bias_layer[i] = dis(gen);
    }
    return bias_layer;
}

void Neural_Layer::SetBias(float *data) {
    float * new_bias = new float[_dimensions.back()];
    for (int b = 0; b < _dimensions.back(); b++) {
        new_bias[b] = data[b];
    }
    this->_bias.reset(new_bias);
}

void Neural_Layer::SetLossFunction(Loss loss) {
    this->_lossFunction = loss;
}

void Neural_Layer::Training(bool train) { 
    if (train) {
        BuildGradient();
    } else {
        _gradient.reset();
    }
}

auto Neural_Layer::ReturnActivationFunction() -> void (*)(float*,float*, int, int) {
        if (_activationFunction == Activation_Function::Sigmoid) {
        return Activation_Functions::sigmoid;
    } else if (_activationFunction == Activation_Function::Relu) {
        return Activation_Functions::relu;
    } else if (_activationFunction == Activation_Function::SoftMax) {
        return Activation_Functions::softmax;
    } else {
        return Activation_Functions::pass;
    }
}

auto Neural_Layer::ReturnActivationFunctionDerivative() -> void (*)(float*, float*, int) {
    if (_activationFunction == Activation_Function::Sigmoid) {
        return Activation_Functions::sigmoid_d;
    } else if (_activationFunction == Activation_Function::Relu) {
        return Activation_Functions::relu_d;
    } else if (_activationFunction == Activation_Function::SoftMax) {
        return Activation_Functions::softmax_d;
    } else {
        return Activation_Functions::pass_d;
    }
}

void Neural_Layer::SetActiveDimensions(int batch_size) {
    this->_outputResults.get()->SetActiveDimension(batch_size);
}

void Neural_Layer::SetBatchDimensions(int batch_size) {
    _dimensions.front() = batch_size;
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(batch_size, _previousLayer.get()->_outputResults.get()->Shape()[1], _dimensions.back()));
}

const float Neural_Layer::ReturnL2() const {
    return _weights.get()->SumTheSquares();
}

void Neural_Layer::ClearGradient() {
    _gradient.get()->ResetTensor();
}

void Neural_Layer::BuildGradient() {
    this->_gradient = std::unique_ptr<Tensor>(new Tensor(this->_dimensions.front(), 1, this->_dimensions.back()));
}

Activation_Function Neural_Layer::ReturnActivationFunctionType() const {
    return _activationFunction;
}

Tensor* Neural_Layer::PreviousLayerGradient() {
    return _previousLayer.get()->_gradient.get();
}

void Neural_Layer::AddInput(float *input) {
  std::cout<<"attempting to add input to a non-input neural layer. aborting..."<<std::endl;
  exit(0);
}

void Neural_Layer::AddInputInBatches(const int dimensions, float **input) {
    std::cout<<"attempting to add input as batches to a non-input neural layer. aborting..."<<std::endl;
    exit(0);
}

void Neural_Layer::CalculateError(float **target, float regularization) {
    std::cout<<"attempting to calculate the rror from a non-output neural layer. aborting..."<<std::endl;
    exit(0);
}

void Neural_Layer::PrintError() {
    std::cout<<"attempting to print error from a non-output neural layer...aborting..."<<std::endl;
    exit(0);
}

void Neural_Layer::ResetLoss() {
    std::cout<<"Attempting to reset the loss of a non-output neural layer...aborting..."<<std::endl;
    exit(0);
}

float Neural_Layer::ReturnLoss() const {
    std::cout<<"Attempting to access the loss of a non-output neural layer...aborting..."<<std::endl;
    exit(0);
}

void Neural_Layer::PrintFinalResults() {
    std::cout<<"Final results: "<<std::endl;
    _outputResults->Print();
}
