#include "Output_Layer.h"

Output_Layer::Output_Layer(std::vector<int> dimensions, Activation_Function af) : Neural_Layer(dimensions, af) {}

Output_Layer::~Output_Layer() {}

Output_Layer::Output_Layer(Output_Layer &&output_layer) : Neural_Layer{std::move(output_layer)} {}

Output_Layer& Output_Layer::operator=(Output_Layer &&output_layer) {
     if (this == &output_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(output_layer));
    _loss = output_layer._loss;
    _error = std::move(output_layer._error);
    return *this;
}

void Output_Layer::Build(std::shared_ptr<Neural_Layer> previous_layer) {
    this->_previousLayer = previous_layer;
    this->_weights = std::unique_ptr<Tensor>(new Tensor(previous_layer.get()->OutputDimensions().back(), _dimensions.back()));
    this->_weights->AssignRandomValues();
    this->_bias = std::unique_ptr<float>(new float[_dimensions.back()]);
    memset(_bias.get(), 0.0f, _dimensions.back() * sizeof(float));
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(1, PreviousLayerOutput()->Shape()[1],PreviousLayerOutput()->Shape()[2], _dimensions.back()));
}

void Output_Layer::Training(bool train) {
    if (train) {
        _error = std::unique_ptr<float>( new float[_dimensions.front() * _dimensions.back()]);
        BuildGradient();
    } else {
        _error.reset();
        _gradient.reset();
    }
}

void Output_Layer::ResetLoss() {
    this->_loss = 0.0f;
    this->_batchesInIteration = 1;
}

float Output_Layer::ReturnLoss() const {
    return (_loss / _batchesInIteration);
}

void Output_Layer::PrintMetaData() {
    std::cout<<"output layer: ("<<_previousLayer->OutputDimensions().back()<<","<<_dimensions.back()<<") "<<std::endl;
}

void Output_Layer::ForwardPropogate() {
    _outputResults.get()->Matmul(*PreviousLayerOutput(), *_weights.get(), _bias.get(), ReturnActivationFunction());
}

void Output_Layer::PrintFinalResults() {
    std::cout<<"Final results: ";
    _outputResults->Print();
    std::cout<<std::endl;
}

void Output_Layer::CalculateError(float **target, float regularization) {
    float temp_loss = 0.0f;
    _batchesInIteration ++;
    auto lf = ReturnLossFunction();
    int output_size =_dimensions.back();
    int dimension = _dimensions.front();
    int active_dimension = _outputResults->ReturnActiveDimension();
    const float *output = _outputResults->ReturnData();
    for (int d = 0; d < active_dimension; d++) {
        int current_dimensions = d * output_size;
        temp_loss += lf(output, target[d], current_dimensions, output_size);
        for (int i_o = 0; i_o < output_size; ++ i_o) {
            _error.get()[current_dimensions + i_o] = (target[d][i_o] - output[current_dimensions + i_o]);// - regularization;
        }
    }
    _loss += temp_loss / (float)active_dimension;
}

auto Output_Layer::ReturnLossFunction() -> float (*)(const float*, float*, int, int) {
    if (_lossFunction == Loss::MSE) {
        return Loss_Function::mse;
    } else if (_lossFunction == Loss::ASE) {
        return Loss_Function::ase;
    } else {
        return Loss_Function::crossentropy;
    }
}

void Output_Layer::PrintError() {
    float averaged_loss = _loss / _batchesInIteration;
    std::cout<<", Loss: "<<averaged_loss<<std::endl;
}
  
void Output_Layer::Backpropogate() {
    _gradient.get()->UpdateTensor(_error.get());
    _gradient.get()->ApplyDerivative(*_outputResults.get(), ReturnActivationFunctionDerivative());
    PreviousLayerGradient()->UpdateGradients(*_gradient.get(), *_weights.get());
    _weights->UpdateWeights(*_gradient.get(), *PreviousLayerOutput());
}