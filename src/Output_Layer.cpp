#include "Output_Layer.h"

Output_Layer::Output_Layer(Dimensions dimensions, Activation_Function af) : Neural_Layer(dimensions, af) {}

Output_Layer::~Output_Layer() {}

Output_Layer::Output_Layer(Output_Layer &&output_layer) noexcept  : Neural_Layer{std::move(output_layer)} {}

Output_Layer& Output_Layer::operator=(Output_Layer &&output_layer) noexcept  {
     if (this == &output_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(output_layer));
    _loss = output_layer._loss;
    _error = std::move(output_layer._error);
    return *this;
}

void Output_Layer::Build(Neural_Layer const* previousLayer) {
    _previousLayer_Dimensions = previousLayer->ReturnDimensions();
    this->_weights = std::unique_ptr<Tensor>(new Tensor(
                                                    _previousLayer_Dimensions.columns,
                                                    _dimensions.columns
                                                    ));
    this->_weights->AssignRandomValues();
    this->_bias = std::unique_ptr<float>(new float[_dimensions.columns]);
    memset(_bias.get(), 0.0f, _dimensions.columns * sizeof(float));
    _output = std::make_unique<Tensor>(Tensor(
                                        _dimensions.dimensions, 
                                        _previousLayer_Dimensions.channels, 
                                        _previousLayer_Dimensions.rows,
                                        _dimensions.columns
                                        ));
}

void Output_Layer::Training(bool train) {
    if (train) {
        _error = std::make_unique<Tensor>(Tensor(_dimensions.dimensions,1,1,_dimensions.columns));
        BuildGradient();
    } else {
        _error.reset();
        _gradient.reset();
    }
}

void Output_Layer::SetLossFunction(Loss loss) {
    this->_lossFunction = loss;
}

void Output_Layer::ResetLoss() {
    this->_loss = 0.0f;
    this->_batchesInIteration = 1;
}

float Output_Layer::ReturnLoss() const {
    return (_loss / _batchesInIteration);
}

void Output_Layer::PrintMetaData() {
    std::cout<<"output layer: ("
                <<_previousLayer_Dimensions.columns
                <<","<<_dimensions.columns
                <<")\n";
}

Tensor const* Output_Layer::ForwardPropogate(Tensor const* input) {
    _input = input;
    _output->Matmul(*_input, *_weights.get(), _bias.get(), ReturnActivationFunction());
    return _output.get();
}

void Output_Layer::PrintFinalResults() {
    std::cout<<"Final results: ";
    _output.get()->Print();
    std::cout<<std::endl;
}

void Output_Layer::SetActiveDimensions(int batch_size) {
    this->_output->SetActiveDimension(batch_size);
    this->_error->SetActiveDimension(batch_size);
}

void Output_Layer::CalculateError(float **target, float regularization) {
    float temp_loss = 0.0f;
    _batchesInIteration ++;
    auto lf = ReturnLossFunction();
    int output_size =_dimensions.columns;
    int active_dimension = _output->ReturnActiveDimension();
    const float *output = _output->ReturnData();
    
    for (int d = 0; d < active_dimension; d++) {
        int current_dimensions = d * output_size;
        temp_loss += lf(output, target[d], current_dimensions, output_size);
        for (int i_o = 0; i_o < output_size; ++ i_o) {
           _error->setNeuron(d, i_o, (target[d][i_o] - output[current_dimensions + i_o]) + regularization);
        }
    }
    _loss += temp_loss / (float)active_dimension;
}

Tensor * Output_Layer::ReturnError() const {
    return _error.get();
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
    std::cout <<", Loss: "
            << _loss / _batchesInIteration
            <<"\n";
}
  
Tensor* Output_Layer::Backpropogate(Tensor* gradient) {

    gradient->ApplyDerivative(*_output, ReturnActivationFunctionDerivative());
    _gradient->UpdateGradients(*gradient, *_weights);
    _weights->UpdateWeights(*gradient, *_input);

    return _gradient.get();
}