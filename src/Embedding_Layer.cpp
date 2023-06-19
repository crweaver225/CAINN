#include "Embedding_Layer.h"

Embedding_Layer::Embedding_Layer(Dimensions dimensions) : Neural_Layer(dimensions, Activation_Function::Pass) {}

Embedding_Layer::Embedding_Layer(Embedding_Layer &&embedding_layer) noexcept  : Neural_Layer{std::move(embedding_layer)} { }

Embedding_Layer& Embedding_Layer::operator=(Embedding_Layer &&embedding_layer) noexcept {
    if (this == &embedding_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(embedding_layer));
    return *this;
}

Embedding_Layer::~Embedding_Layer() {};

void Embedding_Layer::PrintMetaData()  {
    std::cout<<"embeded layer: ("
                <<_dimensions.rows
                <<","
                <<_dimensions.columns
                <<")\n";
}

void Embedding_Layer::Build(Neural_Layer const* previousLayer) {
    _previousLayer_Dimensions = previousLayer->ReturnDimensions();
    this->_weights = std::unique_ptr<Tensor>(new Tensor(_dimensions.rows, _dimensions.columns));
    this->_weights->AssignRandomValues();
    _output = std::make_unique<Tensor>(Tensor(_previousLayer_Dimensions.columns, _dimensions.columns));
    _dimensions.rows = _previousLayer_Dimensions.columns;
}

Tensor const* Embedding_Layer::ForwardPropogate(Tensor const* input){
    _input = input;
    const float *embedded_weights = _weights->ReturnData();
    const float *inputData = input->ReturnData();
    Dimensions previousLayerShape = input->dimensions();
    for (int batch = 0; batch < _dimensions.dimensions; batch++) {
        int batchStartingPoint = batch * previousLayerShape.rows * previousLayerShape.columns;
        for (int index = 0; index < previousLayerShape.columns; index ++) {
            if (inputData[batchStartingPoint + index] == 0) {
                for (int rowValue = 0; rowValue < _dimensions.columns; rowValue++) {
                    _output->updateNeuron(batch, (index * _dimensions.columns) + rowValue, 0.0f);
                }
            } else {
                int current_row = inputData[index + batchStartingPoint];
                for (int rowValue = 0; rowValue < _dimensions.columns; rowValue++) {
                    _output->updateNeuron(batch, (index * _dimensions.columns) + rowValue, embedded_weights[(current_row * _weights.get()->NumberOfColumns()) + rowValue]);
                }
            }
        }
    }
    return _output.get();
}

Tensor* Embedding_Layer::Backpropogate(Tensor* gradient){
    const float *inputData = _input->ReturnData();
    const float *gradientData = gradient->ReturnData();
    for (int batch = 0; batch < _dimensions.dimensions; batch++) { 
        for (int index = 0; index < _dimensions.rows; index ++) { 
            int gradientIndex = ((batch * (_dimensions.columns * index)) + (_dimensions.columns * index));
            for (int i = 0; i < _dimensions.columns; i++) { 
                int input_value = inputData[(_dimensions.rows * batch) + index];
                if (input_value != 0) {
                    int current_weight_index = (_weights.get()->NumberOfColumns() * input_value) + i;
                    float current_weight_value = _weights.get()->ReturnData()[current_weight_index];
                    _weights.get()->updateNeuron(current_weight_index, current_weight_value + ((gradientData[gradientIndex + i] * _weights.get()->_learningRate)));
                }
            }
        }
    }
    return _gradient.get();
    /*
    const float *inputData = PreviousLayerOutput()->ReturnData(); 
    const float *gradientData = _gradient.get()->ReturnData();
    for (int batch = 0; batch < _dimensions.front(); batch++) { 
        for (int index = 0; index < _dimensions[2]; index ++) { 
            int gradientIndex = ((batch * (_dimensions[3] * index)) + (_dimensions[3] * index));
            for (int i = 0; i < _dimensions[3]; i++) { 
                int input_value = inputData[(_dimensions[2] * batch) + index];
                if (input_value != 0) {
                    int current_weight_index = (_weights.get()->Shape()[3] * input_value) + i;
                    float current_weight_value = _weights.get()->ReturnData()[current_weight_index];
                    _weights.get()->updateNeuron(current_weight_index, current_weight_value + ((gradientData[gradientIndex + i] * _weights.get()->_learningRate)));
                }
            }
        }
    }
    */
}

void Embedding_Layer::Training(bool train) {
    if (train) {
        this->_gradient = std::unique_ptr<Tensor>(new Tensor(this->_dimensions.dimensions, 1, _dimensions.rows, _dimensions.columns));
    } else {
        _gradient.reset();
    }
}

 void Embedding_Layer::SetBatchDimensions(int batch_size) {
    _dimensions.dimensions = batch_size;
    _output = std::unique_ptr<Tensor>(new Tensor(batch_size,1, _dimensions.rows, _dimensions.columns));
 }
