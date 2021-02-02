#include "Embedding_Layer.h"

Embedding_Layer::Embedding_Layer(std::vector<int> dimensions) : Neural_Layer(dimensions, Activation_Function::Pass) {}

Embedding_Layer& Embedding_Layer::operator=(Embedding_Layer &&embedding_layer) {
    if (this == &embedding_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(embedding_layer));
    return *this;
}

Embedding_Layer::~Embedding_Layer() {};

void Embedding_Layer::PrintMetaData()  {
    std::cout<<"embeded layer: ("<<PreviousLayerOutput()->Shape().back()<<","<<_dimensions.back()<<")"<<std::endl;
}

void Embedding_Layer::Build(std::shared_ptr<Neural_Layer> previous_layer) {
    this->_previousLayer = previous_layer;
    this->_weights = std::unique_ptr<Tensor>(new Tensor(_dimensions[2], _dimensions.back()));
    this->_weights->AssignRandomValues();
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(PreviousLayerOutput()->Shape().back(), _dimensions.back()));
    _dimensions[2] = PreviousLayerOutput()->Shape().back();
}

void Embedding_Layer::ForwardPropogate() {
    const float *embedded_weights = _weights->ReturnData();
    const float *inputData = PreviousLayerOutput()->ReturnData();
    std::vector<int> previousLayerShape = PreviousLayerOutput()->Shape();
    for (int batch = 0; batch < _dimensions.front(); batch++) {
        int batchStartingPoint = batch * previousLayerShape[2] * previousLayerShape[3];
        for (int index = 0; index < previousLayerShape[3]; index ++) {
            if (inputData[batchStartingPoint + index] == 0) {
                for (int rowValue = 0; rowValue < _dimensions.back(); rowValue++) {
                    _outputResults.get()->updateNeuron(batch, (index * _dimensions.back()) + rowValue, 0.0f);
                }
            } else {
                int current_row = inputData[index + batchStartingPoint];
                for (int rowValue = 0; rowValue < _dimensions.back(); rowValue++) {
                    _outputResults.get()->updateNeuron(batch, (index * _dimensions[3]) + rowValue, embedded_weights[(current_row * _weights.get()->Shape()[3]) + rowValue]);
                }
            }
        }
    }
}

void Embedding_Layer::Backpropogate() {
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
}

void Embedding_Layer::Training(bool train) {
    if (train) {
        this->_gradient = std::unique_ptr<Tensor>(new Tensor(this->_dimensions.front(),1, PreviousLayerOutput()->Shape().back(), _dimensions[3]));
    } else {
        _gradient.reset();
    }
}

 void Embedding_Layer::SetBatchDimensions(int batch_size) {
      _dimensions.front() = batch_size;
     this->_outputResults = std::unique_ptr<Tensor>(new Tensor(batch_size,1, PreviousLayerOutput()->Shape().back(), _dimensions.back()));
 }
