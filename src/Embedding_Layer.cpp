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
    this->_weights = std::unique_ptr<Tensor>(new Tensor(_dimensions[1], _dimensions.back()));
    this->_weights->AssignRandomValues();
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(1, PreviousLayerOutput()->Shape().back(), _dimensions.back()));
    _dimensions[1] = PreviousLayerOutput()->Shape().back();
}

void Embedding_Layer::ForwardPropogate() {
    const float *inputData = PreviousLayerOutput()->ReturnData();
    for (int batch = 0; batch < _dimensions.front(); batch++) {
        int batchStartingPoint = batch * PreviousLayerOutput()->Shape()[1] * PreviousLayerOutput()->Shape()[2];
        for (int index = 0; index < PreviousLayerOutput()->Shape().back(); index ++) {
            if (inputData[index + batchStartingPoint] == 0) {
                float *rowData = new float[_dimensions.back()];
                memset(rowData, 0.0f, _dimensions.back() * sizeof(float));
                for (int rowValue = 0; rowValue < _dimensions.back(); rowValue++) {
                    _outputResults.get()->updateNeuron(batch, (index * _dimensions.back()) + rowValue, rowData[rowValue]);
                }
                delete [] rowData;
            } else {
                const float *rowData = _weights.get()->returnRow(inputData[index + batchStartingPoint]);
                for (int rowValue = 0; rowValue < _dimensions.back(); rowValue++) {
                    _outputResults.get()->updateNeuron(batch, (index * _dimensions.back()) + rowValue, rowData[rowValue]);
                }
                delete [] rowData;
            }
        }
    }
}

void Embedding_Layer::Backpropogate() {
    // weights will be 80000,300
    // output will be 2,4,300
    // gradient shape will be 2,4,300
    const float *inputData = PreviousLayerOutput()->ReturnData(); // 10,1,4 data source
    for (int batch = 0; batch < _dimensions.front(); batch++) { // loop through 10 batches of input data
        for (int index = 0; index < _dimensions[1]; index ++) { // loop through all four columns of input
            const float *gradientRowData = _gradient.get()->returnRow(batch, index); // get gradient row, 300 values
            for (int i = 0; i < _dimensions[2]; i++) { // loop through all columns, 300 values
                int input_value = inputData[(_dimensions[1] * batch) + index];
                if (input_value != 0) {
                    int current_weight_index = (_weights.get()->Shape()[2] * input_value) + i;
                    float current_weight_value = _weights.get()->ReturnData()[current_weight_index];
                    _weights.get()->updateNeuron(current_weight_index, current_weight_value + (gradientRowData[i] * _weights.get()->_learningRate));
                }
            }
            delete [] gradientRowData;
        }
    }
}

void Embedding_Layer::Training(bool train) {
    if (train) {
        this->_gradient = std::unique_ptr<Tensor>(new Tensor(this->_dimensions.front(), PreviousLayerOutput()->Shape().back(), _dimensions[2]));
    } else {
        _gradient.reset();
    }
}

 void Embedding_Layer::SetBatchDimensions(int batch_size) {
      _dimensions.front() = batch_size;
     this->_outputResults = std::unique_ptr<Tensor>(new Tensor(batch_size, PreviousLayerOutput()->Shape().back(), _dimensions.back()));
 }
