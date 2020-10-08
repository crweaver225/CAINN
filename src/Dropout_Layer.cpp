#include "Dropout_Layer.h"


Dropout_Layer::Dropout_Layer(std::vector<int> dimensions, float percentDropped) : Neural_Layer(dimensions, Activation_Function::Pass) {
    _percentage = percentDropped;
}

Dropout_Layer& Dropout_Layer::operator=(Dropout_Layer &&dropout_layer) {
    if (this == &dropout_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(dropout_layer));
    return *this;
}

Dropout_Layer::~Dropout_Layer() {};

void Dropout_Layer::SetBatchDimensions(int batch_size) {
    _dimensions.front() = batch_size;
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(batch_size, _previousLayer.get()->OutputDimensions()[1], _previousLayer.get()->OutputDimensions().back()));
}

void Dropout_Layer::Build(std::shared_ptr<Neural_Layer> previous_layer) {
    this->_previousLayer = previous_layer;
    this->_weights = std::unique_ptr<Tensor>(new Tensor(1, 1));
    _neurons = previous_layer.get()->OutputDimensions().back();
    _dimensions = previous_layer.get()->OutputDimensions();
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(1, previous_layer.get()->OutputDimensions()[1], previous_layer.get()->OutputDimensions().back()));
}

void Dropout_Layer::randomizeDropped() {
    _droppedNeurons.clear();
    int percentNeurons = _neurons * _percentage;
    srand(time(0));
    for (int i = 0; i < percentNeurons; i++) {
        int droppedNeuron = rand()%_neurons;
        _droppedNeurons.push_back(droppedNeuron);
    }
}

void Dropout_Layer::ForwardPropogate() {
    _outputResults.get()->TransferDataFrom(PreviousLayerOutput());
    for (int neuron : _droppedNeurons) {
        _outputResults.get()->updateNeuron(neuron, 0.0);
    }
}

void Dropout_Layer::Backpropogate() {
    PreviousLayerGradient()->TransferDataFrom(_gradient.get());
}

void Dropout_Layer::PrintMetaData() {
    std::cout<<"dropout layer: ("<<_percentage * 100 <<"% dropped)"<<std::endl;
}

void Dropout_Layer::Training(bool train) {
    if (train) {
        BuildGradient();
    } else {
        _droppedNeurons.clear();
    }
}

