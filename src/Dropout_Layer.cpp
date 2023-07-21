#include "Dropout_Layer.h"


Dropout_Layer::Dropout_Layer(Dimensions dimensions, float percentDropped) : Neural_Layer(dimensions, Activation_Function::Pass) {
    _percentage = percentDropped;
    std::cout<<"Dropout Layer constructor\n";
}

Dropout_Layer::Dropout_Layer(Dropout_Layer &&dropout_layer) noexcept  : Neural_Layer{std::move(dropout_layer)} {
    this->_percentage = dropout_layer._percentage;
    this->_neurons = dropout_layer._neurons;
    this->_droppedNeurons = dropout_layer._droppedNeurons;
}

Dropout_Layer& Dropout_Layer::operator=(Dropout_Layer &&dropout_layer) noexcept {
    if (this == &dropout_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(dropout_layer));
    return *this;
}

Dropout_Layer::~Dropout_Layer() { }

void Dropout_Layer::SetBatchDimensions(int batch_size) {
    _dimensions.dimensions = batch_size;
    _previousLayer_Dimensions.dimensions = batch_size;
    _output = std::make_unique<Tensor>(Tensor(batch_size, 1, 
                                            _dimensions.rows, 
                                            _dimensions.columns
                                            ));
}

void Dropout_Layer::Build(Neural_Layer const* previousLayer) {
    _previousLayer_Dimensions = previousLayer->ReturnDimensions();
    _weights = std::make_unique<Tensor>(Tensor(1,1));
    _dimensions = _previousLayer_Dimensions;
    _neurons = _dimensions.rows * _dimensions.columns;
    _output = std::make_unique<Tensor>(Tensor(
                                        _dimensions.dimensions, 
                                        _dimensions.channels, 
                                        _dimensions.rows,
                                        _dimensions.columns
                                        ));
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

Tensor const* Dropout_Layer::ForwardPropogate(Tensor const* input) {
    _input = input;
    _output.get()->TransferDataFrom(input);
    for (int neuron : _droppedNeurons) {
        _output.get()->updateNeuron(neuron, 0.0);
    }
    return _output.get();
}

Tensor* Dropout_Layer::Backpropogate(Tensor* gradient) {
    return gradient;
}

void Dropout_Layer::PrintMetaData() {
    std::cout<<"dropout layer: ("<<_percentage * 100 <<"% dropped)\n";
}

void Dropout_Layer::Training(bool train) {
    if (train) {
        this->_gradient = std::make_unique<Tensor>(Tensor(1,1));
    } else {
        _droppedNeurons.clear();
    }
}

float Dropout_Layer::returnPercentageDropped() const {
    return _percentage;
}