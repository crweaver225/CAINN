#include "Flatten_Layer.h"

Flatten_Layer::Flatten_Layer() : Neural_Layer(std::vector<int>{1,1,1}, Activation_Function::Pass) {}

Flatten_Layer& Flatten_Layer::operator=(Flatten_Layer &&flatten_layer) {
    if (this == &flatten_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(flatten_layer));
    return *this;
}

Flatten_Layer::~Flatten_Layer() {};

void Flatten_Layer::PrintMetaData()  {
    std::cout<<"flatten layer: (1,"<<_outputResults.get()->Shape().back()<<")"<<std::endl;
}

void Flatten_Layer::Build(std::shared_ptr<Neural_Layer> previous_layer) {
    _dimensions.back() =  previous_layer.get()->OutputDimensions()[1] * previous_layer.get()->OutputDimensions().back();
    this->_previousLayer = previous_layer;
    this->_weights = std::unique_ptr<Tensor>(new Tensor(1, 1));
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(1, 1, previous_layer.get()->OutputDimensions()[1] * previous_layer.get()->OutputDimensions().back()));
}

void Flatten_Layer::ForwardPropogate() {
    _outputResults.get()->TransferDataFrom(PreviousLayerOutput());
    _outputResults.get()->flatten();
}

void Flatten_Layer::Backpropogate()  {
    _gradient.get()->reshape(_previousLayer.get()->OutputDimensions()[1], _previousLayer.get()->OutputDimensions().back());
    PreviousLayerGradient()->TransferDataFrom(_gradient.get());
}

void Flatten_Layer::SetBatchDimensions(int batch_size) {
     _dimensions.front() = batch_size;
     this->_outputResults = std::unique_ptr<Tensor>(new Tensor(batch_size, 1, _previousLayer.get()->OutputDimensions()[1] * _previousLayer.get()->OutputDimensions().back()));
}

void Flatten_Layer::Training(bool train) {
    if (train) {
        this->_gradient = std::unique_ptr<Tensor>(new Tensor(this->_dimensions.front(), 1,  _previousLayer.get()->OutputDimensions()[1] * _previousLayer.get()->OutputDimensions().back()));
    } else {
        _gradient.reset();
    }

}