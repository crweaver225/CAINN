#include "Flatten_Layer.h"

Flatten_Layer::Flatten_Layer() : Neural_Layer(Dimensions{1,1,1,1}, Activation_Function::Pass) {}

Flatten_Layer::Flatten_Layer(Flatten_Layer &&flatten_layer)  noexcept  : Neural_Layer{std::move(flatten_layer)} {}

Flatten_Layer& Flatten_Layer::operator=(Flatten_Layer &&flatten_layer)  noexcept {
    if (this == &flatten_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(flatten_layer));
    return *this;
}

Flatten_Layer::~Flatten_Layer() {};

void Flatten_Layer::PrintMetaData()  {
    std::cout<<"flatten layer: (1,"<<_output.get()->NumberOfColumns()<<")"<<std::endl;
}

void Flatten_Layer::Build(Neural_Layer const* previousLayer) {
    _previousLayer_Dimensions = previousLayer->ReturnDimensions();
    _dimensions.columns = _previousLayer_Dimensions.channels * _previousLayer_Dimensions.rows * _previousLayer_Dimensions.columns;
    this->_weights = std::unique_ptr<Tensor>(new Tensor(1, 1));
    _output = std::make_unique<Tensor>(Tensor(1, _dimensions.columns));
}

Tensor const* Flatten_Layer::ForwardPropogate(Tensor const* input){
    _output->TransferDataFrom(input);
    _output->flatten();
    return _output.get();
}

Tensor* Flatten_Layer::Backpropogate(Tensor* gradient)  {
    gradient->reshape(_previousLayer_Dimensions.dimensions, 
                        _previousLayer_Dimensions.channels, 
                        _previousLayer_Dimensions.rows, 
                        _previousLayer_Dimensions.columns);
    return gradient;
}

void Flatten_Layer::SetBatchDimensions(int batch_size) {
     _dimensions.dimensions = batch_size;
     _output = std::make_unique<Tensor>(Tensor(batch_size,
                                                1,
                                                1,
                                                _dimensions.columns
                                                ));
}

void Flatten_Layer::Training(bool train) {
    if (train) {
        this->_gradient = std::unique_ptr<Tensor>(new Tensor(1, 1, 1, 1));
    } else {
        _gradient.reset();
    }

}