#include "Dropout_Layer.h"


Dropout_Layer::Dropout_Layer(Dimensions dimensions, float percentDropped) :
    Neural_Layer(dimensions, Activation_Function::Pass),
    _generator(std::random_device{}()) {
    _percentage = percentDropped;
}

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
    _output = std::make_unique<Tensor>(Tensor(
                                        _dimensions.dimensions, 
                                        _dimensions.channels, 
                                        _dimensions.rows,
                                        _dimensions.columns
                                        ));
}

Tensor const* Dropout_Layer::ForwardPropogate(Tensor const* input) {
    _input = input;
    _output.get()->TransferDataFrom(input);

    if (!_training) {
        return _output.get();
    }

    const int elements = _output->ReturnActiveDimension() * _output->NumberOfElementsPerTensor();
    const float keep_probability = 1.0f - _percentage;
    const float scale = keep_probability > 0.0f ? 1.0f / keep_probability : 0.0f;
    std::bernoulli_distribution keep_neuron(keep_probability);

    _mask.resize(elements);
    for (int i = 0; i < elements; i++) {
        _mask[i] = keep_neuron(_generator) ? scale : 0.0f;
    }

    _output->MultiplyElementwise(_mask.data(), elements);
    return _output.get();
}

Tensor* Dropout_Layer::Backpropogate(Tensor* gradient) {
    gradient->MultiplyElementwise(_mask.data(), _mask.size());
    return gradient;
}

void Dropout_Layer::PrintMetaData() {
    std::cout<<"dropout layer: ("<<_percentage * 100 <<"% dropped)\n";
}

void Dropout_Layer::Training(bool train) {
    _training = train;
    if (train) {
        this->_gradient = std::make_unique<Tensor>(Tensor(1,1));
    } else {
        _mask.clear();
    }
}

float Dropout_Layer::returnPercentageDropped() const {
    return _percentage;
}