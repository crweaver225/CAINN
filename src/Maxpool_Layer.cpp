#include "Maxpool_Layer.h"

Maxpool_Layer::Maxpool_Layer(int kernel_size, int stride) : Neural_Layer({1,1,1,1}, Activation_Function::Pass) {
    _kernel_size = kernel_size;
    _stride = stride;
}

Maxpool_Layer::Maxpool_Layer(Maxpool_Layer &&maxpool_layer)  noexcept  : Neural_Layer{std::move(maxpool_layer)} {
    this->_kernel_size = maxpool_layer._kernel_size;
    this->_stride = maxpool_layer._stride;
    this->_maxpooledIndexes = maxpool_layer._maxpooledIndexes;
}

Maxpool_Layer& Maxpool_Layer::operator=(Maxpool_Layer &&maxpool_layer)  noexcept {
    if (this == &maxpool_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(maxpool_layer));
    return *this;
 }

Maxpool_Layer::~Maxpool_Layer() {}

void Maxpool_Layer::PrintMetaData() {
    std::cout<<"maxpool layer: ("<< _dimensions.channels<<","<<_dimensions.rows<<","<<_dimensions.columns<<")\n";
}

void Maxpool_Layer::Build(Neural_Layer const* previousLayer)  {
    _previousLayer_Dimensions = previousLayer->ReturnDimensions();
    this->_weights = std::unique_ptr<Tensor>(new Tensor(1, 1));

    int outputDimension = ((_previousLayer_Dimensions.columns - _kernel_size) / _stride) + 1;
    _output = std::make_unique<Tensor>(1, _previousLayer_Dimensions.channels, outputDimension, outputDimension);
    _dimensions.channels = _previousLayer_Dimensions.channels;
    _dimensions.rows = outputDimension;
    _dimensions.columns = outputDimension;

    _maxpooledIndexes = std::vector<int>(0, 0);
}

Tensor const* Maxpool_Layer::ForwardPropogate(Tensor const* input) {
    _maxpooledIndexes.assign(_maxpooledIndexes.size(), 0);
    _output->Maxpool(*input, _kernel_size, _stride, _maxpooledIndexes);
    return _output.get();
}

Tensor* Maxpool_Layer::Backpropogate(Tensor* gradient) {
    const float *gradientData =  gradient->ReturnData();
    const int output_size =  _dimensions.channels * _dimensions.rows * _dimensions.columns;
    for (int index = 0; index < output_size; index ++) {
        _gradient->changeNeuron(_maxpooledIndexes[index], gradientData[index]);
    }
    return _gradient.get();
}

void Maxpool_Layer::SetBatchDimensions(int batch_size) {
    _dimensions.dimensions = batch_size;
    _output = std::make_unique<Tensor>(Tensor(batch_size,  _dimensions.channels, _dimensions.rows, _dimensions.columns));
}

void Maxpool_Layer::Training(bool train) {
    if (train) {
        BuildGradient();
        _maxpooledIndexes = std::vector<int>(_dimensions.dimensions * _dimensions.channels * _dimensions.rows * _dimensions.columns, 0);
    } else {
        _gradient.reset();
        _maxpooledIndexes.clear();
    }

}
