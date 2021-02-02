#include "Maxpool_Layer.h"

Maxpool_Layer::Maxpool_Layer(std::vector<int> dimensions) : Neural_Layer(dimensions, Activation_Function::Pass) {}

Maxpool_Layer& Maxpool_Layer::operator=(Maxpool_Layer &&maxpool_layer) {
    if (this == &maxpool_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(maxpool_layer));
    return *this;
 }

Maxpool_Layer::~Maxpool_Layer() {}

void Maxpool_Layer::PrintMetaData() {
    std::cout<<"maxpool layer: ("<<_filters<<","<<_dimensions[2]<<","<<_dimensions[3]<<")"<<std::endl;
}

void Maxpool_Layer::Build(std::shared_ptr<Neural_Layer> previous_layer) {
    this->_weights = std::unique_ptr<Tensor>(new Tensor(1, 1));
    this->_previousLayer = previous_layer;

    _filterSize = _dimensions[2];
    _stride = _dimensions[3];
    _filters = previous_layer.get()->OutputDimensions()[1];

    int outputDimension = ((previous_layer.get()->OutputDimensions()[3] - _filterSize) / _stride) + 1;
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(1, _filters, outputDimension, outputDimension));

    _dimensions[2] = outputDimension;
    _dimensions[3] = outputDimension;

    _outputDimensions = std::vector<int>{this->_dimensions.front(),_filters,_dimensions[2],_dimensions[3]};
}

void Maxpool_Layer::ForwardPropogate() {
    _maxpooledIndexes = this->_outputResults.get()->Maxpool(*PreviousLayerOutput(), _filterSize, _stride);
}

void Maxpool_Layer::Backpropogate() {

   const float *gradientData =  _gradient.get()->ReturnData();
   int outputSize = _dimensions[0] * _filters * _dimensions[2] * _dimensions[3];
   std::vector<int> previousLayerShape = PreviousLayerOutput()->Shape();
   float * updatedGradientData = new float [previousLayerShape[0] * previousLayerShape[1] * previousLayerShape[2] * previousLayerShape[3]];
   memset(updatedGradientData, 0.0f, previousLayerShape[0] * previousLayerShape[1] * previousLayerShape[2] * previousLayerShape[3] * sizeof(float));
   for (int index = 0; index < outputSize; index ++) {
       updatedGradientData[_maxpooledIndexes[index]] += gradientData[index];
     PreviousLayerGradient()->SetData(updatedGradientData);
   }
  PreviousLayerGradient()->clipData();
  delete [] updatedGradientData;
}

void Maxpool_Layer::SetBatchDimensions(int batch_size) {
    _dimensions[0] = batch_size;
    _outputResults.reset();
    this->_outputResults = std::unique_ptr<Tensor>(new Tensor(batch_size, _filters, _dimensions[2], _dimensions[3]));
    _outputDimensions = std::vector<int>{this->_dimensions.front(),_filters,_dimensions[2],_dimensions[3]};
}

void Maxpool_Layer::Training(bool train) {
    if (train) {
        this->_gradient = std::unique_ptr<Tensor>(new Tensor(this->_dimensions.front(), _filters, _dimensions[2], _dimensions[3]));
    } else {
        _gradient.reset();
    }
}

const std::vector<int>& Maxpool_Layer::OutputDimensions() {
    return _outputDimensions;
}

int Maxpool_Layer::returnFilterSize() const {
  return _filterSize;
}
int Maxpool_Layer::returnStride() const {
  return _stride;
}