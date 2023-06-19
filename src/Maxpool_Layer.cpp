#include "Maxpool_Layer.h"

Maxpool_Layer::Maxpool_Layer(Dimensions dimensions) : Neural_Layer(dimensions, Activation_Function::Pass) {}

Maxpool_Layer::Maxpool_Layer(Maxpool_Layer &&maxpool_layer)  noexcept  : Neural_Layer{std::move(maxpool_layer)} {}

Maxpool_Layer& Maxpool_Layer::operator=(Maxpool_Layer &&maxpool_layer)  noexcept {
    if (this == &maxpool_layer) {
        return *this;
    }
    Neural_Layer::operator=(std::move(maxpool_layer));
    return *this;
 }

Maxpool_Layer::~Maxpool_Layer() {}

void Maxpool_Layer::PrintMetaData() {
    std::cout<<"maxpool layer: ("<<_filters<<","<<_dimensions.rows<<","<<_dimensions.columns<<")"<<std::endl;
}

void Maxpool_Layer::Build(Neural_Layer const* previousLayer)  {
    _previousLayer_Dimensions = previousLayer->ReturnDimensions();
    this->_weights = std::unique_ptr<Tensor>(new Tensor(1, 1));
    //this->_previousLayer = previous_layer;

    _filterSize = _dimensions.rows;
    _stride = _dimensions.columns;
    _filters = _previousLayer_Dimensions.channels;

    int outputDimension = ((_previousLayer_Dimensions.columns - _filterSize) / _stride) + 1;
    //this->_output = Tensor(1, _filters, outputDimension, outputDimension);
    _output = std::make_unique<Tensor>(1, _filters, outputDimension, outputDimension);

    _dimensions.rows = outputDimension;
    _dimensions.columns = outputDimension;

    _outputDimensions = std::vector<int>{this->_dimensions.dimensions,_filters,_dimensions.rows,_dimensions.columns};
}

Tensor const* Maxpool_Layer::ForwardPropogate(Tensor const* input) {
    //_maxpooledIndexes = this->_output.Maxpool(tensor, _filterSize, _stride);
    return _output.get();
}

Tensor* Maxpool_Layer::Backpropogate(Tensor* gradient) {
    /*
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
  */
 return _gradient.get();
}

void Maxpool_Layer::SetBatchDimensions(int batch_size) {
    _dimensions.dimensions = batch_size;
    _output = std::make_unique<Tensor>(Tensor(batch_size, _filters, _dimensions.rows, _dimensions.columns));
    _outputDimensions = std::vector<int>{this->_dimensions.dimensions,_filters,_dimensions.rows,_dimensions.columns};
}

void Maxpool_Layer::Training(bool train) {
    if (train) {
        this->_gradient = std::unique_ptr<Tensor>(new Tensor(this->_dimensions.dimensions, _filters, _dimensions.rows, _dimensions.columns));
    } else {
        _gradient.reset();
    }
}
/*
const std::vector<int>& Maxpool_Layer::OutputDimensions() {
    return _outputDimensions;
}
*/

int Maxpool_Layer::returnFilterSize() const {
  return _filterSize;
}
int Maxpool_Layer::returnStride() const {
  return _stride;
}