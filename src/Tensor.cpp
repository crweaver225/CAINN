#include "Tensor.h"

float Tensor::_learningRate = 0.1f;

Tensor::Tensor(const int rows, const int columns) {
    this->_dimensions = 1;
    this->_activeDimensions = 1;
    this->_channels = 1;
    this->_rows = rows;
    this->_columns = columns;
    this->_tensor = new float[rows * columns];
    std::memset(this->_tensor, 0.0f, rows * columns * sizeof(float));
}

Tensor::Tensor(const int rows, const int columns, float *tensor) {
    this->_dimensions = 1;
    this->_activeDimensions = 1;
    this->_channels = 1;
    this->_rows = rows;
    this->_columns = columns;
    this->_tensor = tensor;
}

Tensor::Tensor(const int channels, const int rows, const int columns) {
    this->_dimensions = 1;
    this->_activeDimensions = 1;
    this->_rows = rows;
    this->_columns = columns;
    this->_channels = channels;
    this->_tensor = new float[channels * rows * columns];
    std::memset(this->_tensor, 0.0f, channels * rows * columns * sizeof(float));
}

Tensor::Tensor(const int dimensions, const int channels, const int rows, const int columns) {
    this->_dimensions = dimensions;
    this->_activeDimensions = dimensions;
    this->_rows = rows;
    this->_columns = columns;
    this->_channels = channels;
    this->_tensor = new float[dimensions * channels * rows * columns];
    std::memset(this->_tensor, 0.0f, dimensions * channels * rows * columns * sizeof(float));
}

Tensor::Tensor(const int dimensions, const int channels, const int rows, const int columns, float *tensor) {
    this->_dimensions = dimensions;
    this->_activeDimensions = dimensions;
    this->_rows = rows;
    this->_columns = columns;
    this->_channels = channels;
    this->_tensor = tensor;
}

Tensor::~Tensor() {
    delete [] _tensor;
}

Tensor::Tensor(const Tensor &otherTensor) noexcept  {
    this->_dimensions = otherTensor._dimensions;
    this->_activeDimensions = otherTensor._activeDimensions;
    this->_rows = otherTensor._rows;
    this->_columns = otherTensor._columns;
    this->_channels = otherTensor._channels;
    this->_tensor = new float[_dimensions * _channels * _rows * _columns];
    *this->_tensor = *(otherTensor.ReturnData());
}

Tensor& Tensor::operator = (const Tensor &otherTensor) noexcept  {
    if (this == &otherTensor) { return *this; }
    this->_dimensions = otherTensor._dimensions;
    this->_activeDimensions = otherTensor._activeDimensions;
    this->_rows = otherTensor._rows;
    this->_columns = otherTensor._columns;
    this->_channels = otherTensor._channels;
    this->_tensor = new float[_dimensions * _channels * _rows * _columns];
    *this->_tensor = *(otherTensor._tensor);
    return *this;
}

Tensor::Tensor(Tensor &&otherTensor) noexcept  {
    this->_dimensions = otherTensor._dimensions;
    this->_activeDimensions = otherTensor._activeDimensions;
    this->_rows = otherTensor._rows;
    this->_columns = otherTensor._columns;
    this->_channels = otherTensor._channels;
    this->_tensor = otherTensor._tensor;
    otherTensor._tensor = nullptr;
}

Tensor& Tensor::operator = (Tensor &&otherTensor) noexcept  {
    if (this == &otherTensor) { return *this; }
    this->_dimensions = otherTensor._dimensions;
    this->_activeDimensions = otherTensor._activeDimensions;
    this->_rows = otherTensor._rows;
    this->_columns = otherTensor._columns;
    this->_channels = otherTensor._channels;
    this->_tensor = otherTensor._tensor;
    otherTensor._tensor = nullptr;
    return *this;
}

template<typename a_f>
void Tensor::MatmulInner(const Tensor &m1, Tensor &m2, float *bias, int d, a_f af) {
    int dimension_size = m1._rows * m1._columns;
    int product_dimension_size = m1._rows * m2._columns;
    int i_d = d * dimension_size;
    int o_d = d * product_dimension_size;
    for (int i = 0; i < m1._rows; ++i) {
        for (int j = 0; j < m1._columns; ++j) {
            for (int z = 0; z < m2._columns; ++z) {
                _tensor[o_d + (i * (m2._columns) + z)] += m1._tensor[i_d + ((i * m1._columns) + j)] * m2._tensor[(j * m2._columns) + z];
            }
        }
    }
    af(_tensor, bias, o_d, product_dimension_size);
}
template void Tensor::MatmulInner<void (*)(float*, float*, int, int)>(const Tensor&, Tensor&, float*, int, void (*)(float*, float*, int, int));

template<typename a_f>
void Tensor::Matmul(const Tensor &m1, Tensor &m2, float *bias, a_f af) {
    ResetTensor();
    if (_activeDimensions > 15) {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < _activeDimensions; ++i) {
            threads.emplace_back(std::thread(&Tensor::MatmulInner<a_f>, this, std::ref(m1), std::ref(m2),bias, i,af));
        }
        for (auto &t : threads) {
            t.join();
        }
    } else {
        for (int i = 0; i < _activeDimensions; i++) {
            MatmulInner(m1, m2, bias, i, af);
        }
    }
}
template void Tensor::Matmul<void (*)(float*, float*, int, int)>(const Tensor&, Tensor&, float*, void (*)(float*, float*, int, int));


template<typename a_fd>
void Tensor::ApplyDerivative(const Tensor& output, a_fd afd) {
    const int number_of_elements = _dimensions * _channels * _rows * _columns;
    afd(output._tensor, this->_tensor, number_of_elements);
}
template void Tensor::ApplyDerivative<void (*)(float*, float*, int)>(const Tensor&, void (*)(float*, float*, int));

void Tensor::UpdateGradientInner(const Tensor &gradient, const Tensor &weights, int d) {
    const int gradient_dimension_size = _columns * _rows * _channels * d;
    const int previous_gradient_d_size = gradient._columns * gradient._rows * gradient._channels * d;

    for (int r = 0; r < weights._rows; ++r) {
        int tensor_index = gradient_dimension_size + r;
        int weight_index = gradient._columns * r;
        for (int c = 0; c < gradient._columns; ++c) {
            _tensor[tensor_index] += gradient._tensor[previous_gradient_d_size + c] * weights._tensor[weight_index + c];
        }
    }
}

void Tensor::UpdateGradients(const Tensor &gradient, const Tensor &weights) {
    if (_activeDimensions > 15) {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < _activeDimensions; ++i) {
            threads.emplace_back(std::thread(&Tensor::UpdateGradientInner, this, std::ref(gradient), std::ref(weights), i));
        }
        for (auto &t : threads) {
            t.join();
        }
    } else {
        for (int i = 0; i < _activeDimensions; i++) {
            UpdateGradientInner(gradient, weights, i);
        }
    }
    
    clipData();
    
}

void Tensor::UpdateWeightsInner(const Tensor &gradient, const Tensor &output, const int d) {
    const int gradient_dimensions = output._dimensions;
    const int gradient_index = d * gradient._columns * gradient._rows;
    const int output_index = d * output._columns * output._rows;
    
    for (int output_x = 0; output_x < output._columns; output_x++) {
        int weight_row = output_x * _columns;
        for (int gradient_x = 0; gradient_x < gradient._columns; gradient_x++) {
            float new_value = ((output._tensor[output_x + output_index] * gradient._tensor[gradient_x + gradient_index]) / gradient_dimensions) * _learningRate;
            _tensor[weight_row + gradient_x] += new_value;
        }
    }
}

void Tensor::UpdateWeights(const Tensor &gradient, const Tensor &output) {
    if (output._activeDimensions > 15) {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < output._activeDimensions; ++i) {
            threads.emplace_back(std::thread(&Tensor::UpdateWeightsInner, this, std::ref(gradient), std::ref(output), i));
        }
        for (auto &t : threads) {
            t.join();
        }
    } else {
        for (int i = 0; i < output._activeDimensions; i++) {
            UpdateWeightsInner(gradient, output, i);
        }
    }
}

void Tensor::flatten() {
    int newColumns = _channels * _rows * _columns;
    _channels = 1;
    _rows = 1;
    _columns = newColumns;
}

void Tensor::reshape(int channels, int rows, int columns) {
    _channels = channels;
    _rows = rows;
    _columns = columns;
}

void Tensor::reshape(int dimension, int channels, int rows, int columns) {
    _dimensions = dimension;
    _channels = channels;
    _rows = rows;
    _columns = columns;
}

void Tensor::clipData() {
    int size = _activeDimensions * _channels * _rows * _columns;
    for (int i = 0; i < size; i++) {
        _tensor[i] = clip(_tensor[i]);
    }
}

float Tensor::clip(float x) {
    return std::max(-1.0f, std::min(x, 1.0f));
}

void Tensor::ResetTensor() {
    memset(_tensor, 0.0f, _dimensions * _channels * _rows * _columns * sizeof(float));
}

void Tensor::SetData(const float *tensor) {
    std::memcpy(this->_tensor, tensor, _dimensions * _channels * _rows * _columns * sizeof(float));
}

const float * Tensor::ReturnData() const {
    return _tensor;
}

const float * Tensor::returnColumn(int column) const {
    float *returnData = new float[_rows];
    for (int i = 0; i < _rows; i++) {
        returnData[i] = _tensor[(i * _columns) + column];
    }
    return returnData;
}

const float *Tensor::returnColumn(int batch, int column) const {
    float *returnData = new float[_rows];
    int start_index = batch * _rows * _columns;
    for (int i = 0; i < _rows; i++) {
        returnData[i] = _tensor[(start_index + (i * _columns)) + column];
    }
    return returnData;
}

void Tensor::TransferDataFrom(Tensor const* tensor) {
    std::memcpy(this->_tensor, tensor->ReturnData(),  _activeDimensions * _channels * _rows * _columns * sizeof(float));
}

void Tensor::changeNeuron(int index, float value) {
    _tensor[index] += value;
}

void Tensor::setNeuron(int index, float value) {
    if (_activeDimensions > 1) {
        for (int activeDimension = 0; activeDimension < _activeDimensions; activeDimension++) {
            _tensor[index + (_channels * _rows * _columns * activeDimension)] = value;
        }
    } else {
        _tensor[index] = value;
    }
}

void Tensor::setNeuron(int batch, int index, float value) {
    _tensor[(_channels * _rows * _columns * batch) + index] = value;
}

void Tensor::SetActiveDimension(int batch_size) {
    this->_activeDimensions = batch_size;
}

const int Tensor::ReturnActiveDimension() const {
    return this->_activeDimensions;
}

void Tensor::AssignRandomValues() {
    int matrixSize = _dimensions * _channels * _rows * _columns;
    float range = std::sqrt(6.0 / (_rows + _columns));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-range,range);
    for (int i = 0; i < matrixSize; ++i) {
        _tensor[i] = dis(gen);
    }
}

const float Tensor::SumTheSquares() const {
    float finalValue = 0.0f;
    const int number_of_elements = _dimensions * _rows * _columns;
    for (int i = 0; i < number_of_elements; i++) {
        finalValue += std::pow(_tensor[i],2);
    }
    return (0.01f* finalValue) / (10 * _dimensions * _rows * _columns);
}

void Tensor::UpdateTensor(float *new_tensor) {
    std::memcpy(this->_tensor, new_tensor, _dimensions * _rows * _columns * sizeof(float));
}

void Tensor::Print() const {
    std::cout<<std::endl;
    for (int d = 0; d < _activeDimensions; ++d) {
        int dimensionStartIndex = d * (_channels * _rows * _columns);
        std::cout<<"[";
        for (int channel = 0; channel < _channels; channel++) {
            if (_channels > 1) { std::cout<<"["; }
            int start_point = dimensionStartIndex + (channel * _rows * _columns);
            for (int r = 0; r < _rows; r++) {
                std::cout<<"[";
                for (int column = 0; column < _columns; column++) {
                    std::cout<<_tensor[start_point + (column + (r * _columns))];
                    if (column < _columns - 1) {
                        std::cout<<",";
                    }
                }
                std::cout<<"]";
            }
            if (_channels > 1) { std::cout<<"]"; }
        }
    }
    std::cout<<"]"<<std::endl;
    std::cout<<std::endl;
}

void Tensor::PrintShape() const {
    std::cout<<"("<<_dimensions<<","<<_channels<<","<<_rows<<","<<_columns<<")"<<std::endl;
}

const Dimensions Tensor::dimensions() const {
    return Dimensions{_dimensions, _channels, _rows, _columns};
}

const int Tensor::NumberOfRows() const {
    return _rows;
}

const int Tensor::NumberOfColumns() const {
    return _columns;
}

const int Tensor::NumberOfChannels() const {
    return _channels;
}

const int Tensor::NumberOfDimensions() const {
    return _dimensions;
}

const int Tensor::NumberOfElements() const {
    return _dimensions * _channels * _rows * _columns;
}