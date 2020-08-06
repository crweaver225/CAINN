#include "Tensor.h"

float Tensor::_learningRate = 0.1f;

Tensor::Tensor(const int rows, const int columns) {
    this->_dimensions = 1;
    this->_activeDimensions = 1;
    this->_rows = rows;
    this->_columns = columns;
    this->_tensor = new float[rows * columns];
    std::memset(this->_tensor, 0.0f, rows * columns * sizeof(float));
}

Tensor::Tensor(const int dimensions, const int rows, const int columns) {
    this->_dimensions = dimensions;
    this->_activeDimensions = dimensions;
    this->_rows = rows;
    this->_columns = columns;
    this->_tensor = new float[dimensions * rows * columns];
    std::memset(this->_tensor, 0.0f, dimensions * rows * columns * sizeof(float));
}

Tensor::Tensor(const int rows, const int columns, float *tensor) {
    this->_dimensions = 1;
    this->_activeDimensions = 1;
    this->_rows = rows;
    this->_columns = columns;
    this->_tensor = tensor;
}

Tensor::Tensor(const int dimensions, const int rows, const int columns, float *tensor) {
    this->_dimensions = dimensions;
    this->_activeDimensions = dimensions;
    this->_rows = rows;
    this->_columns = columns;
    this->_tensor = tensor;
}

Tensor::~Tensor() {
    delete [] _tensor;
}

Tensor::Tensor(const Tensor &otherTensor) {
    this->_dimensions = otherTensor._dimensions;
    this->_activeDimensions = otherTensor._dimensions;
    this->_rows = otherTensor._rows;
    this->_columns = otherTensor._columns;
    this->_tensor = new float[_dimensions * _rows * _columns];
    *this->_tensor = *(otherTensor.ReturnData());
}

Tensor& Tensor::operator = (const Tensor &otherTensor) {
    if (this == &otherTensor) { return *this; }
    this->_dimensions = otherTensor._dimensions;
    this->_activeDimensions = otherTensor._dimensions;
    this->_rows = otherTensor._rows;
    this->_columns = otherTensor._columns;
    this->_tensor = new float[_dimensions * _rows * _columns];
    *this->_tensor = *(otherTensor._tensor);
    return *this;
}

Tensor::Tensor(Tensor &&otherTensor) {
    this->_dimensions = otherTensor._dimensions;
    this->_activeDimensions = otherTensor._dimensions;
    this->_rows = otherTensor._rows;
    this->_columns = otherTensor._columns;
    this->_tensor = otherTensor._tensor;
    otherTensor._tensor = nullptr;
}

Tensor& Tensor::operator = (Tensor &&otherTensor) {
    if (this == &otherTensor) { return *this; }
    this->_dimensions = otherTensor._dimensions;
    this->_activeDimensions = otherTensor._dimensions;
    this->_rows = otherTensor._rows;
    this->_columns = otherTensor._columns;
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
            threads.emplace_back(std::thread(&Tensor::MatmulInner<a_f>, this,std::ref(m1), std::ref(m2),bias, i,af));
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
    const int number_of_elements = _dimensions * _rows * _columns;
    afd(output._tensor, this->_tensor, number_of_elements);
}
template void Tensor::ApplyDerivative<void (*)(float*, float*, int)>(const Tensor&, void (*)(float*, float*, int));

void Tensor::UpdateGradientInner(const Tensor &gradient, const Tensor &weights, int d) {
    const int gradient_dimension_size = _columns * _rows;
    const int previous_gradient_d_size = gradient._columns * gradient._rows;
    for (int r = 0; r < weights._rows; ++r) {
        for (int c = 0; c < gradient._columns; ++c) {
            _tensor[(d * gradient_dimension_size) + r] += gradient._tensor[(d * previous_gradient_d_size) + c] * weights._tensor[(r * gradient._columns) + c];
        }
    }
}

void Tensor::UpdateGradients(const Tensor &gradient, const Tensor &weights) {
    const int gradient_dimension_size = _columns * _rows;
    const int previous_gradient_d_size = gradient._columns * gradient._rows;
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
    const int final_size = _activeDimensions * _rows * _columns;
    for (int i = 0; i < final_size; i++) {
        _tensor[i] = clip(_tensor[i]);
    }
}

void Tensor::UpdateWeightsInner(const Tensor &gradient, const Tensor &output, const int d) {
    const int gradient_dimensions = output._dimensions;
    float averaged_gradient[_columns * _rows];
    memset(averaged_gradient, 0.0f, _columns * _rows * sizeof(float));

    const int gradient_index = d * gradient._columns * gradient._rows;
    const int output_index = d * output._columns * output._rows;
    for (int gradient_x = 0; gradient_x < gradient._columns; gradient_x++) {
        for (int output_x = 0; output_x < output._columns; output_x++) {
            averaged_gradient[(output_x * _columns) + gradient_x] += gradient._tensor[gradient_index + gradient_x] * output._tensor[output_index + output_x];
        }
    }
    for (int i = 0; i < _columns * _rows; ++i) {
        _tensor[i] += (averaged_gradient[i] / gradient_dimensions) * _learningRate;
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

float Tensor::clip(float x) {
    float value = roundf(x * 100000) / 100000;
    return std::max(-0.1f, std::min(value, 0.1f));
}

void Tensor::ResetTensor() {
    memset(_tensor, 0.0f, _dimensions * _rows * _columns * sizeof(float));
}

void Tensor::SetData(float *tensor) {
    std::memcpy(this->_tensor, tensor, _dimensions * _rows * _columns * sizeof(float));
}

const float * Tensor::ReturnData() const {
    return _tensor;
}

void Tensor::SetActiveDimension(int batch_size) {
    this->_activeDimensions = batch_size;
}

const int Tensor::ReturnActiveDimension() const {
    return this->_activeDimensions;
}

void Tensor::AssignRandomValues() {
    int matrixSize = _rows * _columns;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1,0.1);
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
    return (0.001* finalValue) / (10 * _dimensions * _rows * _columns);
}

void Tensor::UpdateTensor(float *new_tensor) {
    std::memcpy(this->_tensor, new_tensor, _dimensions * _rows * _columns * sizeof(float));
}

void Tensor::Print() const {
    std::cout<<std::endl;
    for (int d = 0; d < _dimensions; ++d) {
        int dimension_size = _rows * _columns * d;
        std::cout<<"[";
        for (int i = 0; i < _rows; ++i) {
            std::cout<<"[";
            for (int j = 0; j < _columns; ++j) {
                std::cout<<_tensor[dimension_size + (j + (i * _columns))];
                if (j < _columns - 1) {
                    std::cout<<",";
                }
            }
            std::cout<<"]";
            if (i < _rows - 1) {
                std::cout<<std::endl;
            }
        }
    }
    std::cout<<"]"<<std::endl;
}

void Tensor::PrintShape() const {
    std::cout<<"("<<_dimensions<<","<<_rows<<","<<_columns<<")"<<std::endl;
}

std::vector<int> Tensor::Shape() const{
    std::vector<int> shape_array(3);
    shape_array[0] = _dimensions;
    shape_array[1] = _rows;
    shape_array[2] = _columns;
    return shape_array;
}