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

// Old matmul
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
    const int gradient_dimension_size = _columns * _rows * d;
    const int previous_gradient_d_size = gradient._columns * gradient._rows * d;

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
    
    const int final_size = _activeDimensions * _rows * _columns;
    for (int i = 0; i < final_size; i++) {
        _tensor[i] = clip(_tensor[i]);
    }
    
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

void Tensor::TransferDataFrom(const Tensor *tensor) {
    std::memcpy(this->_tensor, tensor->ReturnData(),  _dimensions * _rows * _columns * sizeof(float));
}

void Tensor::updateNeuron(int index, float value) {
    if (_activeDimensions > 1) {
        for (int activeDimension = 0; activeDimension < _activeDimensions; activeDimension++) {
            _tensor[index + (_rows * _columns)] = value;
        }
    } else {
        _tensor[index] = value;
    }
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
    return (0.1f* finalValue) / (10 * _dimensions * _rows * _columns);
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

/*
template<typename a_f>
void Tensor::Matmul(const Tensor &m1, Tensor &m2, float *bias, a_f af) {
    ResetTensor();
    if (_activeDimensions > 15) {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < _activeDimensions; ++i) {
            threads.emplace_back(std::thread(&Tensor::MatmulDimension<a_f>, this,std::ref(m1), std::ref(m2),bias, i,af));
        }
        for (auto &t : threads) {
            t.join();
        }
    } else {
        for (int i = 0; i < _activeDimensions; i++) {
            MatmulDimension(m1, m2, bias, i, af);
        }
    }
}
template void Tensor::Matmul<void (*)(float*, float*, int, int)>(const Tensor&, Tensor&, float*, void (*)(float*, float*, int, int));

template<typename a_f>
void Tensor::MatmulDimension(const Tensor &m1, Tensor &m2, float *bias, int d, a_f af) {
    int dimension_size = m1._rows * m1._columns;
    int product_dimension_size = m1._rows * m2._columns;
    int i_d = d * dimension_size;
    int o_d = d * product_dimension_size;
    
    // mat mul on 8x8 grids of a and b
    for (int i = 0; i <= m1._rows - 8; i += 8) {
        for (int j = 0; j <= m1._columns - 8; j += 8) {
            for (int k = 0; k <= m2._columns - 8; k += 8) {
                MatmulInner(m1, m2, i + i_d, j + i_d, j, k, d);
            }
        }
    }
    
    const int m1_row_uncovered = m1._rows % 8;
    const int m1_column_uncovered = m1._columns % 8;
    const int m2_column_uncovered = m2._columns % 8;

    // Take care of extra columns in A where A row was partially done
    for (int i = 0; i < m1._rows - m1_row_uncovered; i++) {
        for (int j = m1._columns - m1_column_uncovered; j < m1._columns; j++) {
            for (int k = 0; k < m2._columns; k++) {
                _tensor[((i * _columns) + k) + o_d] += m1._tensor[((i * m1._columns) + j) + i_d] * m2._tensor[(j * m2._columns) + k];
            }
        }
    }

    // Take care of extra columns in B
    for (int i = m2._columns - m2_column_uncovered; i < m2._columns; i++) {
        for (int k = 0; k < m1._rows - m1_row_uncovered; k++) {
            for (int j = 0; j < m2._rows - m1_column_uncovered; j++) {
                _tensor[((_columns * k) + i) + o_d] += m1._tensor[((k * m1._columns) + j) + i_d] * m2._tensor[(j * m2._columns) + i];
            }
        }
    }

    // Take care of remainding full rows of A
    for (int i = m1._rows - m1_row_uncovered; i < m1._rows; i++) {
        for (int j = 0; j < m1._columns; j++) {
            for (int k = 0; k < m2._columns; k++) {
                _tensor[((i * _columns) + k) + o_d] += m1._tensor[((i * m1._columns) + j) + i_d] * m2._tensor[(j * m2._columns) + k];
            }
        }
    }
    
    af(_tensor, bias, o_d, product_dimension_size);
}
template void Tensor::MatmulDimension<void (*)(float*, float*, int, int)>(const Tensor&, Tensor&, float*, int, void (*)(float*, float*, int, int));


////////////////////////////
///// AVX Matmul inner /////
////////////////////////////
void Tensor::MatmulInner(const Tensor &m1, Tensor &m2, int a_row, int a_column, int b_row, int b_columm, int dimension) {

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
    ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15, ymmC;

    int product_dimension_size = m1._rows * m2._columns;
    int o_d = dimension * product_dimension_size;
    
    //Read the eight rows of Matrix B into ymm registers
    ymm8 = _mm256_load_ps((float *) (m2._tensor + b_row * m2._columns + b_columm));
    ymm9 = _mm256_load_ps((float *) (m2._tensor + (b_row + 1) * m2._columns + b_columm));
    ymm10 = _mm256_load_ps((float *) (m2._tensor + (b_row + 2) * m2._columns + b_columm));
    ymm11 = _mm256_load_ps((float *) (m2._tensor + (b_row + 3) * m2._columns + b_columm));
    ymm12 = _mm256_load_ps((float *) (m2._tensor + (b_row + 4) * m2._columns + b_columm));
    ymm13 = _mm256_load_ps((float *) (m2._tensor + (b_row + 5) * m2._columns + b_columm));
    ymm14 = _mm256_load_ps((float *) (m2._tensor + (b_row + 6) * m2._columns + b_columm));
    ymm15 = _mm256_load_ps((float *) (m2._tensor + (b_row + 7) * m2._columns + b_columm));
    
    //Broadcast each element of Matrix A Row 1 into a ymm register
    int row0 = ((m1._columns * a_row) + a_column);
    ymm0 = _mm256_broadcast_ss(m1._tensor + row0);
    ymm1 = _mm256_broadcast_ss(m1._tensor + row0 + 1);
    ymm2 = _mm256_broadcast_ss(m1._tensor + row0 + 2);
    ymm3 = _mm256_broadcast_ss(m1._tensor + row0 + 3);
    ymm4 = _mm256_broadcast_ss(m1._tensor + row0 + 4);
    ymm5 = _mm256_broadcast_ss(m1._tensor + row0 + 5);
    ymm6 = _mm256_broadcast_ss(m1._tensor + row0 + 6);
    ymm7 = _mm256_broadcast_ss(m1._tensor + row0 + 7);
    
    //Multiply A11 times Row 1 of Matrix B
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    
    //Multiply A12 times Row 2 of Matrix B
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    
    //Create the first partial sum
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    
    //Repeat for A13, A14, and Rows 3, 4 of Matrix B
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);

    //Repeat for A15, A16, and Rows 5, 6 of Matrix B
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);

    //Repeat for A17, A18, and Rows 7, 8 of Matrix B
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    
    //Perform the final three adds
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    
    //Store the result to Row 1 of Matrix C
    ymmC = _mm256_load_ps((float *) (_tensor + (((_columns * a_row) + b_columm) + o_d)));
    _mm256_store_ps((float *) (_tensor + (((_columns * a_row) + b_columm) + o_d)), ymm0 + ymmC);
    
    //Repeat using Matrix A Row 2
    int row1 = ((m1._columns * (a_row + 1)) + a_column);
    ymm0 = _mm256_broadcast_ss(m1._tensor + row1);
    ymm1 = _mm256_broadcast_ss(m1._tensor + row1 + 1);
    ymm2 = _mm256_broadcast_ss(m1._tensor + row1 + 2);
    ymm3 = _mm256_broadcast_ss(m1._tensor + row1 + 3);
    ymm4 = _mm256_broadcast_ss(m1._tensor + row1 + 4);
    ymm5 = _mm256_broadcast_ss(m1._tensor + row1 + 5);
    ymm6 = _mm256_broadcast_ss(m1._tensor + row1 + 6);
    ymm7 = _mm256_broadcast_ss(m1._tensor + row1 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    ymmC = _mm256_load_ps((float *) (_tensor + (((_columns * (a_row + 1)) + b_columm)) + o_d));
    _mm256_store_ps((float *) (_tensor + (((_columns * (a_row + 1)) + b_columm) + o_d)), ymm0 + ymmC);

    //Repeat using Matrix A Row 3
    int row2 = ((m1._columns * (a_row + 2)) + a_column);
    ymm0 = _mm256_broadcast_ss(m1._tensor + row2);
    ymm1 = _mm256_broadcast_ss(m1._tensor+ row2 + 1);
    ymm2 = _mm256_broadcast_ss(m1._tensor + row2 + 2);
    ymm3 = _mm256_broadcast_ss(m1._tensor + row2 + 3);
    ymm4 = _mm256_broadcast_ss(m1._tensor + row2 + 4);
    ymm5 = _mm256_broadcast_ss(m1._tensor + row2 + 5);
    ymm6 = _mm256_broadcast_ss(m1._tensor + row2 + 6);
    ymm7 = _mm256_broadcast_ss(m1._tensor + row2 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    ymmC = _mm256_load_ps((float *) (_tensor + (((_columns * (a_row + 2)) + b_columm) + o_d)));
    _mm256_store_ps((float *) (_tensor + (((_columns * (a_row + 2)) + b_columm) + o_d)), ymm0 + ymmC);

    //Repeat using Matrix A Row 4
    int row3 = ((m1._columns * (a_row + 3)) + a_column);
    ymm0 = _mm256_broadcast_ss(m1._tensor + row3);
    ymm1 = _mm256_broadcast_ss(m1._tensor + row3 + 1);
    ymm2 = _mm256_broadcast_ss(m1._tensor + row3 + 2);
    ymm3 = _mm256_broadcast_ss(m1._tensor + row3 + 3);
    ymm4 = _mm256_broadcast_ss(m1._tensor + row3 + 4);
    ymm5 = _mm256_broadcast_ss(m1._tensor + row3 + 5);
    ymm6 = _mm256_broadcast_ss(m1._tensor + row3 + 6);
    ymm7 = _mm256_broadcast_ss(m1._tensor + row3 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    ymmC = _mm256_load_ps((float *) (_tensor + (((_columns * (a_row + 3) + o_d)) + b_columm)));
    _mm256_store_ps((float *) (_tensor+ (((_columns * (a_row + 3)) + b_columm) + o_d)), ymm0 + ymmC);

    //Repeat using Matrix A Row 5
    int row4 = ((m1._columns * (a_row + 4)) + a_column);
    ymm0 = _mm256_broadcast_ss(m1._tensor + row4);
    ymm1 = _mm256_broadcast_ss(m1._tensor + row4 + 1);
    ymm2 = _mm256_broadcast_ss(m1._tensor + row4 + 2);
    ymm3 = _mm256_broadcast_ss(m1._tensor + row4 + 3);
    ymm4 = _mm256_broadcast_ss(m1._tensor + row4 + 4);
    ymm5 = _mm256_broadcast_ss(m1._tensor + row4 + 5);
    ymm6 = _mm256_broadcast_ss(m1._tensor + row4 + 6);
    ymm7 = _mm256_broadcast_ss(m1._tensor + row4 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    ymmC = _mm256_load_ps((float *) (_tensor + ((_columns * (a_row + 4)) + b_columm)));
    _mm256_store_ps((float *) (_tensor + (((_columns * (a_row + 4)) + b_columm) + o_d)), ymm0 + ymmC);

    //Repeat using Matrix A Row 6
    int row5 = ((m1._columns * (a_row + 5)) + a_column);
    ymm0 = _mm256_broadcast_ss(m1._tensor + row5);
    ymm1 = _mm256_broadcast_ss(m1._tensor + row5 + 1);
    ymm2 = _mm256_broadcast_ss(m1._tensor + row5 + 2);
    ymm3 = _mm256_broadcast_ss(m1._tensor + row5 + 3);
    ymm4 = _mm256_broadcast_ss(m1._tensor + row5 + 4);
    ymm5 = _mm256_broadcast_ss(m1._tensor + row5 + 5);
    ymm6 = _mm256_broadcast_ss(m1._tensor + row5 + 6);
    ymm7 = _mm256_broadcast_ss(m1._tensor + row5 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    ymmC = _mm256_load_ps((float *) (_tensor + (((_columns * (a_row + 5) + o_d)) + b_columm)));
    _mm256_store_ps((float *) (_tensor + (((_columns * (a_row + 5)) + b_columm) + o_d)), ymm0 + ymmC);

    //Repeat using Matrix A Row 7
    int row6 = ((m1._columns * (a_row + 6)) + a_column);
    ymm0 = _mm256_broadcast_ss(m1._tensor + row6);
    ymm1 = _mm256_broadcast_ss(m1._tensor + row6 + 1);
    ymm2 = _mm256_broadcast_ss(m1._tensor + row6 + 2);
    ymm3 = _mm256_broadcast_ss(m1._tensor + row6 + 3);
    ymm4 = _mm256_broadcast_ss(m1._tensor + row6 + 4);
    ymm5 = _mm256_broadcast_ss(m1._tensor + row6 + 5);
    ymm6 = _mm256_broadcast_ss(m1._tensor + row6 + 6);
    ymm7 = _mm256_broadcast_ss(m1._tensor+ row6 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    ymmC = _mm256_load_ps((float *) (_tensor + (((_columns * (a_row + 6)) + b_columm) + o_d)));
    _mm256_store_ps((float *) (_tensor+ (((_columns * (a_row + 6)) + b_columm) + o_d)), ymm0 + ymmC);

    //Repeat using Matrix A Row 8
    int row7 = ((m1._columns * (a_row + 7)) + a_column);
    ymm0 = _mm256_broadcast_ss(m1._tensor+ row7);
    ymm1 = _mm256_broadcast_ss(m1._tensor + row7 + 1);
    ymm2 = _mm256_broadcast_ss(m1._tensor + row7 + 2);
    ymm3 = _mm256_broadcast_ss(m1._tensor + row7 + 3);
    ymm4 = _mm256_broadcast_ss(m1._tensor + row7 + 4);
    ymm5 = _mm256_broadcast_ss(m1._tensor + row7 + 5);
    ymm6 = _mm256_broadcast_ss(m1._tensor + row7 + 6);
    ymm7 = _mm256_broadcast_ss(m1._tensor + row7 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    ymmC = _mm256_load_ps((float *) (_tensor + (((_columns * (a_row + 7)) + b_columm) + o_d)));
    _mm256_store_ps((float *) (_tensor + (((_columns * (a_row + 7)) + b_columm) + o_d)), ymm0 + ymmC);
}
*/