#include "Tensor.h"

float Tensor::learning_rate = 0.1f;

Tensor::Tensor(const int rows, const int columns) {
  //  std::cout<<"Tensor constructor with "<<rows<<" rows and "<<columns<<" columns called"<<std::endl;
    this->dimensions = 1;
    this->active_dimensions = 1;
    this->rows = rows;
    this->columns = columns;
    this->tensor = new float[rows * columns];
    std::memset(this->tensor, 0.0f, rows * columns * sizeof(float));
}

Tensor::Tensor(const int dimensions, const int rows, const int columns) {
  //  std::cout<<"Tensor constructor with dimensions, rows, and columns called"<<std::endl;
    this->dimensions = dimensions;
    this->active_dimensions = dimensions;
    this->rows = rows;
    this->columns = columns;
    this->tensor = new float[dimensions * rows * columns];
    std::memset(this->tensor, 0.0f, dimensions * rows * columns * sizeof(float));
}

Tensor::Tensor(const int rows, const int columns, float *tensor) {
   // std::cout<<"Tensor constructor with rows, columns, and data called"<<std::endl;
    this->dimensions = 1;
    this->active_dimensions = 1;
    this->rows = rows;
    this->columns = columns;
    this->tensor = tensor;
}

Tensor::Tensor(const int dimensions, const int rows, const int columns, float *tensor) {
   // std::cout<<"Tensor constructor with dimensions, rows, columns, and data called"<<std::endl;
    //std::cout<<dimensions<<","<<rows<<","<<columns<<std::endl;
    this->dimensions = dimensions;
    this->active_dimensions = dimensions;
    this->rows = rows;
    this->columns = columns;
    this->tensor = tensor;
}

Tensor::~Tensor() {
   // std::cout<<"Tensor destructor called"<<std::endl;
    delete [] tensor;
}

Tensor::Tensor(const Tensor &otherTensor) {
   // std::cout<<"Tensor copy constructor called"<<std::endl;
    this->dimensions = otherTensor.dimensions;
    this->active_dimensions = otherTensor.dimensions;
    this->rows = otherTensor.rows;
    this->columns = otherTensor.columns;
    this->tensor = new float[dimensions * rows * columns];
    *this->tensor = *(otherTensor.returnData());
}

Tensor& Tensor::operator = (const Tensor &otherTensor) {
   // std::cout<<"Tensor copy assignment operator called"<<std::endl;
    if (this == &otherTensor) { return *this; }
    this->dimensions = otherTensor.dimensions;
    this->active_dimensions = otherTensor.dimensions;
    this->rows = otherTensor.rows;
    this->columns = otherTensor.columns;
    this->tensor = new float[dimensions * rows * columns];
    *this->tensor = *(otherTensor.tensor);
    return *this;
}

Tensor::Tensor(Tensor &&otherTensor) {
   // std::cout<<"Tensor move constructor called"<<std::endl;
    this->dimensions = otherTensor.dimensions;
    this->active_dimensions = otherTensor.dimensions;
    this->rows = otherTensor.rows;
    this->columns = otherTensor.columns;
    this->tensor = otherTensor.tensor;
    otherTensor.tensor = nullptr;
}

Tensor& Tensor::operator = (Tensor &&otherTensor) {
   // std::cout<<"Tensor move assignment operator called"<<std::endl;
    if (this == &otherTensor) { return *this; }
    this->dimensions = otherTensor.dimensions;
    this->active_dimensions = otherTensor.dimensions;
    this->rows = otherTensor.rows;
    this->columns = otherTensor.columns;
    this->tensor = otherTensor.tensor;
    otherTensor.tensor = nullptr;
    return *this;
}

template<typename a_f>
void Tensor::matmul(const Tensor &m1, Tensor &m2, float *bias, a_f af) {
    int dimension_size = m1.rows * m1.columns;
    int product_dimension_size = m1.rows * m2.columns;
    resetTensor();
    for (int d = 0; d < active_dimensions; ++d) {
        int i_d = d * dimension_size;
        int o_d = d * product_dimension_size;
        for (int i = 0; i < m1.rows; ++i) {
            for (int j = 0; j < m1.columns; ++j) {
                for (int z = 0; z < m2.columns; ++z) {
                    tensor[o_d + (i * (m2.columns) + z)] += m1.tensor[i_d + ((i * m1.columns) + j)] * m2.tensor[(j * m2.columns) + z];
                }
            }
        }
    }
    for (int d = 0; d < active_dimensions; ++d) {
        int dimension_location = d * product_dimension_size;
        af(tensor, bias, dimension_location, product_dimension_size);
    }
}
template void Tensor::matmul<void (*)(float*, float*, int, int)>(const Tensor&, Tensor&, float*, void (*)(float*, float*, int, int));

template<typename a_fd>
void Tensor::applyDerivative(const Tensor& output, a_fd afd) {
    const int number_of_elements = dimensions * rows * columns;
    afd(output.tensor, this->tensor, number_of_elements);
}
template void Tensor::applyDerivative<void (*)(float*, float*, int)>(const Tensor&, void (*)(float*, float*, int));

void Tensor::updateGradients(const Tensor &gradient, const Tensor &weights) {
    const int gradient_dimension_size = columns * rows;
    const int previous_gradient_d_size = gradient.columns * gradient.rows;
    for (int d = 0; d < dimensions; ++d) {
        for (int r = 0; r < weights.rows; ++r) {
            for (int c = 0; c < gradient.columns; ++c) {
                tensor[(d * gradient_dimension_size) + r] += gradient.tensor[(d * previous_gradient_d_size) + c] * weights.tensor[(r * gradient.columns) + c];
            }
        }
    }
    const int final_size = dimensions * rows * columns;
    for (int i = 0; i < final_size; i++) {
        tensor[i] = clip(tensor[i]);
    }
}

void Tensor::updateWeights(const Tensor &gradient, const Tensor &output) {
    const int gradient_dimensions = output.dimensions;
    float averaged_gradient[columns * rows];
    const int weight_dimension_size = columns * rows;
    memset(averaged_gradient, 0.0f, weight_dimension_size * sizeof(float));
    const int gradient_dimension_size = gradient.columns * gradient.rows;
    const int output_dimension_size = output.columns * output.rows;
    for (int d = 0; d < gradient_dimensions; ++d) {
        const int gradient_index = d * gradient_dimension_size;
        const int output_index = d * output_dimension_size;
        for (int gradient_x = 0; gradient_x < gradient.columns; gradient_x++) {
            for (int output_x = 0; output_x < output.columns; output_x++) {
                averaged_gradient[(output_x * columns) + gradient_x] += gradient.tensor[gradient_index + gradient_x] * output.tensor[output_index + output_x];
            }
        }
    }
    for (int i = 0; i < weight_dimension_size; ++i) {
        tensor[i] += (averaged_gradient[i] / gradient_dimensions) * learning_rate;
    }
}

float Tensor::clip(float x) {
    float value = roundf(x * 100000) / 100000;
    return std::max(-0.1f, std::min(value, 0.1f));
}

void Tensor::resetTensor() {
    memset(tensor, 0.0f, dimensions * rows * columns * sizeof(float));
}

void Tensor::setData(float *tensor) {
    std::memcpy(this->tensor, tensor, dimensions * rows * columns * sizeof(float));
}

const float * Tensor::returnData() const {
    return tensor;
}

void Tensor::setActiveDimension(int batch_size) {
    this->active_dimensions = batch_size;
}

const int Tensor::returnActiveDimension() const {
    return this->active_dimensions;
}

void Tensor::assignRandomValues() {
    int matrixSize = rows * columns;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1,0.1);
    for (int i = 0; i < matrixSize; ++i) {
        tensor[i] = dis(gen);
    }
}

const float Tensor::sumTheSquares() const {
    float finalValue = 0.0f;
    const int number_of_elements = dimensions * rows * columns;
    for (int i = 0; i < number_of_elements; i++) {
        finalValue += std::pow(tensor[i],2);
    }
    return (0.001* finalValue) / (10 * dimensions * rows * columns);
}

void Tensor::updateTensor(float *new_tensor) {
    std::memcpy(this->tensor, new_tensor, dimensions * rows * columns * sizeof(float));
}

void Tensor::print() const {
    std::cout<<std::endl;
    for (int d = 0; d < dimensions; ++d) {
        int dimension_size = rows * columns * d;
        std::cout<<"[";
        for (int i = 0; i < rows; ++i) {
            std::cout<<"[";
            for (int j = 0; j < columns; ++j) {
                std::cout<<tensor[dimension_size + (j + (i * columns))];
                if (j < columns - 1) {
                    std::cout<<",";
                }
            }
            std::cout<<"]";
            if (i < rows - 1) {
                std::cout<<std::endl;
            }
        }
    }
    std::cout<<"]"<<std::endl;
}

void Tensor::printShape() const {
    std::cout<<"("<<dimensions<<","<<rows<<","<<columns<<")"<<std::endl;
}

std::vector<int> Tensor::shape() const{
    std::vector<int> shape_array(3);
    shape_array[0] = dimensions;
    shape_array[1] = rows;
    shape_array[2] = columns;
    return shape_array;
}