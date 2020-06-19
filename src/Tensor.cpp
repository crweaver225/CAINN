#include "Tensor.h"

float Tensor::learning_rate = 0.1f;

Tensor::Tensor(const int rows, const int columns) {
  //  std::cout<<"Tensor constructor with "<<rows<<" rows and "<<columns<<" columns called"<<std::endl;
    this->dimensions = 1;
    this->rows = rows;
    this->columns = columns;
    this->tensor = new float[rows * columns];
    std::memset(this->tensor, 0.0f, rows * columns * sizeof(float));
}

Tensor::Tensor(const int dimensions, const int rows, const int columns) {
  //  std::cout<<"Tensor constructor with dimensions, rows, and columns called"<<std::endl;
    this->dimensions = dimensions;
    this->rows = rows;
    this->columns = columns;
    this->tensor = new float[dimensions * rows * columns];
    std::memset(this->tensor, 0.0f, dimensions * rows * columns * sizeof(float));
}

Tensor::Tensor(const int rows, const int columns, float *tensor) {
   // std::cout<<"Tensor constructor with rows, columns, and data called"<<std::endl;
    this->dimensions = 1;
    this->rows = rows;
    this->columns = columns;
    this->tensor = tensor;
}

Tensor::Tensor(const int dimensions, const int rows, const int columns, float *tensor) {
   // std::cout<<"Tensor constructor with dimensions, rows, columns, and data called"<<std::endl;
    //std::cout<<dimensions<<","<<rows<<","<<columns<<std::endl;
    this->dimensions = dimensions;
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
    this->rows = otherTensor.rows;
    this->columns = otherTensor.columns;
    this->tensor = new float[dimensions * rows * columns];
    *this->tensor = *(otherTensor.returnData());
}

Tensor& Tensor::operator = (const Tensor &otherTensor) {
   // std::cout<<"Tensor copy assignment operator called"<<std::endl;
    if (this == &otherTensor) { return *this; }
    this->dimensions = otherTensor.dimensions;
    this->rows = otherTensor.rows;
    this->columns = otherTensor.columns;
    this->tensor = new float[dimensions * rows * columns];
    *this->tensor = *(otherTensor.tensor);
    return *this;
}

Tensor::Tensor(Tensor &&otherTensor) {
   // std::cout<<"Tensor move constructor called"<<std::endl;
    this->dimensions = otherTensor.dimensions;
    this->rows = otherTensor.rows;
    this->columns = otherTensor.columns;
    this->tensor = otherTensor.tensor;
    otherTensor.tensor = nullptr;
}

Tensor& Tensor::operator = (Tensor &&otherTensor) {
   // std::cout<<"Tensor move assignment operator called"<<std::endl;
    if (this == &otherTensor) { return *this; }
    this->dimensions = otherTensor.dimensions;
    this->rows = otherTensor.rows;
    this->columns = otherTensor.columns;
    this->tensor = otherTensor.tensor;
    otherTensor.tensor = nullptr;
    return *this;
}

template<typename a_f>
Tensor* Tensor::matmul(Tensor &tensor, float *bias, a_f af) const {

   // print();
   // std::cout<<" * ";
    //tensor.print();

    int dimension_size = this->rows * this->columns;
    int product_dimension_size = this->rows * tensor.columns;
    float *product = new float[dimensions * product_dimension_size];
    memset(product, 0.0f, dimensions * product_dimension_size * sizeof(float));
    for (int d = 0; d < dimensions; ++d) {
        int i_d = d * dimension_size;
        int o_d = d * product_dimension_size;
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->columns; ++j) {
                for (int z = 0; z < tensor.columns; ++z) {
                    product[o_d + (i * (tensor.columns) + z)] += this->tensor[i_d + ((i * this->columns) + j)] * tensor.tensor[(j * tensor.columns) + z];
                }
            }
        }
    }

   // std::cout<<" retsults: ";
   // Tensor temp = Tensor(dimensions, this->rows, tensor.columns, product);
   // temp.print();

    for (int d = 0; d < dimensions; ++d) {
        int dimension_location = d * product_dimension_size;
        for (int i = 0; i < product_dimension_size; ++i) {
           // std::cout<<"value before af "<<product[i + dimension_location]<<" + bias: "<<bias[i]<<" = "<< product[i + dimension_location] + bias[i]<<" through af = "<<af(product[i + dimension_location] + bias[i])<<std::endl;
            product[i + dimension_location] = af(product[i + dimension_location] + bias[i]);
        }
    }
    Tensor *productTensor = new Tensor(dimensions, this->rows, tensor.columns, product);
    
  //  std::cout<<"after activation function: ";
  //  productTensor->print();

    return productTensor;
}
template Tensor* Tensor::matmul<float (*)(float)>(Tensor&, float*, float (*)(float)) const;

template<typename a_fd>
void Tensor::applyDerivative(const Tensor& output, a_fd afd) {

  //  std::cout<<"output before derivative: ";
   // print();

    const int number_of_elements = dimensions * rows * columns;
    for (int i = 0; i < number_of_elements; ++i) {
       // std::cout<<"output value was: "<<output.tensor[i]<<" its derivative was: "<<afd(output.tensor[i])<<std::endl;
        tensor[i] = afd(output.tensor[i]) * tensor[i];
    }

   // std::cout<<"output after derivative: ";
   // print();
}
template void Tensor::applyDerivative<float (*)(float)>(const Tensor&, float (*)(float));

void Tensor::updateGradients(const Tensor &gradient, const Tensor &weights) {
    const int gradient_dimension_size = columns * rows;
    const int previous_gradient_d_size = gradient.columns * gradient.rows;
    for (int d = 0; d < dimensions; ++d) {
        for (int r = 0; r < weights.rows; ++r) {
            for (int c = 0; c < gradient.columns; ++c) {
               // std::cout<<"the gradient value: "<<gradient.tensor[(d * previous_gradient_d_size) + c]<<", the weight value: "<<weights.tensor[(r * gradient.columns) + c]<<std::endl;
                tensor[(d * gradient_dimension_size) + r] += clip(gradient.tensor[(d * previous_gradient_d_size) + c] * weights.tensor[(r * gradient.columns) + c]);
              //  std::cout<<"gradient value after the update: "<<tensor[(d * gradient_dimension_size) + r]<<std::endl;
            }
        }
    }
}

void Tensor::updateWeights(const Tensor &gradient, const Tensor &output) {

  //  std::cout<<"-updating weights-"<<std::endl;
  //  std::cout<<"old weight: ";
   // print();

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

              //  std::cout<<"The output value was: "<<output.tensor[output_index + output_x]<<" The gradient value was: "<<gradient.tensor[gradient_index + gradient_x];
              //  std::cout<<" results: "<<gradient.tensor[gradient_index + gradient_x] * output.tensor[output_index + output_x]<<std::endl;
                averaged_gradient[(output_x * columns) + gradient_x] += gradient.tensor[gradient_index + gradient_x] * output.tensor[output_index + output_x];
               // std::cout<<"Averaged gradient at index: "<<(output_x * columns) + gradient_x<<" updated to: "<<averaged_gradient[(output_x * columns) + gradient_x]<<std::endl;
            }
        }
    }
    for (int i = 0; i < weight_dimension_size; ++i) {
        tensor[i] += (averaged_gradient[i] / gradient_dimensions) * learning_rate;
    }

  //  std::cout<<"new weight value: ";
  //  print();
   // std::cout<<"-ending updating weights-"<<std::endl;
}

float Tensor::clip(float x) {
    float value = roundf(x * 100000) / 100000;
    return std::max(-10.0f, std::min(value, 10.0f));
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
    return (0.001 * finalValue) / (10 * dimensions * rows * columns);
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

std::vector<int> Tensor::shape(){
    std::vector<int> shape_array(3);
    shape_array[0] = dimensions;
    shape_array[1] = rows;
    shape_array[2] = columns;
    return shape_array;
}