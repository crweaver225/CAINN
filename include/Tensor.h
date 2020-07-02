#include <memory>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstring>
#include <string.h>
#include "Activation_Function.h"
#include "Activation_Functions.h"
#include "Loss.h"

#ifndef TENSOR_H_
#define TENSOR_H_

class Tensor {

private:
    int dimensions;
    int active_dimensions;
    int rows;
    int columns;
    float *tensor;
    float clip(float x);

public:
    Tensor(const int rows, const int columns);
    Tensor(const int dimensions, const int rows, const int columns);
    Tensor(const int rows, const int columns, float *tensor);
    Tensor(const int dimensions, const int rows, const int columns, float *tensor);
    ~Tensor();
    Tensor(const Tensor &Tensor);
    Tensor& operator = (const Tensor &tensor);
    Tensor(Tensor &&tensor);
    Tensor& operator = (Tensor &&tensor);

    static float learning_rate;
    
    void setData(float *tensor);
    void assignRandomValues();

    const float * returnData() const;
    const float sumTheSquares() const;

    template<typename a_f>
    void matmul(const Tensor &m1, Tensor &m2, float *bias, a_f af);

    void resetTensor();
    void updateTensor(float *new_tensor);
    void updateGradients(const Tensor &gradient, const Tensor &weights);
    void updateWeights(const Tensor &gradient, const Tensor &output);
    
    template<typename a_fd>
    void applyDerivative(const Tensor& output, a_fd afd);

    void setActiveDimension(int batch_size);
    const int returnActiveDimension() const;

    void print() const;
    void printShape() const;
    std::vector<int> shape() const;
};

#endif /* TENSOR_H_ */
