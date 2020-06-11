#include <memory>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstring>
#include "Activation_Function.h"
#include "Activation_Functions.h"

#ifndef TENSOR_H_
#define TENSOR_H_

class Tensor {

private:
    int dimensions;
    int rows;
    int columns;
    float *tensor;
    Activation_Function activation_function;
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
    
    void setActivation_Function(Activation_Function activation_function);
    void setData(float *tensor);
    void assignRandomValues();

    std::shared_ptr<float> returnData();

    template<typename a_f>
    Tensor* matmul(Tensor &tensor, float *bias, a_f af) const;

    void resetTensor();
    void updateTensor(float *new_tensor);
    void updateGradients(const Tensor &gradient, const Tensor &weights);
    void updateWeights(const Tensor &gradient, const Tensor &output);
    
    template<typename a_fd>
    void applyDerivative(const Tensor& output, a_fd afd);

    void print() const;
    void printShape() const;
    std::unique_ptr<int> shape();
};

#endif /* TENSOR_H_ */
