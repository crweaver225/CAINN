#include <memory>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstring>
#include <string.h>
#include "Activation_Function.h"
#include "Activation_Functions.h"
#include "Loss.h"
#include <thread>
#include <x86intrin.h>

#ifndef TENSOR_H_
#define TENSOR_H_

class Tensor {

private:
    int _dimensions;
    int _activeDimensions;
    int _rows;
    int _columns;
    float *_tensor;
    float clip(float x);
    

    /*
    template<typename a_f>
    void MatmulInner(const Tensor &m1, Tensor &m2, float *bias, int d, a_f af);
    */
    
    template<typename a_f>
    void MatmulDimension(const Tensor &m1, Tensor &m2, float *bias, int d, a_f af);
    void MatmulInner(const Tensor &m1, Tensor &m2, int a_row, int a_column, int b_row, int b_columm, int dimension);
    
    void UpdateGradientInner(const Tensor &gradient, const Tensor &weights, int d);
    void UpdateWeightsInner(const Tensor &gradient, const Tensor &output, const int d);

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

    static float _learningRate;
    
    void SetData(float *tensor);
    void AssignRandomValues();

    const float * ReturnData() const;
    const float SumTheSquares() const;

    template<typename a_f>
    void Matmul(const Tensor &m1, Tensor &m2, float *bias, a_f af);

    void ResetTensor();
    void UpdateTensor(float *new_tensor);
    void UpdateGradients(const Tensor &gradient, const Tensor &weights);
    void UpdateWeights(const Tensor &gradient, const Tensor &output);
    
    template<typename a_fd>
    void ApplyDerivative(const Tensor& output, a_fd afd);

    void SetActiveDimension(int batch_size);
    const int ReturnActiveDimension() const;

    void Print() const;
    void PrintShape() const;
    std::vector<int> Shape() const;
};

#endif /* TENSOR_H_ */
