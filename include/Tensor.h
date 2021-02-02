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
#include <map>

#ifndef TENSOR_H_
#define TENSOR_H_

class Tensor {

private:
    int _dimensions;
    int _activeDimensions;
    int _rows;
    int _columns;
    int _channels;
    float *_tensor;
    float clip(float x);
    
    
    template<typename a_f>
    void MatmulInner(const Tensor &m1, Tensor &m2, float *bias, int d, a_f af);

    void UpdateGradientInner(const Tensor &gradient, const Tensor &weights, int d);
    void UpdateWeightsInner(const Tensor &gradient, const Tensor &output, const int d);

public:
    Tensor(const int rows, const int columns);
    Tensor(const int rows, const int columns, float *tensor);
    Tensor(const int channels, const int rows, const int columns);
    Tensor(const int dimensions, const int channels, const int rows, const int columns);
    Tensor(const int dimensions, const int channels, const int rows, const int columns, float *tensor);
    ~Tensor();
    Tensor(const Tensor &Tensor);
    Tensor& operator = (const Tensor &tensor);
    Tensor(Tensor &&tensor);
    Tensor& operator = (Tensor &&tensor);

    static float _learningRate;

    void clipData();
    
    void updateNeuron(int index, float value);
    void updateNeuron(int batch, int index, float value);
    void SetData(float *tensor);
    void TransferDataFrom(const Tensor *tensor);
    void AssignRandomValues();

    const float * ReturnData() const;
    const float *returnColumn(int column) const;
    const float *returnColumn(int batch, int column) const;
    const float SumTheSquares() const;

    template<typename a_f>
    void Matmul(const Tensor &m1, Tensor &m2, float *bias, a_f af);

    std::vector<int> Maxpool(const Tensor &input, int filter_size, int stride);

    void ResetTensor();
    void UpdateTensor(float *new_tensor);
    void UpdateGradients(const Tensor &gradient, const Tensor &weights);
    void UpdateWeights(const Tensor &gradient, const Tensor &output);

    void flatten();
    void reshape(int channels, int rows, int columns);
    
    template<typename a_fd>
    void ApplyDerivative(const Tensor& output, a_fd afd);

    void SetActiveDimension(int batch_size);
    const int ReturnActiveDimension() const;

    void Print() const;
    void PrintShape() const;
    std::vector<int> Shape() const;
};

#endif /* TENSOR_H_ */
