#ifndef TENSOR_H_
#define TENSOR_H_

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
#include <mutex>
#include "Dimensions.h"
#include "Vision.h"
#include "Thread_Pool.h"

class Tensor {

private:
    int _dimensions;
    int _activeDimensions;
    int _rows;
    int _columns;
    int _channels;
    float *_tensor;
    float clip(float x);

    Thread_Pool _threadPool;
    
    template<typename a_f>
    void MatmulInner(const Tensor &m1, Tensor &m2, float *bias, int d, const int n_d, a_f af);

    void UpdateGradientInner(const Tensor &gradient, const Tensor &weights, int d, int n_d);
    void UpdateWeightsInner(const Tensor &gradient, const Tensor &output, const int d, const int n_d);

public:
 
    Tensor(const int rows, const int columns);
    Tensor(const int rows, const int columns, float *tensor);
    Tensor(const int channels, const int rows, const int columns);
    Tensor(const int dimensions, const int channels, const int rows, const int columns);
    Tensor(const int dimensions, const int channels, const int rows, const int columns, float *tensor);
    ~Tensor();
    Tensor(const Tensor &Tensor) noexcept;
    Tensor& operator = (const Tensor &tensor) noexcept;
    Tensor(Tensor &&tensor) noexcept;
    Tensor& operator = (Tensor &&tensor) noexcept;

    static float _learningRate;

    void clipData();

    void optimizeForTraining();
    void optimizeForInference();
    
    void changeNeuron(int index, float value);
    void setNeuron(int index, float value);
    void setNeuron(int batch, int index, float value);
    void SetData(const float *tensor);
    void TransferDataFrom(Tensor const* tensor);
    void AssignRandomValues();

    const float *ReturnData() const;
    const float SumTheSquares() const;

    template<typename a_f>
    void Matmul(const Tensor &m1, Tensor &m2, float *bias, a_f af);
    
    // Convolution functions
    void UpdateKernel(const Tensor &input, const Tensor &gradient, int stride);
    void Convolve(const Tensor &input, const Tensor &kernel, int stride);
    void Backward(const Tensor &gradient, const Tensor &kernel, int stride);

    void Maxpool(const Tensor &input, int filter_size, int stride, std::vector<unsigned int> &maxpool_indexes);

    void ResetTensor();
    void UpdateTensor(float *new_tensor);
    void UpdateGradients(const Tensor &gradient, const Tensor &weights);
    void UpdateWeights(const Tensor &gradient, const Tensor &output);

    void flatten();
    void reshape(int dimension, int channels, int rows, int columns);
    void reshape(int channels, int rows, int columns);
    
    template<typename a_fd>
    void ApplyDerivative(const Tensor& output, a_fd afd);

    void SetActiveDimension(int batch_size);
    const int ReturnActiveDimension() const;

    void Print() const;
    void PrintShape() const;

    const Dimensions dimensions() const;
    const int NumberOfRows() const;
    const int NumberOfColumns() const;
    const int NumberOfChannels() const;
    const int NumberOfElements() const;
    const int NumberOfElementsPerTensor() const;
    const int NumberOfDimensions() const;
};

#endif /* TENSOR_H_ */
