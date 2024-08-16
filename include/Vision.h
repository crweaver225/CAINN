#ifndef Vision_H_
#define Vision_H_

#include "Tensor.h"
#include "Thread_Pool.h"
#include <thread>

class Tensor;

namespace Vision {
 
    static void ConvolveInner(const Tensor &input, Tensor &Output, const Tensor &kernel, int stride, const int d, const int n_d);
    static void InnerUpdateKernel(const Tensor &input, Tensor &kernel, const Tensor &gradient, int stride, const int d, const int n_d);
    static void InnerBackward(const Tensor &gradient, const Tensor &kernel, Tensor &_gradient, int stride, const int d, const int n_d);
    static void InnerFindMax(const Tensor &input, Tensor &output, int filter_size, int stride, std::vector<unsigned int> &maxpool_indexes, const int d, const int n_d);

    
    void Convolve(const Tensor &input, Tensor &Output, const Tensor &kernel, Thread_Pool &threadPool, int stride);
    void UpdateKernel(const Tensor &input, Tensor &weight, const Tensor &gradient, Thread_Pool& threadPool, int stride);
    void Backward(const Tensor &gradient, const Tensor &kernel, Tensor &_gradient, Thread_Pool& threadPool, int stride);

    
    void FindMax(const Tensor &input, Tensor &output, Thread_Pool &threadPool, int filter_size, int stride, std::vector<unsigned int> &maxpool_indexes);
    
}

#endif /*Vision_H_*/