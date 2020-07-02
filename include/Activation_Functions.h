#include <limits>
#include "Activation_Function.h"

namespace Activation_Functions {

    void sigmoid(float* x, float *bias, int location, int size);
    void relu(float* x, float *bias, int location, int size);
    void softmax(float* x, float* bias, int location, int size);
    void pass(float* x, float *bias, int location, int size);
    void sigmoid_d(float* output, float* derivative, int size);
    void relu_d(float* output, float* derivative, int size);
    void softmax_d(float* output, float* derivative, int size);
    void pass_d(float* output, float* derivative, int size);
    
};