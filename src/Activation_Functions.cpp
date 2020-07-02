#include <math.h>
#include "Activation_Functions.h"
#include<iostream>

void Activation_Functions::sigmoid(float* x, float *bias, int location, int size) {
    for (int i = 0; i < size; i++) {
        float sig = 1 / (1 + exp(-(x[location + i]  + bias[i])));
        sig = std::min(sig, 0.999f);
        sig = std::max(sig, 0.001f);
        x[location + i] = sig;
    }
}

void Activation_Functions::relu(float* x, float *bias, int location, int size) {
    for (int i = 0; i < size; i++) {
        x[location + i] = std::max((x[location + i] + bias[i]) * 0.001f, x[location + i] + bias[i]);
    }
}

void Activation_Functions::softmax(float* x, float *bias, int location, int size) {
    float max_value = std::numeric_limits<float>::min();
    for (int i = 0; i < size; i++) {
        if (x[location + i] > max_value) {
            max_value = x[location + i];
        }
    }

    float y_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[location + i] = std::exp(x[location + i] - max_value);
        y_sum += x[location + i];
    }
    for (int i = 0; i < size; i++) {
        x[location + i] = x[location + i] / y_sum;
        x[location + i] = std::min(0.99f, x[location + i]);
    }
}
 
void Activation_Functions::pass(float* x, float *bias, int location, int size) { }

void Activation_Functions::sigmoid_d(float* output, float* derivative, int size) {
    for (int i = 0; i < size; i++) {
        derivative[i] = (output[i] * (1.0f - output[i])) * derivative[i];
    }
}

void Activation_Functions::relu_d(float* output, float* derivative, int size) {
    for (int i = 0; i < size; i++) {
        derivative[i] = ((output[i] > 0.0f) ? 1.0f : 0.001f) * derivative[i];
    }
}

void Activation_Functions::softmax_d(float* output, float* derivative, int size) {
    float* output_derivative = new float[size];
    for (int i = 0; i < size; i++) {
        float sum = 0.0f;
        const float neg_sft_i = -output[i];
        for (int j = 0; j < size; j++) {
            float mul = derivative[j] * output[j] * neg_sft_i;
            sum += mul;
        }
        output_derivative[i] = sum;
    }
    for (int i = 0; i < size; i++) {
        output_derivative[i] += output[i] * derivative[i];
    } 
    *derivative = *output_derivative;
}

void Activation_Functions::pass_d(float* output, float* derivative, int size) {}

