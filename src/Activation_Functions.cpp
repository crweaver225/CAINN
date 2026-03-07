#include <math.h>
#include "Activation_Functions.h"
#include <iostream>

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
        x[location + i] = std::max(0.0f, x[location + i] + bias[i]);
    }
}

void Activation_Functions::relu(float* x, int location, int size) {
    for (int i = 0; i < size; i++) {
        x[location + i] = std::max(0.0f, x[location + i]);
    }
}

void Activation_Functions::relu_convolution(float* x, float* bias, int batch, int channels, int height, int width) {
    int idx = 0;
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < channels; oc++) {
            float bval = bias[oc];
            for (int y = 0; y < height; y++) {
                for (int xw = 0; xw < width; xw++) {
                    x[idx] = std::max(0.0f, x[idx] + bval);
                    idx++;
                }
            }
        }
    }
}

void Activation_Functions::leaky_relu(float* x, int location, int size) {
    for (int i = 0; i < size; i++) {
        x[location + i] = std::max(x[location + i] * 0.01f, x[location + i]);
    }
}

void Activation_Functions::leaky_relu(float* x, float *bias, int location, int size) {
    for (int i = 0; i < size; i++) {
        x[location + i] = std::max((x[location + i] + bias[i]) * 0.01f, x[location + i] + bias[i]);
    }
}

void Activation_Functions::softmax(float* x, float *bias, int location, int size) {
    float max_value = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < size; i++) {
        max_value = std::max(max_value, x[location + i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[location + i] = std::exp(x[location + i] - max_value);
        sum += x[location + i];
    }

    for (int i = 0; i < size; i++) {
        x[location + i] /= sum;
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
        derivative[i] *= ((output[i] > 0.0f) ? 1.0f : 0.0f);
    }
}

void Activation_Functions::leaky_relu_d(float* output, float* derivative, int size) {
    for (int i = 0; i < size; i++) {
       derivative[i] *= ((output[i] > 0.0f) ? 1.0f : 0.001f);
    }
}

void Activation_Functions::pass_d(float* output, float* derivative, int size) {}

