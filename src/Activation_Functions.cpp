#include <math.h>
#include "Activation_Functions.h"
#include<iostream>

float Activation_Functions::sigmoid(float x) {
    float sig = 1 / (1 + exp(-x));
    sig = std::min(sig, 0.999f);
    sig = std::max(sig, 0.001f);
    return sig;
}

float Activation_Functions::relu(float x) {
    //return std::max(0.0f, x);
   return std::max(x * 0.001f, x);
}
        
float Activation_Functions::pass(float x) {
    return x;    
}

float Activation_Functions::sigmoid_d(float x) {
    return x * (1.0f - x);
    //return 1.0f / (1.0f + exp(-x));
}

float Activation_Functions::relu_d(float x) {
    return (x > 0) ? 1.0f : 0.001f;
}

float Activation_Functions::pass_d(float x) {
    return x;
}
