#include "Loss.h"

float Loss_Function::mse(const float *output, float *target, int location, int size) {
    float return_loss = 0.0f;
    for (int i = 0; i < size; i++) {
        return_loss += pow(target[i] - output[location + i], 2);
    }
    return return_loss;
}
float Loss_Function::ase(const float *output, float *target, int location, int size) {
    float return_loss = 0.0f;
    for (int i = 0; i < size; i++) {
        return_loss += fabs(target[i] - output[location + i]);
    }
    return return_loss;
}
float Loss_Function::crossentropy(const float *output, float *target, int location, int size) {
    float return_loss = 0.0f;
    float loss_min = std::numeric_limits<float>::min();
    for (int i = 0; i < size; ++i) {
        return_loss -= (target[i] * log10(output[location + i])) + ((1 - target[i]) * (log10(std::max(1 - output[location + i], loss_min))));
    }
    return return_loss;
}