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
    float loss = 0.0f;
    for (int i = 0; i < size; ++i) {
        float p = std::clamp(output[location + i], 1e-7f, 1.0f - 1e-7f);
        loss -= target[i] * log(p);
    }
    return loss;
}