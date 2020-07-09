#include <math.h>
#include <cmath>
#include <iostream>

#ifndef LOSS_H_
#define LOSS_H_

enum Loss {MSE, ASE, CrossEntropy};

#endif /*LOSS_H_*/

namespace Loss_Function {
    float mse(const float *output, float *target, int location, int size);
    float ase(const float *output, float *target, int location, int size);
    float crossentropy(const float *output, float *target, int location, int size);
};