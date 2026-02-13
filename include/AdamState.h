#ifndef ADAMSTATE_H
#define ADAMSTATE_H

#include <vector>
#include <iostream>

struct AdamState {
    std::vector<float> m; // first moment
    std::vector<float> v; // second moment
    std::vector<float> gradientAccumulation;
    size_t t = 0;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    AdamState () = delete;
    explicit AdamState (size_t layer_size) : m(layer_size, 0.0f),
        v(layer_size, 0.0f),
        gradientAccumulation(layer_size, 0.0f)
    {}
};

#endif