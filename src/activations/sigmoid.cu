//sigmoid activation function in cuda optimised kernel
#include "activation.h"
#include <cudaruntime.h>
#include <iostream>

CUDA_DEVICE float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

CUDA_GLOBAL void sigmoid(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sigmoid(input[idx]);
    }
}