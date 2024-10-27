#include "activation.h"

CUDA_DEVICE float relu(float x){
    return x > 0 ? x : 0;
}

CUDA_GLOBAL void relu(float* input, float* output, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = relu(input[idx]);
    }
}