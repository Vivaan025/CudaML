#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#else
#define CUDA_DEVICE
#define CUDA_GLOBAL
#endif

CUDA_DEVICE float sigmoid(float x);
CUDA_DEVICE float relu(float x);

CUDA_GLOBAL void sigmoidActivation(float* input, float* output, int size);
CUDA_GLOBAL void reluActivation(float* input, float* output, int size);

#ifdef __cplusplus
}
#endif

#endif 
