#include <iostream>
#include "cuda_runtime.h"

__global__ void mse(float* y_pred, float *y_true, float* output, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        output[idx] = (y_pred[idx] - y_true[idx]) * (y_pred[idx] - y_true[idx]);
    }
}