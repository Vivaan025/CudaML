//generate docs for binary cross entropy loss function explain when to use it


#include <iostream>
#include "cuda_runtime.h"

__global__ void mseKernel(float* y_pred, float *y_true, float* output, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        output[idx] = (y_pred[idx] - y_true[idx]) * (y_pred[idx] - y_true[idx]);
    }
}

__global__ void reduceSum(float* input, float* result, int size) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared_data[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared_data[0] / size);
    }
}