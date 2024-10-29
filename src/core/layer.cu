#include "../include/layer.h"
#include "../include/activations/activation.h"
#include "cuda_runtime.h"
#include <iostream>

__global__ void forwardKernel(float* input, float* output, float* bias, float* weights, int input_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        int batch_idx = idx / output_size;
        int output_idx = idx % output_size;
        output[idx] = bias[output_idx];
        for (int i = 0; i < input_size; i++) {
            output[idx] += input[batch_idx * input_size + i] * weights[output_idx * input_size + i];
        }
    }
}

void layerForward(float* input, float* output, float* bias, float* weights, int input_size, int output_size, int batch_size) {
    float *d_input, *d_output, *d_bias, *d_weights;
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_bias, output_size * sizeof(float));
    cudaMalloc(&d_weights, output_size * input_size * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (batch_size * output_size + threads - 1) / threads;
    forward<<<blocks, threads>>>(input, output, bias, weights, input_size, output_size, batch_size);

    cudaMemcpy(output, d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bias);
    cudaFree(d_weights);
}