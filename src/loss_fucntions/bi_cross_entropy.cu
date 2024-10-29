/**
 * @file bi_cross_entropy.cu
 * @brief Implementation of the Binary Cross Entropy Loss function.
 *
 * The Binary Cross Entropy (BCE) Loss function is used for binary classification tasks.
 * It measures the performance of a classification model whose output is a probability value between 0 and 1.
 * The BCE loss increases as the predicted probability diverges from the actual label.
 *
 * @note
 * Use the Binary Cross Entropy Loss function when:
 * - You are dealing with a binary classification problem.
 * - Your model outputs probabilities (values between 0 and 1).
 * - You want to penalize predictions that are far from the actual labels.
 */


#include <iostream>
#include "cuda_runtime.h"

__global__ void binaryCrossEntropyLossKernel(float *y_pred, float *y_true, float *output, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        output[idx] = -y_true[idx] * log(y_pred[idx]) - (1 - y_true[idx]) * log(1 - y_pred[idx]);
    }
}