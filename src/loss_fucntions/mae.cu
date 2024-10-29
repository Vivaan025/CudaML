/**
 * @file mae.cu
 * @brief Implementation of the Mean Absolute Error (MAE) loss function.
 *
 * The Mean Absolute Error (MAE) loss function is used to measure the average magnitude of errors 
 * in a set of predictions, without considering their direction. It is calculated as the average 
 * of the absolute differences between predicted values and actual values.
 *
 * MAE is particularly useful in regression problems where the goal is to predict continuous values. 
 * It is less sensitive to outliers compared to Mean Squared Error (MSE), making it a good choice 
 * when dealing with data that contains outliers.
 *
 * Usage:
 * - Use MAE when you want a loss function that is robust to outliers.
 * - Suitable for regression tasks where the distribution of errors is not Gaussian.
 * - Prefer MAE over MSE when you want to minimize the average absolute error.
 */


#include <iostream>
#include "cuda_runtime.h"

__global__ void maeKernel(float *y_pred, float *y_true, float *output, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        output[idx] = abs(y_pred[idx] - y_true[idx]);
    }
}